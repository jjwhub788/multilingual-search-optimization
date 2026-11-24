"""Exact-token-budget prioritization on the fixed TourAPI candidate corpus.

All three methods select subsets of the same 68 precomputed NLLB selective
translation documents. Strategy C estimates individual marginal MRR gains
on the existing development queries and maximizes their sum by knapsack.
Development results are not independent evidence of generalization.

With --holdout-file, evaluate the already locked allocations on new queries;
do not recalculate priorities or tune against those query results.
"""

import argparse
import csv
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prepare_large_corpus import (
    OUT as LARGE, PROCESSED, RESULTS, BASE, QUERIES, LANGS, read, write, write_json, sha256, key,
)
from evaluate_large_corpus import rank_condition, keyed

OUT = RESULTS / "tourapi_budget_priority"
FEATURE = RESULTS / "tourapi_full_feature_enrichment"
MT = RESULTS / "tourapi_enrichment_method_comparison"
HY = RESULTS / "tourapi_hybrid_policy"
TARGETS = (*LANGS, "ALL_PLACES")
SEED = 261120
RANDOM_TRIALS = 500
FRACTIONS = (.20, .40, .60, .80)


def load_inputs():
    index = read(LARGE / "corpus_index.csv")
    gold = read(PROCESSED / "tourapi_pilot_details_300.csv")
    base_metadata = json.loads((BASE / "run_metadata.json").read_text())
    assert sha256(PROCESSED / "tourapi_pilot_details_300.csv") == base_metadata["corpus_sha256"]
    old_queries = read(QUERIES / "queries.csv")
    full_index = {key(r): i for i, r in enumerate(index)}
    gold_lookup = {(r["place_id"], r["language"]): i for i, r in enumerate(gold)}
    gold_positions = np.array([full_index[key(r)] for r in gold])
    extra_positions = np.array([int(r["embedding_row"]) for r in read(LARGE / "extra_detail_index.csv")])
    base = np.load(BASE / "document_embeddings.npy", allow_pickle=False)
    select = np.load(MT / "document_embeddings_selective_mt_68.npy", allow_pickle=False)
    changes = defaultdict(list)
    for r in read(FEATURE / "description_changes.csv"):
        changes[r["place_id"], r["language"]].append(r)
    workloads = {(r["place_id"], r["language"]): int(r["selective_output_tokens"])
                 for r in read(HY / "policy_document_decisions.csv") if r["setting"] == "NLLB_68"}
    assert set(changes) == set(workloads) and len(changes) == 68
    pool = []
    for j, k in enumerate(sorted(changes)):
        i = gold_lookup[k]
        missing = sum(r["original_coverage"] == "MISSING" for r in changes[k])
        partial = sum(r["original_coverage"] == "PARTIAL" for r in changes[k])
        assert missing+partial == len(changes[k])
        pool.append(dict(pool_index=j, place_id=k[0], language=k[1], title=gold[i]["title"],
                         gold_row=i, full_row=int(gold_positions[i]), output_tokens=workloads[k],
                         missing_intents=missing, partial_intents=partial, severity=2*missing+partial))
    assert sum(r["output_tokens"] for r in pool) == 3457
    assert {i for i in range(300) if not np.array_equal(base[i], select[i])} == {r["gold_row"] for r in pool}
    signature = {str(p.relative_to(PROCESSED.parent.parent)): sha256(p) for p in (
        LARGE / "corpus_index.csv", LARGE / "metadata_embeddings.npy",
        LARGE / "extra_detail_embeddings.npy", LARGE / "extra_detail_index.csv",
        BASE / "document_embeddings.npy", MT / "document_embeddings_selective_mt_68.npy",
        QUERIES / "queries.csv", QUERIES / "query_embeddings.npy",
        FEATURE / "description_changes.csv", HY / "policy_document_decisions.csv",
    )}
    return dict(index=index, gold=gold, old_queries=old_queries, pool=pool, signature=signature,
                gold_positions=gold_positions, extra_positions=extra_positions, base=base, select=select)


class ScopeRanker:
    """Exact ranking under arbitrary subsets, without rescoring fixed records.

    Candidate IDs and ties match the full-corpus evaluator. Only linked
    groups with changed documents are recomputed. Fixed-group rank counts
    for each possible target-group score are precomputed per query.
    """

    def __init__(self, scores, new_scores, index, queries, pool, target):
        doc_positions = sorted(
            (i for i, r in enumerate(index) if target == "ALL_PLACES" or r["language"] == target),
            key=lambda i: (index[i]["candidate_group_id"], index[i]["language"],
                           index[i]["source_contentid"]),
        )
        groups = [index[i]["candidate_group_id"] for i in doc_positions]
        starts = np.array([i for i, group in enumerate(groups) if i == 0 or group != groups[i-1]])
        ends = np.r_[starts[1:], len(doc_positions)]
        group_index = {groups[start]: j for j, start in enumerate(starts)}
        self.group_count = len(starts)
        target_positions = np.array([group_index["place:"+q["place_id"]] for q in queries])
        all_group_scores = np.maximum.reduceat(scores[:, doc_positions], starts, axis=1)
        self.query_index = np.arange(len(queries))
        self.target_base = all_group_scores[self.query_index, target_positions]
        pool_lookup = {r["full_row"]: r["pool_index"] for r in pool}
        changed_groups = defaultdict(list)
        for r in pool:
            i = r["full_row"]
            if target == "ALL_PLACES" or index[i]["language"] == target:
                changed_groups[group_index[index[i]["candidate_group_id"]]].append(r["pool_index"])
        # The Korean-only candidate pool has no foreign-language interventions.
        # Its ranks are invariant across every translation allocation.
        self.fixed_ranks = None
        if not changed_groups:
            ahead = (all_group_scores > self.target_base[:, None]) | (
                (all_group_scores == self.target_base[:, None])
                & (np.arange(self.group_count)[None, :] < target_positions[:, None])
            )
            self.fixed_ranks = 1+np.count_nonzero(ahead, axis=1)
            return
        self.affected_positions = np.array(sorted(changed_groups), dtype=int)
        self.affected_members = [sorted(changed_groups[i]) for i in self.affected_positions]
        assert max(map(len, self.affected_members)) <= 2
        self.active_pool_indices = np.array([i for group in self.affected_members for i in group], dtype=int)
        self.active_starts = np.array([0]+list(np.cumsum([len(m) for m in self.affected_members[:-1]])),
                                      dtype=int)
        active_docs = [pool[i]["full_row"] for i in self.active_pool_indices]
        self.old_values = scores[:, active_docs]
        self.new_values = new_scores[:, self.active_pool_indices]
        self.fixed_max = np.full((len(queries), len(self.affected_positions)), -np.inf, dtype=np.float32)
        for j, pos in enumerate(self.affected_positions):
            members = [i for i in doc_positions[starts[pos]:ends[pos]] if i not in pool_lookup]
            if members:
                self.fixed_max[:, j] = scores[:, members].max(axis=1)
        affected_lookup = {pos: j for j, pos in enumerate(self.affected_positions)}
        self.target_affected = np.array([affected_lookup.get(pos, -1) for pos in target_positions])
        self.target_members = np.full((len(queries), 2), -1, dtype=int)
        fixed_positions = np.array([i for i in range(self.group_count) if i not in affected_lookup])
        self.fixed_counts = np.zeros((len(queries), 4), dtype=np.int32)
        self.tie_ahead = self.affected_positions[None, :] < target_positions[:, None]
        for q in range(len(queries)):
            fixed_scores = all_group_scores[q, fixed_positions]
            fixed_ties = fixed_positions < target_positions[q]
            affected = self.target_affected[q]
            if affected == -1:
                values = [self.target_base[q]]*4
            else:
                members = self.affected_members[affected]
                self.target_members[q, :len(members)] = members
                values = []
                for code in range(4):
                    candidates = [self.fixed_max[q, affected]]
                    for j, member in enumerate(members):
                        candidates.append(new_scores[q, member] if code & (1 << j)
                                          else scores[q, pool[member]["full_row"]])
                    values.append(max(candidates))
            for code, threshold in enumerate(values):
                self.fixed_counts[q, code] = np.count_nonzero(
                    (fixed_scores > threshold) | ((fixed_scores == threshold) & fixed_ties)
                )
        del all_group_scores

    def ranks(self, selected):
        if self.fixed_ranks is not None:
            return self.fixed_ranks
        values = np.where(selected[self.active_pool_indices][None, :], self.new_values, self.old_values)
        group_scores = np.maximum(self.fixed_max, np.maximum.reduceat(values, self.active_starts, axis=1))
        threshold = self.target_base.copy()
        has_target = self.target_affected >= 0
        threshold[has_target] = group_scores[self.query_index[has_target], self.target_affected[has_target]]
        padded = np.r_[selected, False]
        codes = padded[self.target_members[:, 0]].astype(np.int32) + 2*padded[self.target_members[:, 1]]
        ahead = (group_scores > threshold[:, None]) | (
            (group_scores == threshold[:, None]) & self.tie_ahead
        )
        return 1+self.fixed_counts[self.query_index, codes]+np.count_nonzero(ahead, axis=1)


def build_rankers(inputs, queries, query_vec):
    scores = query_vec @ np.load(LARGE / "metadata_embeddings.npy", mmap_mode="r", allow_pickle=False).T
    scores[:, inputs["gold_positions"]] = query_vec @ inputs["base"].T
    scores[:, inputs["extra_positions"]] = query_vec @ np.load(
        LARGE / "extra_detail_embeddings.npy", allow_pickle=False
    ).T
    updated = query_vec @ inputs["select"].T
    new_scores = updated[:, [r["gold_row"] for r in inputs["pool"]]]
    rankers = [ScopeRanker(scores, new_scores, inputs["index"], queries, inputs["pool"], lang)
               for lang in TARGETS]
    return rankers, scores, new_scores


def rank_subset(rankers, selected):
    return np.column_stack([r.ranks(selected) for r in rankers])


def scope_masks(queries, pool, development):
    langs = np.array([q["query_language"] for q in queries])
    targets = np.array(TARGETS)
    same = langs[:, None] == targets[None, :]
    mixed = np.broadcast_to(targets[None, :] == "ALL_PLACES", same.shape).copy()
    pool_keys = {(r["place_id"], r["language"]) for r in pool}
    targeted_docs = np.array([[(q["place_id"], lang) in pool_keys for lang in TARGETS] for q in queries])
    masks = {"MIXED": mixed, "SAME_LANGUAGE": same, "CROSS_LANGUAGE": ~same & ~mixed,
             "TARGETED_DOCUMENTS": targeted_docs}
    if development:
        reference = read(FEATURE / "rank_comparison.csv")
        gaps = {(r["query_id"], r["target_language"]) for r in reference if r["gap_target"] == "YES"}
        targeted = np.array([[(q["query_id"], lang) in gaps for lang in TARGETS] for q in queries])
        assert targeted.sum() == 219
        masks = {"GAP_TARGETS": targeted, "GAP_TARGETS_CROSS": targeted & ~same, **masks}
    return masks


def metrics(ranks, masks):
    out = {}
    for scope, mask in masks.items():
        a = ranks[mask]
        assert len(a) > 0
        out[scope] = dict(evaluations=len(a), recall_at_1=float((a == 1).mean()),
                          recall_at_5=float((a <= 5).mean()), recall_at_10=float((a <= 10).mean()),
                          mrr=float((1/a).mean()), median_rank=float(np.median(a)))
    return out


def counts_by_suffix(costs):
    """Count feasible exact-cost subsets (arbitrary-precision Python integers)."""
    total = sum(costs)
    n = len(costs)
    table = [[0]*(total+1) for _ in range(n+1)]
    table[n][0] = 1
    for i in range(n-1, -1, -1):
        c, nxt = costs[i], table[i+1]
        row = nxt.copy()
        for spent in range(c, total+1):
            row[spent] += nxt[spent-c]
        table[i] = row
    return table


def uniform_exact_subset(costs, counts, budget, rng):
    chosen = []
    remaining = budget
    for i, c in enumerate(costs):
        skip_count = counts[i+1][remaining]
        take_count = counts[i+1][remaining-c] if remaining >= c else 0
        assert skip_count+take_count > 0
        if rng.randrange(skip_count+take_count) >= skip_count:
            chosen.append(i)
            remaining -= c
    assert remaining == 0 and sum(costs[i] for i in chosen) == budget
    return chosen


def optimum_exact_subset(costs, utility, budget):
    """0–1 exact-cost knapsack using fixed development-derived item values."""
    n = len(costs)
    dp = np.full((n+1, budget+1), -np.inf)
    dp[n, 0] = 0
    take = np.zeros((n, budget+1), dtype=bool)
    for i in range(n-1, -1, -1):
        c = costs[i]
        dp[i] = dp[i+1]
        if c <= budget:
            alt = dp[i+1, :budget+1-c]+utility[i]
            # Deterministic skip on exact utility ties.
            improve = alt > dp[i, c:]+1e-14
            dp[i, c:][improve] = alt[improve]
            take[i, c:] = improve
    assert np.isfinite(dp[0, budget])
    remain = budget
    chosen = []
    for i, c in enumerate(costs):
        if take[i, remain]:
            chosen.append(i)
            remain -= c
    assert remain == 0
    assert np.isclose(sum(utility[i] for i in chosen), dp[0, budget], atol=1e-10)
    return chosen


def mask_of(n, chosen):
    mask = np.zeros(n, dtype=bool)
    mask[chosen] = True
    return mask


def verify_rankers(inputs, rankers, scores, new_scores, queries):
    n = len(inputs["pool"])
    refs = [r for r in read(LARGE / "method_strategy_rank_results.csv") if r["setting"] == "MAIN_68_AND_200"]
    positions = {q["query_id"]: i for i, q in enumerate(queries)}
    for method, selected in (("BASELINE", np.zeros(n, dtype=bool)), ("SELECTIVE_NLLB_68", np.ones(n, dtype=bool))):
        actual = rank_subset(rankers, selected)
        saved = [r for r in refs if r["condition"] == method]
        assert len(saved) == 2400
        assert all(actual[positions[r["query_id"]], TARGETS.index(r["target_language"])] == int(r["rank"])
                   for r in saved)
    # Also check a nontrivial subset against the original full-matrix rank evaluator.
    sample = list(range(0, n, 3))
    selected = mask_of(n, sample)
    changed = scores.copy()
    changed[:, [inputs["pool"][i]["full_row"] for i in sample]] = new_scores[:, sample]
    dummy_gaps = {(q["query_id"], lang): "NO" for q in queries for lang in TARGETS}
    full = rank_condition(changed, inputs["index"], queries, list(range(len(inputs["index"]))),
                          "CHECK", "FULL_18038", dummy_gaps, {})
    fast = rank_subset(rankers, selected)
    assert all(fast[positions[r["query_id"]], TARGETS.index(r["target_language"])] == r["rank"] for r in full)
    print("PASS: fast subset ranks match full-corpus evaluation (baseline, all documents, partial subset)", flush=True)


def evaluate_allocations(plan, rankers, queries, masks, output, pool, stage):
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    n = len(pool)
    costs = [r["output_tokens"] for r in pool]
    total = sum(costs)
    summaries, trial_rows, allocation_rows, rank_rows = [], [], [], []

    def record(budget, method, chosen):
        assert sum(costs[i] for i in chosen) == budget
        ranks = rank_subset(rankers, mask_of(n, chosen))
        met = metrics(ranks, masks)
        for i, q in enumerate(queries):
            for j, target in enumerate(TARGETS):
                rank_rows.append(dict(stage=stage, budget_tokens=budget, strategy=method,
                                      query_id=q["query_id"], place_id=q["place_id"],
                                      query_language=q["query_language"], target_language=target,
                                      rank=int(ranks[i, j])))
        return met

    def row(budget, strategy, scope, scores, doc_count, pct="", r1p05="", r1p95="", mrrp05="", mrrp95=""):
        return dict(stage=stage, budget_tokens=budget, budget_percent=round(100*budget/total, 3),
                    strategy=strategy, scope=scope, selected_documents=doc_count, **scores,
                    random_mrr_percentile=pct, random_r1_p05=r1p05, random_r1_p95=r1p95,
                    random_mrr_p05=mrrp05, random_mrr_p95=mrrp95)

    for tokens, chosen in ((0, []), (total, list(range(n)))):
        method = "BASELINE" if not tokens else "ALL_68"
        for scope, m in record(tokens, method, chosen).items():
            summaries.append(row(tokens, method, scope, m, len(chosen)))
    for entry in plan["budgets"]:
        budget = entry["budget_tokens"]
        random_metrics = defaultdict(list)
        chosen_counts = []
        for trial, chosen in enumerate(entry["random_subsets"], 1):
            assert sum(costs[i] for i in chosen) == budget
            chosen_counts.append(len(chosen))
            met = metrics(rank_subset(rankers, mask_of(n, chosen)), masks)
            for scope, m in met.items():
                random_metrics[scope].append(m)
                trial_rows.append(dict(stage=stage, budget_tokens=budget, trial=trial,
                                       selected_documents=len(chosen), scope=scope, **m))
        for scope, values in random_metrics.items():
            r1 = np.array([m["recall_at_1"] for m in values])
            mrr = np.array([m["mrr"] for m in values])
            averages = {k: float(np.mean([m[k] for m in values])) for k in values[0]}
            summaries.append(row(budget, "A_RANDOM_MEAN", scope, averages, float(np.mean(chosen_counts)),
                                 r1p05=float(np.quantile(r1, .05)), r1p95=float(np.quantile(r1, .95)),
                                 mrrp05=float(np.quantile(mrr, .05)), mrrp95=float(np.quantile(mrr, .95))))
        for method, chosen in (("B_SEVERITY", entry["severity_subset"]),
                               ("C_GAIN_KNAPSACK", entry["gain_subset"])):
            for i in chosen:
                allocation_rows.append(dict(stage=stage, budget_tokens=budget, strategy=method, **pool[i]))
            for scope, m in record(budget, method, chosen).items():
                random_mrr = np.array([x["mrr"] for x in random_metrics[scope]])
                summaries.append(row(budget, method, scope, m, len(chosen),
                                     pct=float(np.mean(random_mrr <= m["mrr"]+1e-14))))
        print(f"{stage}: exact {budget:,} tokens, {len(entry['random_subsets'])} random allocations", flush=True)
    if output is not None:
        write(output / "budget_summary.csv", summaries)
        write(output / "random_trials.csv", trial_rows)
        write(output / "allocation_documents.csv", allocation_rows)
        write(output / "deterministic_rank_results.csv", rank_rows)
    return summaries


def make_holdout_template(inputs):
    pool_places = {r["place_id"] for r in inputs["pool"]}
    sources = {r["place_id"]: r for r in inputs["gold"] if r["language"] == "ko"}
    rng = random.Random(SEED+1)
    selected = rng.sample(sorted(pool_places), 20)+rng.sample(sorted(set(sources)-pool_places), 10)
    rng.shuffle(selected)
    template = [dict(holdout_intent_id=f"H{i:02d}", place_id=p, title_ko=sources[p]["title"],
                     reference_description_ko=sources[p]["overview"], query_ko="", query_en="",
                     query_ja="", author_note="") for i, p in enumerate(selected, 1)]
    path = OUT / "holdout_query_template.csv"
    if not path.exists():
        write(path, template)
    return [(r["holdout_intent_id"], r["place_id"]) for r in template]


def development(inputs):
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "locked_allocations.json"
    if plan_path.exists():
        raise FileExistsError("Allocation plan already locked. Use --holdout-file for independent evaluation.")
    queries = inputs["old_queries"]
    qvec = np.load(QUERIES / "query_embeddings.npy", allow_pickle=False)
    rankers, scores, new_scores = build_rankers(inputs, queries, qvec)
    verify_rankers(inputs, rankers, scores, new_scores, queries)
    del scores, new_scores
    pool = inputs["pool"]
    costs = [r["output_tokens"] for r in pool]
    masks = scope_masks(queries, pool, development=True)
    base_met = metrics(rank_subset(rankers, mask_of(len(pool), [])), masks)
    for i, r in enumerate(pool):
        m = metrics(rank_subset(rankers, mask_of(len(pool), [i])), masks)
        gain = m["GAP_TARGETS"]["mrr"]-base_met["GAP_TARGETS"]["mrr"]
        r.update(development_mrr_gain=gain, development_mrr_gain_per_token=gain/r["output_tokens"],
                 development_r1_gain_pp=100*(m["GAP_TARGETS"]["recall_at_1"]-base_met["GAP_TARGETS"]["recall_at_1"]))
    write(OUT / "document_priority_scores.csv", pool)
    utility = [r["development_mrr_gain"] for r in pool]
    severity = [r["severity"] for r in pool]
    counts = counts_by_suffix(costs)
    rng = random.Random(SEED)
    budgets = []
    total = sum(costs)
    for frac in FRACTIONS:
        desired = int(round(frac*total))
        budget = max(b for b in range(desired+1) if counts[0][b] > 0)
        budgets.append(dict(target_percent=100*frac, budget_tokens=budget,
                            exact_feasible_subset_count=str(counts[0][budget]),
                            severity_subset=optimum_exact_subset(costs, severity, budget),
                            gain_subset=optimum_exact_subset(costs, utility, budget),
                            random_subsets=[uniform_exact_subset(costs, counts, budget, rng)
                                            for _ in range(RANDOM_TRIALS)]))
    plan = dict(source_signature=inputs["signature"], seed=SEED, total_tokens=total,
                primary_development_metric="GAP_TARGETS_MRR", candidate_pool=pool, budgets=budgets,
                holdout_intent_place_pairs=make_holdout_template(inputs),
                strategy_A="Uniform random sample among exact-cost feasible subsets, 500 trials per budget.",
                strategy_B="Exact-cost 0–1 knapsack maximizing sum of 2*MISSING + PARTIAL.",
                strategy_C="Exact-cost 0–1 knapsack maximizing sum of individual document marginal "
                           "development GAP_TARGETS MRR gains against unchanged baseline.",
                limitation="Additive marginal gains are a surrogate; joint retrieval effects need not add.",
                evaluation_status="DEVELOPMENT_ONLY; independent query file not yet supplied")
    write_json(plan_path, plan)
    summary = evaluate_allocations(plan, rankers, queries, masks, OUT, pool, "DEVELOPMENT")
    for r in summary:
        if r["scope"] == "GAP_TARGETS":
            print(r["budget_tokens"], r["strategy"], f"R@1={r['recall_at_1']:.2%}",
                  f"MRR={r['mrr']:.4f}", flush=True)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def verify_plan(inputs):
    """Replay the frozen development allocations without modifying results."""
    plan = json.loads((OUT / "locked_allocations.json").read_text())
    assert plan["source_signature"] == inputs["signature"]
    pool = plan["candidate_pool"]
    assert [(r["place_id"], r["language"], r["output_tokens"]) for r in pool] == [
        (r["place_id"], r["language"], r["output_tokens"]) for r in inputs["pool"]
    ]
    costs = [r["output_tokens"] for r in pool]
    for entry in plan["budgets"]:
        selections = [entry["severity_subset"], entry["gain_subset"], *entry["random_subsets"]]
        assert len(selections) == RANDOM_TRIALS+2
        assert all(len(chosen) == len(set(chosen)) and
                   sum(costs[i] for i in chosen) == entry["budget_tokens"] for chosen in selections)
    queries = inputs["old_queries"]
    qvec = np.load(QUERIES / "query_embeddings.npy", allow_pickle=False)
    rankers, scores, new_scores = build_rankers(inputs, queries, qvec)
    verify_rankers(inputs, rankers, scores, new_scores, queries)
    del scores, new_scores
    masks = scope_masks(queries, pool, development=True)
    actual = evaluate_allocations(plan, rankers, queries, masks, None, pool, "DEVELOPMENT")
    expected = read(OUT / "budget_summary.csv")
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        for k, value in a.items():
            if isinstance(value, (float, int)):
                assert np.isclose(value, float(b[k]), rtol=0, atol=1e-12), (k, value, b[k])
            else:
                assert str(value) == b[k]
    print("PASS: every frozen allocation has the exact budget and development metrics reproduce; no files changed.",
          flush=True)


def holdout(inputs, path):
    plan = json.loads((OUT / "locked_allocations.json").read_text())
    assert plan["source_signature"] == inputs["signature"], "Frozen experimental inputs changed"
    pool = plan["candidate_pool"]
    assert [(r["place_id"], r["language"]) for r in pool] == [
        (r["place_id"], r["language"]) for r in inputs["pool"]
    ]
    expected = {tuple(x) for x in plan["holdout_intent_place_pairs"]}
    filled = read(path)
    assert len(filled) == 30 and {(r["holdout_intent_id"], r["place_id"]) for r in filled} == expected
    old_text = {normalized(q["query"]) for q in inputs["old_queries"]}
    queries = []
    for r in filled:
        for lang in LANGS:
            text = r[f"query_{lang}"].strip()
            if not text or normalized(text) in old_text:
                raise ValueError("Independent queries must be nonempty and not exact copies of development queries.")
            queries.append(dict(query_id=r["holdout_intent_id"]+"-"+lang,
                                query_language=lang, query_type="HELD_OUT", place_id=r["place_id"], query=text))
    assert len({normalized(q["query"]) for q in queries}) == len(queries) == 90
    output = OUT / "heldout_validation"
    if output.exists():
        raise FileExistsError("A held-out result directory already exists; do not tune or overwrite the first test.")
    from prepare_large_corpus import compatibility, encode
    from sentence_transformers import SentenceTransformer
    meta = json.loads((BASE / "run_metadata.json").read_text())
    model = SentenceTransformer(meta["model"], device=meta["device"], local_files_only=True,
                                model_kwargs={"torch_dtype":meta["precision"]})
    model.max_seq_length = meta["max_seq_length"]
    match = compatibility(model, inputs["gold"])
    qvec = encode(model, [r["query"] for r in queries], 4)
    del model
    rankers, scores, new_scores = build_rankers(inputs, queries, qvec)
    del scores, new_scores
    masks = scope_masks(queries, pool, development=False)
    output.mkdir()
    write(output / "queries.csv", queries)
    np.save(output / "query_embeddings.npy", qvec)
    write_json(output / "holdout_metadata.json", dict(
        source_query_sha256=sha256(path), allocation_plan_sha256=sha256(OUT / "locked_allocations.json"),
        compatibility=match, queries=90, places=30, primary_scope="TARGETED_DOCUMENTS",
        limitation="New query texts for known places; not an unseen-place or independently validated "
                   "human study unless query origin is documented separately."
    ))
    evaluate_allocations(plan, rankers, queries, masks, output, pool, "HELD_OUT")


def main():
    p = argparse.ArgumentParser()
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--holdout-file", type=Path)
    modes.add_argument("--verify-plan", action="store_true")
    args = p.parse_args()
    inputs = load_inputs()
    if args.holdout_file:
        holdout(inputs, args.holdout_file)
    elif args.verify_plan:
        verify_plan(inputs)
    else:
        development(inputs)


if __name__ == "__main__":
    main()
