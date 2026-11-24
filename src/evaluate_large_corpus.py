"""Evaluate the fixed 100-place reference set against 300 vs 18,038 records.

Three input conditions are kept distinct:
  METADATA: title + address for every record (fair corpus-size comparison).
  AVAILABLE_ORIGINAL: real cached descriptions wherever available.
  AVAILABLE_ENRICHED: the same corpus with only the original 68 gold
                      documents selectively supplemented.

The latter two conditions have very unequal gold/non-gold description
coverage, which must be reported when interpreting their absolute scores.
"""

import json
import statistics
from collections import defaultdict

import numpy as np

from prepare_large_corpus import (
    OUT, PROCESSED, RESULTS, BASE, QUERIES, LANGS, read, write, write_json, sha256, key,
)
from statistical_validate_enrichment import paired_test, holm

FEATURE = RESULTS / "tourapi_full_feature_enrichment"
COVERAGE = RESULTS / "tourapi_coverage_retrieval_analysis"
CONDITIONS = ("METADATA", "AVAILABLE_ORIGINAL", "AVAILABLE_ENRICHED")
CORPORA = ("PILOT_300", "FULL_18038")
TARGETS = (*LANGS, "ALL_PLACES")
SCOPES = (
    "MIXED", "SAME_LANGUAGE", "CROSS_LANGUAGE", "ALL_SINGLE_LANGUAGE",
    "GAP_TARGETS", "GAP_TARGETS_CROSS", "PRESENT_TARGETS", "GAP_INTENT_MIXED",
)


def rank_values(scores, target_positions):
    """Rank with stable candidate-ID tie breaking, independent of row order."""
    target_positions = np.asarray(target_positions, dtype=int)
    qindices = np.arange(scores.shape[0])
    correct = scores[qindices, target_positions]
    candidate_positions = np.arange(scores.shape[1])
    ranks = 1+np.count_nonzero(scores > correct[:, None], axis=1)
    ranks += np.count_nonzero((scores == correct[:, None])
                              & (candidate_positions[None, :] < target_positions[:, None]), axis=1)
    top = np.argmax(scores, axis=1)
    return ranks, correct, top, scores[qindices, top]


def rank_condition(scores, index, queries, subset, condition, corpus, gaps, coverage):
    """Return all 2,400 evaluation rows in the same four-scope format."""
    subset = sorted(subset, key=lambda i: (
        index[i]["candidate_group_id"], index[i]["language"], index[i]["source_contentid"]
    ))
    gold = {r["gold_place_id"]+"/"+r["language"]: i for i, r in enumerate(index)
            if r["gold_place_id"]}

    def row(q, target, candidate_count, rank, correct, top_idx, top_score):
        r = index[top_idx]
        k = q["query_id"], target
        return dict(
            condition=condition, corpus=corpus, query_id=q["query_id"],
            query_type=q["query_type"], query_language=q["query_language"],
            target_language=target, place_id=q["place_id"], query=q["query"],
            candidate_count=candidate_count, gap_target=gaps[k],
            feature_coverage=coverage.get(k, ""),
            rank=int(rank), correct_score=round(float(correct), 6),
            top1_candidate_group=r["candidate_group_id"],
            top1_source_contentid=r["source_contentid"], top1_title=r["title"],
            top1_language=r["language"], top1_score=round(float(top_score), 6),
        )

    rows = []
    for lang in LANGS:
        doc_indices = [i for i in subset if index[i]["language"] == lang]
        pos = {doc_i: i for i, doc_i in enumerate(doc_indices)}
        targets = [pos[gold[q["place_id"]+"/"+lang]] for q in queries]
        values = scores[:, doc_indices]
        ranks, correct, top, top_scores = rank_values(values, targets)
        rows.extend(row(q, lang, len(doc_indices), ranks[i], correct[i], doc_indices[top[i]],
                        top_scores[i]) for i, q in enumerate(queries))
        del values

    # Only previously linked triples are merged. Otherwise retain their
    # source record identities; do not assume unmatched records are different places.
    group_ids = [index[i]["candidate_group_id"] for i in subset]
    starts = np.array([i for i, group in enumerate(group_ids)
                       if i == 0 or group != group_ids[i-1]], dtype=int)
    ends = np.r_[starts[1:], len(subset)]
    groups = [group_ids[i] for i in starts]
    positions = {group: i for i, group in enumerate(groups)}
    targets = [positions["place:"+q["place_id"]] for q in queries]
    values = np.maximum.reduceat(scores[:, subset], starts, axis=1)
    ranks, correct, top, top_scores = rank_values(values, targets)
    for i, q in enumerate(queries):
        group_docs = subset[starts[top[i]]:ends[top[i]]]
        best_doc = group_docs[int(np.argmax(scores[i, group_docs]))]
        rows.append(row(q, "ALL_PLACES", len(groups), ranks[i], correct[i], best_doc, top_scores[i]))
    assert len(rows) == 2400 and len({(r["query_id"], r["target_language"]) for r in rows}) == 2400
    return rows


def belongs(r, scope):
    target, source = r["target_language"], r["query_language"]
    if scope == "MIXED":
        return target == "ALL_PLACES"
    if scope == "SAME_LANGUAGE":
        return target == source
    if scope == "CROSS_LANGUAGE":
        return target in LANGS and target != source
    if scope == "ALL_SINGLE_LANGUAGE":
        return target in LANGS
    if scope == "GAP_TARGETS":
        return r["gap_target"] == "YES"
    if scope == "GAP_TARGETS_CROSS":
        return r["gap_target"] == "YES" and target in LANGS and target != source
    if scope == "PRESENT_TARGETS":
        return r["feature_coverage"] == "PRESENT"
    if scope == "GAP_INTENT_MIXED":
        return r["gap_target"] == "YES_MIXED"
    raise ValueError(scope)


def metric(rows, rank_key="rank"):
    ranks = [int(r[rank_key]) for r in rows]
    n = len(ranks)
    result = dict(evaluations=n, distinct_places=len({r["place_id"] for r in rows}))
    for k in (1, 5, 10, 20, 50):
        h = sum(r <= k for r in ranks)
        result[f"hits_at_{k}"] = h
        result[f"recall_at_{k}"] = round(h/n, 6)
    result["mrr"] = round(statistics.mean(1/r for r in ranks), 6)
    result["mean_rank"] = round(statistics.mean(ranks), 3)
    result["median_rank"] = statistics.median(ranks)
    return result


def keyed(records):
    result = {(r["query_id"], r["target_language"]): r for r in records}
    assert len(result) == len(records) == 2400
    return result


def pair_rows(before, after):
    a, b = keyed(before), keyed(after)
    assert a.keys() == b.keys()
    output = []
    for k, r in a.items():
        s = b[k]
        assert r["place_id"] == s["place_id"] and r["gap_target"] == s["gap_target"]
        output.append(dict(
            query_id=r["query_id"], query_type=r["query_type"],
            query_language=r["query_language"], target_language=r["target_language"],
            place_id=r["place_id"], gap_target=r["gap_target"], feature_coverage=r["feature_coverage"],
            rank_before=r["rank"], rank_after=s["rank"],
        ))
    return output


def main():
    progress = json.loads((OUT / "embedding_progress.json").read_text())
    corpus_info = json.loads((OUT / "corpus_metadata.json").read_text())
    assert progress["completed_metadata_documents"] == corpus_info["source_documents"] == 18038
    assert progress["extra_details_complete"] and progress["data_signature"] == corpus_info["data_signature"]
    signature = progress["data_signature"]
    assert signature["natural_queries_sha256"] == sha256(QUERIES / "queries.csv")
    assert signature["query_embeddings_sha256"] == sha256(QUERIES / "query_embeddings.npy")
    assert signature["original_gold_embeddings_sha256"] == sha256(BASE / "document_embeddings.npy")
    assert signature["enriched_gold_embeddings_sha256"] == sha256(FEATURE / "document_embeddings_enriched.npy")

    index = read(OUT / "corpus_index.csv")
    gold = read(PROCESSED / "tourapi_pilot_details_300.csv")
    queries = read(QUERIES / "queries.csv")
    source_positions = {key(r): i for i, r in enumerate(index)}
    gold_positions = [source_positions[key(r)] for r in gold]
    assert len(index) == len(source_positions) == 18038
    assert len(gold) == len(set(gold_positions)) == 300
    assert len(queries) == 600 and len({r["place_id"] for r in queries}) == 100
    assert all(int(r["embedding_row"]) == i for i, r in enumerate(index))
    assert all(index[i]["gold_place_id"] == d["place_id"] for i, d in zip(gold_positions, gold))

    # Confirm that the uniform-metadata corpus did not silently hit the
    # same sequence limit used in the original experiments.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(progress["settings"]["model"], local_files_only=True)
    token_lengths = []
    for start in range(0, len(index), 512):
        batch = [r["metadata_text"] for r in index[start:start+512]]
        result = tokenizer(batch, add_special_tokens=True, truncation=False, verbose=False)
        token_lengths.extend(len(ids) for ids in result["input_ids"])
    max_tokens = max(token_lengths)
    assert len(token_lengths) == 18038 and max_tokens <= progress["settings"]["max_seq_length"]
    print(f"Verified no metadata truncation; maximum length {max_tokens} tokens.", flush=True)

    prior = read(FEATURE / "rank_comparison.csv")
    gaps = {(r["query_id"], r["target_language"]): r["gap_target"] for r in prior}
    eligible = [r for r in read(COVERAGE / "coverage_rank_joined.csv")
                if r["eligible_primary"] == "YES"]
    coverage = {(r["query_id"], r["target_language"]): r["coverage_group"] for r in eligible}
    assert len(gaps) == 2400 and len(coverage) == 1185
    assert sum(v == "YES" for v in gaps.values()) == 219

    query_vec = np.load(QUERIES / "query_embeddings.npy", allow_pickle=False)
    metadata_vec = np.load(OUT / "metadata_embeddings.npy", mmap_mode="r", allow_pickle=False)
    base_vec = np.load(BASE / "document_embeddings.npy", allow_pickle=False)
    enriched_vec = np.load(FEATURE / "document_embeddings_enriched.npy", allow_pickle=False)
    extra_vec = np.load(OUT / "extra_detail_embeddings.npy", allow_pickle=False)
    extra_indices = read(OUT / "extra_detail_index.csv")
    extra_positions = [int(r["embedding_row"]) for r in extra_indices]
    assert query_vec.shape == (600, 1024) and metadata_vec.shape == (18038, 1024)
    assert base_vec.shape == enriched_vec.shape == (300, 1024) and extra_vec.shape == (63, 1024)
    assert all(np.isfinite(x).all() for x in (query_vec, metadata_vec, base_vec, enriched_vec, extra_vec))
    assert set(extra_positions).isdisjoint(gold_positions)
    assert all(index[i]["description_source"] == "EXISTING_CACHE" for i in extra_positions)
    changed = [i for i in range(300) if not np.array_equal(base_vec[i], enriched_vec[i])]
    assert len(changed) == 68

    # Exactly the same candidate scores are used when changing only corpus size.
    all_positions = list(range(18038))
    by_condition = {}
    print("Computing 600 × 18,038 metadata cosine scores...", flush=True)
    scores = query_vec @ metadata_vec.T
    for condition in CONDITIONS:
        if condition == "AVAILABLE_ORIGINAL":
            # Reuse the original saved embeddings for all 300 gold records
            # and add the 63 genuinely cached non-gold descriptions.
            scores[:, gold_positions] = query_vec @ base_vec.T
            scores[:, extra_positions] = query_vec @ extra_vec.T
        elif condition == "AVAILABLE_ENRICHED":
            # Only the 68 existing intervention vectors differ; all non-gold
            # scores stay completely unchanged. Score multiplication uses the
            # same original 300-document matrix size to preserve the baseline.
            scores[:, gold_positions] = query_vec @ enriched_vec.T
        for corpus, subset in (("PILOT_300", gold_positions), ("FULL_18038", all_positions)):
            print(f"Ranking: {condition} / {corpus}", flush=True)
            by_condition[condition, corpus] = rank_condition(
                scores, index, queries, subset, condition, corpus, gaps, coverage
            )
        smaller = keyed(by_condition[condition, "PILOT_300"])
        larger = keyed(by_condition[condition, "FULL_18038"])
        assert all(larger[k]["rank"] >= row["rank"] for k, row in smaller.items()), +            "With identical scores and target groups, adding candidates cannot improve the rank."
    del scores

    # The pilot available-description arms must reproduce the earlier research.
    for condition, column in (("AVAILABLE_ORIGINAL", "rank_before"), ("AVAILABLE_ENRICHED", "rank_after")):
        actual = keyed(by_condition[condition, "PILOT_300"])
        mismatch = sum(int(r[column]) != actual[r["query_id"], r["target_language"]]["rank"] for r in prior)
        assert mismatch == 0, f"{condition}: {mismatch} prior ranks did not reproduce"

    all_rows = [r for condition in CONDITIONS for corpus in CORPORA for r in by_condition[condition, corpus]]
    assert len(all_rows) == 14400
    write(OUT / "rank_results.csv", all_rows)
    metrics = []
    language_metrics = []
    for (condition, corpus), records in by_condition.items():
        for scope in SCOPES:
            group = [r for r in records if belongs(r, scope)]
            metrics.append(dict(condition=condition, corpus=corpus, scope=scope, **metric(group)))
        for q_lang in LANGS:
            for target in TARGETS:
                group = [r for r in records if r["query_language"] == q_lang and r["target_language"] == target]
                language_metrics.append(dict(condition=condition, corpus=corpus, query_language=q_lang,
                                             target_language=target, candidate_count=group[0]["candidate_count"],
                                             **metric(group)))
    write(OUT / "metrics.csv", metrics)
    write(OUT / "language_metrics.csv", language_metrics)

    comparisons = (
        ("EXPANSION_METADATA", ("METADATA", "PILOT_300"), ("METADATA", "FULL_18038")),
        ("EXPANSION_AVAILABLE_ORIGINAL", ("AVAILABLE_ORIGINAL", "PILOT_300"), ("AVAILABLE_ORIGINAL", "FULL_18038")),
        ("ENRICHMENT_PILOT", ("AVAILABLE_ORIGINAL", "PILOT_300"), ("AVAILABLE_ENRICHED", "PILOT_300")),
        ("ENRICHMENT_FULL", ("AVAILABLE_ORIGINAL", "FULL_18038"), ("AVAILABLE_ENRICHED", "FULL_18038")),
    )
    paired_summaries = []
    full_enrichment_pairs = None
    for label, before, after in comparisons:
        pairs = pair_rows(by_condition[before], by_condition[after])
        if label == "ENRICHMENT_FULL":
            full_enrichment_pairs = pairs
            write(OUT / "full_corpus_enrichment_rank_comparison.csv", [
                dict(**r, rank_gain=r["rank_before"]-r["rank_after"]) for r in pairs
            ])
        for scope in SCOPES:
            group = [r for r in pairs if belongs(r, scope)]
            first, second = metric(group, "rank_before"), metric(group, "rank_after")
            paired_summaries.append(dict(
                comparison=label, scope=scope, evaluations=len(group), distinct_places=first["distinct_places"],
                improved=sum(r["rank_after"] < r["rank_before"] for r in group),
                worse=sum(r["rank_after"] > r["rank_before"] for r in group),
                unchanged=sum(r["rank_after"] == r["rank_before"] for r in group),
                r1_before=first["recall_at_1"], r1_after=second["recall_at_1"],
                r1_change_pp=round(100*(second["recall_at_1"]-first["recall_at_1"]), 4),
                r5_before=first["recall_at_5"], r5_after=second["recall_at_5"],
                r10_before=first["recall_at_10"], r10_after=second["recall_at_10"],
                mrr_before=first["mrr"], mrr_after=second["mrr"],
                mean_rank_before=first["mean_rank"], mean_rank_after=second["mean_rank"],
                median_rank_before=first["median_rank"], median_rank_after=second["median_rank"],
            ))
    write(OUT / "paired_summary.csv", paired_summaries)

    # Quantify the information-coverage gap with the same fixed labels, not
    # an assumed coverage measure for the incompletely described distractors.
    coverage_gap = []
    for corpus in CORPORA:
        for condition in ("AVAILABLE_ORIGINAL", "AVAILABLE_ENRICHED"):
            records = by_condition[condition, corpus]
            for relation in ("ALL", "SAME", "CROSS"):
                def matching(r):
                    return (relation == "ALL" or
                            (r["target_language"] == r["query_language"]) == (relation == "SAME"))
                present = [r for r in records if r["feature_coverage"] == "PRESENT" and matching(r)]
                missing = [r for r in records if r["gap_target"] == "YES" and matching(r)]
                pm, gm = metric(present), metric(missing)
                coverage_gap.append(dict(
                    corpus=corpus, condition=condition, language_relation=relation,
                    present_evaluations=len(present), gap_evaluations=len(missing),
                    present_r1=pm["recall_at_1"], gap_r1=gm["recall_at_1"],
                    r1_gap_pp=round(100*(pm["recall_at_1"]-gm["recall_at_1"]), 4),
                    present_mrr=pm["mrr"], gap_mrr=gm["mrr"],
                ))
    write(OUT / "coverage_gap_comparison.csv", coverage_gap)

    print("Paired exploratory significance tests for full-corpus selective enrichment...", flush=True)
    tests = []
    rng = np.random.default_rng(20261120)
    for scope, measure in (("GAP_TARGETS", "R@1"), ("GAP_TARGETS", "R@5"), ("GAP_TARGETS", "MRR"),
                           ("GAP_TARGETS_CROSS", "R@1"), ("MIXED", "R@1"), ("CROSS_LANGUAGE", "R@1")):
        group = [r for r in full_enrichment_pairs if belongs(r, scope)]
        rec = paired_test("FULL_"+scope+"_"+measure, group, measure, rng,
                          primary=(scope == "GAP_TARGETS" and measure == "R@1"))
        tests.append(rec)
    for row, adjusted in zip(tests, holm([r["raw_p_value"] for r in tests])):
        row["holm_adjusted_p_value"] = adjusted
        row["holm_p_lt_0_05"] = "YES" if adjusted < 0.05 else "NO"
    write(OUT / "statistical_tests.csv", tests)

    write_json(OUT / "experiment_metadata.json", dict(
        source_records=18038, gold_places=100, gold_records=300, queries=600,
        pilot_candidates_per_language=100, full_candidates_by_language={
            lang: sum(r["language"] == lang for r in index) for lang in LANGS
        }, full_mixed_candidate_groups=17378,
        distinct_evaluation_keys_per_condition=2400, condition_corpus_combinations=6,
        total_recorded_evaluations=14400, changed_gold_documents=68,
        metadata_condition="Uniform title and addresses only for all documents.",
        metadata_max_tokens=max_tokens, metadata_truncated_documents=0,
        max_extra_cached_description_tokens=progress["max_extra_detail_tokens"],
        available_condition="363 real existing descriptions (300 gold and 63 non-gold); remaining metadata only.",
        selective_enrichment_condition="Same available-information corpus, replacing exactly 68 previously enriched gold documents.",
        query_embeddings_reused=True, original_gold_embeddings_reused=True,
        fresh_model_compatibility=progress["compatibility"],
        source_signature=signature,
        score="Dot product of L2-normalized 1024D BGE-M3 vectors, stable candidate-ID tie break.",
        primary_corpus_size_comparison="METADATA, same representation and queries in PILOT_300 and FULL_18038.",
        limitations=[
            "No additional API data were collected. The available-description condition has much richer gold documents than distractors; absolute accuracy can be optimistic.",
            "The full 18,038 records are not 18,038 unique attractions. Only 330 known triples are linked, leaving 17,378 conservative candidate groups in mixed retrieval.",
            "Language-restricted candidate counts are unequal (12,581 Korean, 2,822 English, 2,635 Japanese); raw cross-language rank differences may partly reflect pool size.",
            "Gold-set target IDs are designated by the fixed AI-assisted 100-place evaluation set, not independent human relevance judgments.",
            "The same AI-generated queries informed gap labeling and enrichment; this remains in-sample exploratory evaluation.",
            "Query/document comparisons and repeated language evaluations are not independent observations or real-world search traffic.",
            "Paired tests cluster by place; this does not account for every dependency induced by shared ranking candidates.",
        ],
    ))
    for row in paired_summaries:
        if row["scope"] in ("MIXED", "GAP_TARGETS"):
            print(row["comparison"], row["scope"], f"R@1 {row['r1_before']:.2%} -> {row['r1_after']:.2%}",
                  flush=True)
    print("Full-corpus evaluation complete.", flush=True)


if __name__ == "__main__":
    main()
