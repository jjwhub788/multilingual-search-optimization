"""Choose fixed-size AI Hub enrichment subsets without reading final test data.

Input dev_results CSV columns: query_id, language (ko/en/ja), field
(experience/opening_hours/holidays/admission_fee/parking), split (dev),
rank_before (rank of first relevant place in the unchanged baseline).

Strategies use the same exact number of items per language. Token cost is
reported separately; the canonical per-item token sum is a budget proxy, not
the final concatenated input length. For structure comparison, reuse one
strategy's fact IDs, never select separately by representation.
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"
LANGS = ("ko", "en", "ja")
FIELDS = ("experience", "opening_hours", "holidays", "admission_fee", "parking")
STRATEGIES = ("random", "deficit", "intent")
GAP_WEIGHT = {"MISSING": 1.0, "PARTIAL": 0.5, "PRESENT": 0.0}


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def deficit_scores(gap_rows):
    result = defaultdict(float)
    keys = set()
    for r in gap_rows:
        key = r["place_id"], r["language"], r["field"]
        if key in keys:
            raise ValueError(f"Duplicate gap record: {key}")
        keys.add(key)
        result[key[:2]] += GAP_WEIGHT[r["tourapi_final_status"]]
    assert len(keys) == 2070 and len(result) == 414
    return result


def intent_weights(dev):
    """Development-only expected reciprocal-rank shortfall per intent.

    For a language/field, sum 1-1/rank across its development queries and
    divide by all development queries in that language. Thus a frequent
    intent with poor baseline ranks receives greater priority, without using
    a final evaluation query or a gold answer to an individual test query.
    """
    if not dev:
        raise ValueError("Development baseline results are required for intent priority.")
    required = {"query_id", "language", "field", "split", "rank_before"}
    if not required <= set(dev[0]):
        raise ValueError("dev_results needs columns: " + ", ".join(sorted(required)))
    seen = set()
    counts = Counter()
    shortfall = defaultdict(float)
    for r in dev:
        if r["split"].strip().lower() != "dev":
            raise ValueError("Only development rows are allowed; remove all final test rows.")
        lang, field = r["language"], r["field"]
        if lang not in LANGS or field not in FIELDS:
            raise ValueError(f"Unknown language or intent field: {lang}/{field}")
        key = r["query_id"], lang
        if key in seen or not r["query_id"]:
            raise ValueError(f"Duplicate or empty development query ID: {key}")
        seen.add(key)
        rank = int(r["rank_before"])
        if not 1 <= rank <= 138:
            raise ValueError(f"Baseline rank out of range: {rank}")
        counts[lang] += 1
        shortfall[lang, field] += 1.0 - 1.0 / rank
    if set(counts) != set(LANGS):
        raise ValueError("Development results must contain all three languages.")
    return {(l, f): shortfall[l, f] / counts[l] for l in LANGS for f in FIELDS}


def select(items, gaps, dev, budget, seed):
    if budget <= 0:
        raise ValueError("Per-language budget must be positive.")
    if len({r["fact_id"] for r in items}) != len(items):
        raise ValueError("Duplicate enrichment fact IDs")
    scores = deficit_scores(gaps)
    weights = intent_weights(dev)
    out = []
    for lang_index, lang in enumerate(LANGS):
        candidates = sorted((r for r in items if r["language"] == lang), key=lambda r: r["fact_id"])
        if budget > len(candidates):
            raise ValueError(f"Budget {budget} exceeds available {lang} candidates ({len(candidates)})")
        rng = random.Random(seed + 1009 * lang_index)
        rng.shuffle(candidates)
        tie_order = {r["fact_id"]: i for i, r in enumerate(candidates)}

        def severity(r):
            return GAP_WEIGHT[r["tourapi_status"]]

        def deficit(r):
            return scores[r["place_id"], lang]

        def intent(r):
            return weights[lang, r["field"]] * severity(r)

        ordered = {
            "random": candidates,
            "deficit": sorted(candidates, key=lambda r: (-deficit(r), -severity(r), tie_order[r["fact_id"]])),
            "intent": sorted(candidates, key=lambda r: (-intent(r), tie_order[r["fact_id"]])),
        }
        for strategy, rows in ordered.items():
            for position, r in enumerate(rows[:budget], 1):
                out.append({
                    "strategy": strategy, "budget_per_language": str(budget), "seed": str(seed),
                    "fact_id": r["fact_id"], "place_id": r["place_id"], "language": lang, "field": r["field"],
                    "selection_position_within_language": str(position), "tourapi_status": r["tourapi_status"],
                    "deficit_score": str(round(deficit(r), 6)), "intent_weight_dev": str(round(weights[lang, r["field"]], 8)),
                    "intent_priority_score": str(round(intent(r), 8)), "canonical_budget_tokens": r["canonical_budget_tokens"],
                })
    assert len(out) == len(STRATEGIES) * len(LANGS) * budget
    assert len({(r["strategy"], r["fact_id"]) for r in out}) == len(out)
    counts = Counter((r["strategy"], r["language"]) for r in out)
    assert all(counts[s, l] == budget for s in STRATEGIES for l in LANGS)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-results", type=Path, required=True)
    parser.add_argument("--budget-per-language", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = read(ROOT / "common_enrichment_item_pool.csv")
    gaps = read(ROOT / "common_gap_review_final.csv")
    dev = read(args.dev_results)
    assert len(items) == 440
    result = select(items, gaps, dev, args.budget_per_language, args.seed)
    with args.output.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result[0]))
        w.writeheader()
        w.writerows(result)
    assert read(args.output) == result
    for strategy in STRATEGIES:
        rows = [r for r in result if r["strategy"] == strategy]
        print(strategy, "items", len(rows), "canonical token cost", sum(int(r["canonical_budget_tokens"]) for r in rows), "by field", Counter(r["field"] for r in rows))


if __name__ == "__main__":
    main()
