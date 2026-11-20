"""Read-only verification of the saved research dataset and headline results.

No API requests, model downloads, or result-file modifications are performed.
Run from the project root: python src/verify_research.py
"""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from run_bge_m3 import evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
LANGUAGES = {"ko", "en", "ja"}
TARGETS = LANGUAGES | {"ALL_PLACES"}


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def embedding(path, expected_shape):
    array = np.load(path, allow_pickle=False)
    check(array.shape == expected_shape, f"Unexpected embedding shape: {path}")
    check(np.isfinite(array).all(), f"Non-finite embedding values: {path}")
    return array


def rank_index(records, column="rank"):
    result = {(r["query_id"], r["target_language"]): int(r[column]) for r in records}
    check(len(result) == len(records), "Duplicate query/target-language evaluation")
    check(all(1 <= rank <= 100 for rank in result.values()), "Rank outside 1–100")
    return result


def metric_matches(records, rank_column, saved, suffix=""):
    ranks = [int(r[rank_column]) for r in records]
    check(bool(ranks), "Empty evaluation group")
    for k in (1, 5, 10):
        computed = sum(rank <= k for rank in ranks) / len(ranks)
        check(
            math.isclose(computed, float(saved[f"recall_at_{k}{suffix}"]), abs_tol=0.00000051),
            f"R@{k} does not match stored result",
        )
    mrr = sum(1 / rank for rank in ranks) / len(ranks)
    check(math.isclose(mrr, float(saved[f"mrr{suffix}"]), abs_tol=0.00000051),
          "MRR does not match stored result")


def scope_rows(records, scope):
    if scope == "MIXED":
        return [r for r in records if r["target_language"] == "ALL_PLACES"]
    if scope == "CROSS_LANGUAGE":
        return [r for r in records if r["target_language"] in LANGUAGES
                and r["target_language"] != r["query_language"]]
    if scope == "GAP_TARGETS":
        return [r for r in records if r["gap_target"] == "YES"]
    if scope == "GAP_TARGETS_CROSS":
        return [r for r in scope_rows(records, "GAP_TARGETS")
                if r["target_language"] in LANGUAGES
                and r["target_language"] != r["query_language"]]
    raise ValueError(f"Unknown scope: {scope}")


def main():
    pilot_path = DATA / "tourapi_pilot_details_300.csv"
    standard_path = DATA / "tourapi_standardized.csv"
    input_query_path = DATA / "tourapi_ai_generated_queries_600_natural.csv"
    docs, standard, intents = map(rows, (pilot_path, standard_path, input_query_path))
    check(len(docs) == len(standard) == 300, "Expected 300 documents")
    places = {r["place_id"] for r in docs}
    check(len(places) == 100, "Expected 100 aligned places")
    check({(r["place_id"], r["language"]) for r in docs}
          == {(p, lang) for p in places for lang in LANGUAGES}, "Language alignment failure")
    check(all(r["collection_status"] == "COMPLETE" and r["overview"].strip() for r in docs),
          "Incomplete source description")
    check(all((d["place_id"], d["language"], d["title"], d["overview"])
              == (s["place_id"], s["language"], s["title"], s["description"])
              for d, s in zip(docs, standard)), "Standardized data differ from source")
    check(len(intents) == 200 and Counter(r["place_id"] for r in intents)
          == Counter({place: 2 for place in places}), "Expected two intents per place")
    check(len({r[f"query_{lang}"] for r in intents for lang in LANGUAGES}) == 600,
          "Expected 600 unique multilingual queries")
    print("PASS: 100 places, 300 aligned documents, 200 intents / 600 queries")

    base = RESULTS / "bge_m3_dense_full"
    natural = RESULTS / "tourapi_natural_query_eval_600"
    full_feature = RESULTS / "tourapi_full_feature_enrichment"
    meta = json.loads((base / "run_metadata.json").read_text(encoding="utf-8"))
    natural_meta = json.loads((natural / "metadata.json").read_text(encoding="utf-8"))
    check(sha256(pilot_path) == meta["corpus_sha256"], "Baseline corpus hash mismatch")
    check(sha256(base / "queries.csv") == meta["queries_sha256"], "Baseline query hash mismatch")
    check(sha256(input_query_path) == natural_meta["query_sha256"], "Natural query hash mismatch")
    check(meta["model"] == "BAAI/bge-m3" and meta["max_seq_length"] == 1024
          and meta["truncated_documents"] == 0, "Unexpected baseline settings")
    base_queries = rows(base / "queries.csv")
    check(len(base_queries) == 360, "Expected 360 initial queries")
    base_vec = embedding(base / "document_embeddings.npy", (300, 1024))
    base_query_vec = embedding(base / "query_embeddings.npy", (360, 1024))
    check(rank_index(evaluate(docs, base_queries, base_vec, base_query_vec, np))
          == rank_index(rows(base / "query_results.csv")), "Initial ranks do not match embeddings")
    print("PASS: source hashes and initial 360-query baseline verified")

    natural_queries = rows(natural / "queries.csv")
    check(len(natural_queries) == 600, "Expected 600 natural queries")
    expected_evaluations = {(r["query_id"], target) for r in natural_queries for target in TARGETS}
    query_vec = embedding(natural / "query_embeddings.npy", (600, 1024))
    enriched_vec = embedding(full_feature / "document_embeddings_enriched.npy", (300, 1024))
    comparison = rows(full_feature / "rank_comparison.csv")
    before = rank_index(comparison, "rank_before")
    after = rank_index(comparison, "rank_after")
    check(set(before) == expected_evaluations == set(after), "Missing evaluation combinations")
    check(rank_index(rows(natural / "query_results_original.csv")) == before,
          "Original ranks differ between analyses")
    check(rank_index(evaluate(docs, natural_queries, base_vec, query_vec, np)) == before,
          "Natural-query baseline ranks do not match embeddings")
    check(rank_index(evaluate(docs, natural_queries, enriched_vec, query_vec, np)) == after,
          "Enriched ranks do not match embeddings")
    check(sum(not np.array_equal(x, y) for x, y in zip(base_vec, enriched_vec)) == 68,
          "Unexpected number of changed document embeddings")
    summary = {r["group"]: r for r in rows(full_feature / "summary.csv")}
    for scope in ("MIXED", "CROSS_LANGUAGE", "GAP_TARGETS", "GAP_TARGETS_CROSS"):
        group = scope_rows(comparison, scope)
        saved = summary[f"ALL_{scope}"]
        check(len(group) == int(saved["query_evaluations"]), f"{scope}: count mismatch")
        metric_matches(group, "rank_before", saved, "_before")
        metric_matches(group, "rank_after", saved, "_after")
    print("PASS: 2,400 evaluation combinations, 68 changed documents, enrichment metrics")

    method_dir = RESULTS / "tourapi_enrichment_method_comparison"
    methods = rows(method_dir / "query_rank_comparison.csv")
    method_summary = rows(method_dir / "method_metrics.csv")
    for saved in method_summary:
        if saved["query_type"] != "ALL" or saved["scope"] not in (
            "MIXED", "CROSS_LANGUAGE", "GAP_TARGETS", "GAP_TARGETS_CROSS"
        ):
            continue
        group = scope_rows(methods, saved["scope"])
        check(len(group) == int(saved["evaluations"]), "Method comparison count mismatch")
        metric_matches(group, "rank_" + saved["method"].lower(), saved)
    print("PASS: selective and full-translation comparison metrics")

    hybrid_dir = RESULTS / "tourapi_hybrid_policy"
    hybrid = rows(hybrid_dir / "rank_comparison.csv")
    hybrid_summary = rows(hybrid_dir / "policy_performance.csv")
    for saved in hybrid_summary:
        group = scope_rows([r for r in hybrid if r["setting"] == saved["setting"]], saved["scope"])
        check(len(group) == int(saved["evaluations"]), "Hybrid comparison count mismatch")
        metric_matches(group, "rank_" + saved["method"].lower(), saved)
    print("PASS: both hybrid-policy settings and corresponding performance metrics")

    gap = summary["ALL_GAP_TARGETS"]
    print(f"Gap-target R@1: {float(gap['recall_at_1_before']):.2%} -> "
          f"{float(gap['recall_at_1_after']):.2%}")
    print("All saved-result checks completed. No files changed.")


if __name__ == "__main__":
    main()
