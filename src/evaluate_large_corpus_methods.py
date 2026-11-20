"""Retest previously saved translation strategies in the expanded candidate pool.

All arms retain the same 17,738 non-gold records, including 63 existing
non-gold descriptions. Only previously computed gold document vectors
change. This is a robustness check, not an independent translation study.
"""

import json
from pathlib import Path

import numpy as np

from prepare_large_corpus import OUT, PROCESSED, RESULTS, BASE, QUERIES, read, write, write_json, sha256, key
from evaluate_large_corpus import rank_condition, metric, keyed, pair_rows, belongs
from statistical_validate_enrichment import paired_test, holm

MT = RESULTS / "tourapi_enrichment_method_comparison"
QC = RESULTS / "tourapi_quality_controlled_sample"
HY = RESULTS / "tourapi_hybrid_policy"
FEATURE = RESULTS / "tourapi_full_feature_enrichment"
COVERAGE = RESULTS / "tourapi_coverage_retrieval_analysis"
SCOPES = ("MIXED", "SAME_LANGUAGE", "CROSS_LANGUAGE", "GAP_TARGETS", "GAP_TARGETS_CROSS")


def vectors(path):
    arr = np.load(path, allow_pickle=False)
    assert arr.shape == (300, 1024) and np.isfinite(arr).all()
    return arr


def main():
    metadata = json.loads((OUT / "experiment_metadata.json").read_text())
    assert metadata["source_records"] == 18038
    index = read(OUT / "corpus_index.csv")
    gold = read(PROCESSED / "tourapi_pilot_details_300.csv")
    queries = read(QUERIES / "queries.csv")
    source_positions = {key(r): i for i, r in enumerate(index)}
    gold_positions = [source_positions[key(r)] for r in gold]
    extra_positions = [int(r["embedding_row"]) for r in read(OUT / "extra_detail_index.csv")]
    assert len(index) == 18038 and len(gold_positions) == 300 and len(extra_positions) == 63
    assert set(gold_positions).isdisjoint(extra_positions)
    all_positions = list(range(18038))
    query_vec = np.load(QUERIES / "query_embeddings.npy", allow_pickle=False)
    metadata_vec = np.load(OUT / "metadata_embeddings.npy", mmap_mode="r", allow_pickle=False)
    extra_vec = np.load(OUT / "extra_detail_embeddings.npy", allow_pickle=False)
    base = vectors(BASE / "document_embeddings.npy")
    assert query_vec.shape == (600, 1024) and metadata_vec.shape == (18038, 1024)
    assert extra_vec.shape == (63, 1024)
    scores = query_vec @ metadata_vec.T
    scores[:, extra_positions] = query_vec @ extra_vec.T
    coverage = {(r["query_id"], r["target_language"]): r["coverage_group"]
                for r in read(COVERAGE / "coverage_rank_joined.csv") if r["eligible_primary"] == "YES"}

    compare_saved = read(MT / "query_rank_comparison.csv")
    hybrid_saved = read(HY / "rank_comparison.csv")
    mt_work = {r["method"]: r for r in read(MT / "translation_workload.csv")}
    hy_work = {(r["setting"], r["method"]): r for r in read(HY / "policy_performance.csv")
               if r["scope"] == "MIXED"}
    method_settings = {
        "MAIN_68_AND_200": [
            ("BASELINE", BASE / "document_embeddings.npy", "BASELINE", compare_saved, "rank_baseline"),
            ("SELECTIVE_AI_68", FEATURE / "document_embeddings_enriched.npy", "SELECTIVE",
             compare_saved, "rank_selective"),
            ("SELECTIVE_NLLB_68", MT / "document_embeddings_selective_mt_68.npy", "SELECTIVE_MT_68",
             compare_saved, "rank_selective_mt_68"),
            ("FULL_NLLB_68", MT / "document_embeddings_full_mt_68.npy", "FULL_MT_68",
             compare_saved, "rank_full_mt_68"),
            ("FULL_NLLB_200", MT / "document_embeddings_full_mt_200.npy", "FULL_MT_200",
             compare_saved, "rank_full_mt_200"),
            ("HYBRID_NLLB_68", HY / "nllb_68_hybrid_embeddings.npy", None,
             [r for r in hybrid_saved if r["setting"] == "NLLB_68"], "rank_hybrid"),
        ],
        "SOURCE_CHECKED_20": [
            ("BASELINE", BASE / "document_embeddings.npy", "BASELINE", None, "rank_baseline"),
            ("SELECTIVE", QC / "document_embeddings_source_checked_selective_20.npy", "SELECTIVE", None,
             "rank_selective"),
            ("FULL", QC / "document_embeddings_source_checked_full_20.npy", "FULL", None, "rank_full"),
            ("HYBRID", HY / "source_checked_20_hybrid_embeddings.npy", "HYBRID", None, "rank_hybrid"),
        ],
    }
    by_method = {}
    summary = []
    ranks_full = []
    signatures = {}
    for setting, methods in method_settings.items():
        sample_ref = [r for r in hybrid_saved if r["setting"] == "SOURCE_CHECKED_20"]
        reference = compare_saved if setting == "MAIN_68_AND_200" else sample_ref
        gaps = {(r["query_id"], r["target_language"]): r["gap_target"] for r in reference}
        assert len(gaps) == 2400
        assert sum(v == "YES" for v in gaps.values()) == (219 if setting == "MAIN_68_AND_200" else 69)
        for method, path, workload, saved, rank_col in methods:
            print(f"Full-corpus strategy: {setting} / {method}", flush=True)
            current = vectors(path)
            signatures[str(path.relative_to(RESULTS))] = sha256(path)
            changed = sum(not np.array_equal(x, y) for x, y in zip(current, base))
            if setting == "MAIN_68_AND_200":
                if workload is None:
                    w = hy_work["NLLB_68", "HYBRID"]
                    tokens = int(w["translated_output_tokens"])
                    assert changed == int(w["full_docs"])+int(w["selective_docs"])
                else:
                    w = mt_work[workload]
                    tokens = int(w["newly_translated_output_bge_tokens"])
                    assert changed == int(w["foreign_documents_changed"])
            else:
                w = hy_work["SOURCE_CHECKED_20", workload]
                tokens = int(w["translated_output_tokens"])
                assert changed == int(w["full_docs"])+int(w["selective_docs"])

            scores[:, gold_positions] = query_vec @ current.T
            pilot = rank_condition(scores, index, queries, gold_positions, method, "PILOT_300", gaps, coverage)
            full = rank_condition(scores, index, queries, all_positions, method, "FULL_18038", gaps, coverage)
            pilot_index, full_index = keyed(pilot), keyed(full)
            reference_rows = saved if saved is not None else sample_ref
            assert len(reference_rows) == 2400
            assert all(pilot_index[r["query_id"], r["target_language"]]["rank"] == int(r[rank_col])
                       for r in reference_rows), f"{setting}/{method}: original pilot ranks changed"
            assert all(full_index[k]["rank"] >= r["rank"] for k, r in pilot_index.items())
            by_method[setting, method] = full
            ranks_full.extend(dict(setting=setting, **r) for r in full)
            for corpus, rows in (("PILOT_300", pilot), ("FULL_18038", full)):
                for scope in SCOPES:
                    part = [r for r in rows if belongs(r, scope)]
                    summary.append(dict(setting=setting, method=method, corpus=corpus, scope=scope,
                                        changed_gold_documents=changed, translated_output_tokens=tokens,
                                        **metric(part)))
    write(OUT / "method_strategy_metrics.csv", summary)
    write(OUT / "method_strategy_rank_results.csv", ranks_full)

    # Same-engine and same-scope comparisons only; a small, exploratory family.
    rng = np.random.default_rng(20261121)
    tests = []
    for setting, pairs in (
        ("MAIN_68_AND_200", (("SELECTIVE_NLLB_68", "FULL_NLLB_68"),
                            ("FULL_NLLB_68", "HYBRID_NLLB_68"),
                            ("SELECTIVE_NLLB_68", "HYBRID_NLLB_68"))),
        ("SOURCE_CHECKED_20", (("SELECTIVE", "FULL"), ("FULL", "HYBRID"), ("SELECTIVE", "HYBRID"))),
    ):
        for before, after in pairs:
            paired = pair_rows(by_method[setting, before], by_method[setting, after])
            for scope in ("GAP_TARGETS", "MIXED"):
                rows = [r for r in paired if belongs(r, scope)]
                test = paired_test(f"{setting}:{after}_MINUS_{before}:{scope}", rows, "R@1", rng)
                test.update(setting=setting, before_method=before, after_method=after, scope=scope)
                tests.append(test)
    for row, adjusted in zip(tests, holm([r["raw_p_value"] for r in tests])):
        row["holm_adjusted_p_value"] = adjusted
        row["holm_p_lt_0_05"] = "YES" if adjusted < .05 else "NO"
    write(OUT / "method_strategy_tests.csv", tests)
    write_json(OUT / "method_strategy_metadata.json", dict(
        conditions="Same available-information 18,038-record corpus; only precomputed gold vectors change.",
        source_records=18038, gold_places=100, queries=600, methods_per_setting={k: len(v)
                                                                               for k,v in method_settings.items()},
        original_pilot_ranks_verified=True, method_vector_sha256=signatures,
        limitations=[
            "This is a retest of the existing interventions, not an independent holdout.",
            "Only 363 original descriptions are available. Gold documents have much richer information than non-gold records.",
            "Full translation of 200 refers to the 200 foreign-language gold documents, not all 18,038 records.",
            "The 20-document source-checked sample differs from the 68-document experiment; do not compare their raw target-group metrics as if on the same sample.",
            "Token volume is a workload proxy, not measured money or human time; no equivalence margin is specified.",
        ],
    ))
    for r in summary:
        if r["corpus"] == "FULL_18038" and r["scope"] in ("MIXED", "GAP_TARGETS"):
            print(r["setting"], r["method"], r["scope"], f"R@1={r['recall_at_1']:.2%}", flush=True)
    print("Full-corpus translation strategy comparison complete.", flush=True)


if __name__ == "__main__":
    main()
