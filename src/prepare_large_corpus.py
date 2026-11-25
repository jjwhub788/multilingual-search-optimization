"""Prepare and encode the 18,038-record TourAPI candidate corpus.

The original 100-place evaluation set and its 600 queries are held fixed.
All records have a metadata-only representation. Existing, genuinely cached
descriptions are used in a separate available-information condition; no API
requests or synthetic descriptions are generated here.

Long embedding runs are checkpointed and can be resumed with the same inputs.
"""

import argparse
import csv
import hashlib
import html
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from run_bge_m3 import MODEL

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "tourapi"
RESULTS = ROOT / "results"
OUT = RESULTS / "tourapi_large_corpus_eval"
BASE = RESULTS / "bge_m3_dense_full"
QUERIES = RESULTS / "tourapi_natural_query_eval_600"
LANGS = ("ko", "en", "ja")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row["language"], row["source_contentid"]


def clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(str(text)))).strip()


def prepare():
    source_path = PROCESSED / "tourapi_clean.csv"
    gold_path = PROCESSED / "tourapi_pilot_details_300.csv"
    source = read(source_path)
    gold = read(gold_path)
    matches = read(PROCESSED / "tourapi_final_matches.csv")
    audits = read(PROCESSED / "tourapi_sample_review_20.csv")
    source_by_key = {key(r): r for r in source}
    gold_by_key = {key(r): r for r in gold}
    assert len(source) == len(source_by_key) == 18038
    assert len(gold) == len(gold_by_key) == 300 and set(gold_by_key) <= set(source_by_key)
    assert len({r["place_id"] for r in gold}) == 100
    assert all(r["collection_status"] == "COMPLETE" and r["overview"].strip() for r in gold)
    assert all(
        all(r[c] == source_by_key[key(r)][c] for c in ("title", "addr1", "addr2", "mapx", "mapy"))
        for r in gold
    )
    assert not any(r["overview"].strip() for r in source), "Unexpected original metadata schema"

    # Group only the previously matched triples, excluding the flagged sample.
    # Unlinked records remain separate; do not infer 17,378 unique real places.
    held = {r["id_ko"] for r in audits if r["sample_review_status"] != "MATCH"}
    matched = {}
    for r in matches:
        if r["id_ko"] in held:
            continue
        for lang in LANGS:
            k = (lang, r[f"id_{lang}"])
            assert k in source_by_key and k not in matched
            matched[k] = r["id_ko"]
    assert len(held) == 1 and len(matched) == 990
    assert all(matched[k] == r["place_id"] for k, r in gold_by_key.items())

    # Same cache precedence as the original pilot collector: no new requests.
    cached = {}
    for r in read(PROCESSED / "tourapi_enriched.csv"):
        if r["overview"].strip():
            cached[key(r)] = r["overview"]
    with (RAW / "detail_overviews.jsonl").open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                text = clean_text(r["overview"])
                k = (r["language"], str(r["source_contentid"]))
                if text or k not in cached:
                    cached[k] = text
            except (ValueError, KeyError, TypeError):
                continue
    assert all(cached.get(k) == r["overview"] for k, r in gold_by_key.items())

    index = []
    for i, r in enumerate(source):
        k = key(r)
        target = gold_by_key.get(k)
        detail = target["overview"] if target else cached.get(k, "")
        group = f"place:{matched[k]}" if k in matched else f"record:{k[0]}:{k[1]}"
        index.append(dict(
            embedding_row=i, language=k[0], source_contentid=k[1],
            candidate_group_id=group, matched_place_id=matched.get(k, ""),
            gold_place_id=target["place_id"] if target else "",
            title=r["title"], addr1=r["addr1"], addr2=r["addr2"],
            metadata_text=r["search_text"],
            description_available="YES" if detail else "NO",
            description_source="FROZEN_GOLD" if target else "EXISTING_CACHE" if detail else "NONE",
            available_text=target["search_text"] if target else clean_text(r["search_text"] + " " + detail)
            if detail else r["search_text"],
        ))
    assert len({r["candidate_group_id"] for r in index}) == 17378
    assert all(r["metadata_text"].strip() for r in index)
    assert sum(r["description_available"] == "YES" for r in index) == 363
    assert sum(r["description_source"] == "EXISTING_CACHE" for r in index) == 63

    data_signature = {
        "clean_source_sha256": sha256(source_path),
        "gold_corpus_sha256": sha256(gold_path),
        "cache_sha256": sha256(RAW / "detail_overviews.jsonl"),
        "enriched_source_sha256": sha256(PROCESSED / "tourapi_enriched.csv"),
        "matches_sha256": sha256(PROCESSED / "tourapi_final_matches.csv"),
        "sample_audit_sha256": sha256(PROCESSED / "tourapi_sample_review_20.csv"),
        "natural_queries_sha256": sha256(QUERIES / "queries.csv"),
        "query_embeddings_sha256": sha256(QUERIES / "query_embeddings.npy"),
        "original_gold_embeddings_sha256": sha256(BASE / "document_embeddings.npy"),
        "enriched_gold_embeddings_sha256": sha256(
            RESULTS / "tourapi_full_feature_enrichment" / "document_embeddings_enriched.npy"
        ),
    }
    # The rows include the exact text and order used for both retrieval conditions.
    index_sha = hashlib.sha256(json.dumps(index, ensure_ascii=False).encode()).hexdigest()
    data_signature["candidate_index_sha256"] = index_sha
    coverage = []
    for lang in (*LANGS, "ALL"):
        part = [r for r in index if lang == "ALL" or r["language"] == lang]
        n = len(part)
        detail_n = sum(r["description_available"] == "YES" for r in part)
        gold_n = sum(bool(r["gold_place_id"]) for r in part)
        extra = sum(r["description_source"] == "EXISTING_CACHE" for r in part)
        coverage.append(dict(
            language=lang, records=n, with_title=sum(bool(r["title"].strip()) for r in part),
            with_address=sum(bool(r["addr1"].strip() or r["addr2"].strip()) for r in part),
            with_existing_description=detail_n, without_description=n-detail_n,
            description_coverage=round(detail_n/n, 8), gold_records=gold_n,
            non_gold_cached_descriptions=extra,
            non_gold_description_coverage=round(extra/(n-gold_n), 8),
            known_linked_records=sum(bool(r["matched_place_id"]) for r in part),
        ))
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "corpus_index.csv", index)
    write(OUT / "corpus_coverage.csv", coverage)
    write_json(OUT / "corpus_metadata.json", dict(
        source_documents=len(index), gold_places=100, gold_documents=300,
        matched_triples=330, excluded_flagged_triples=len(held), mixed_candidate_groups=17378,
        detailed_documents=363, gold_detailed_documents=300, other_detailed_documents=63,
        data_signature=data_signature,
        limitations=[
            "The 18,038 items are source records, not 18,038 unique places.",
            "Only 330 previously matched triples are grouped; unlinked records remain individual candidates.",
            "All 18,038 records have metadata; only 363 have existing descriptions, of which 300 are gold records.",
            "Available-description condition is intentionally separated from the uniform metadata condition.",
            "The 100-place reference evaluation set is purposively selected and AI-assisted, not independently human-validated.",
        ],
    ))
    return index, gold, data_signature


def encode(model, texts, batch_size):
    array = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    assert array.shape == (len(texts), 1024) and np.isfinite(array).all()
    return array


def compatibility(model, gold):
    """Check fresh outputs against saved study vectors before sharing embeddings."""
    old_gold = np.load(BASE / "document_embeddings.npy", allow_pickle=False)
    old_query = np.load(QUERIES / "query_embeddings.npy", allow_pickle=False)
    queries = read(QUERIES / "queries.csv")
    assert old_gold.shape == (300, 1024) and old_query.shape == (600, 1024)
    gold_sample = np.linspace(0, 299, num=9, dtype=int)
    query_sample = np.linspace(0, 599, num=9, dtype=int)
    current = np.concatenate([
        encode(model, [gold[i]["search_text"] for i in gold_sample], 4),
        encode(model, [queries[i]["query"] for i in query_sample], 4),
    ])
    older = np.concatenate([old_gold[gold_sample], old_query[query_sample]])
    cos = (current * older).sum(axis=1) / (
        np.linalg.norm(current, axis=1) * np.linalg.norm(older, axis=1)
    )
    minimum = float(cos.min())
    print(f"Saved-vector compatibility: min cosine={minimum:.8f}", flush=True)
    if minimum < 0.999:
        raise RuntimeError(
            "Cached BGE-M3 weights/settings did not reproduce the original embeddings closely. "
            "Do not mix them; re-encode the fixed gold corpus and queries consistently."
        )
    return dict(gold_sample_indices=gold_sample.tolist(), query_sample_indices=query_sample.tolist(),
                minimum_cosine=minimum, mean_cosine=float(cos.mean()), threshold=0.999)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = p.parse_args()
    if args.batch_size < 1 or args.chunk_size < 1:
        p.error("Batch and chunk sizes must be positive")
    index, gold, signature = prepare()
    print(f"Corpus: {len(index):,} documents; 100 evaluation places; 363 existing descriptions.", flush=True)
    if args.prepare_only:
        return

    import torch
    from sentence_transformers import SentenceTransformer

    meta = json.loads((BASE / "run_metadata.json").read_text())
    if (args.device, "float16" if args.device == "mps" else "float32") != (
        meta["device"], meta["precision"]
    ):
        raise ValueError("The full-corpus run must match the saved gold embeddings' device and precision.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is required to reuse the original MPS vectors.")

    print(f"Loading cached {MODEL} on {args.device}...", flush=True)
    model = SentenceTransformer(MODEL, device=args.device, local_files_only=True,
                                model_kwargs={"torch_dtype": meta["precision"]})
    model.max_seq_length = meta["max_seq_length"]
    check = compatibility(model, gold)
    settings = dict(model=MODEL, device=args.device, precision=meta["precision"],
                    max_seq_length=model.max_seq_length,
                    model_revision=str(getattr(model[0].auto_model.config, "_commit_hash", None)))
    progress_file = OUT / "embedding_progress.json"
    emb_file = OUT / "metadata_embeddings.npy"
    n = len(index)
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        if progress["data_signature"] != signature or progress["settings"] != settings:
            raise ValueError("Input or model settings changed; preserve prior vectors in a separate run directory.")
        vectors = np.lib.format.open_memmap(emb_file, mode="r+")
        start = int(progress["completed_metadata_documents"])
        assert vectors.shape == (n, 1024) and 0 <= start <= n
        print(f"Resuming from {start:,}/{n:,}", flush=True)
    else:
        if emb_file.exists():
            raise FileExistsError("Embedding file exists without progress metadata")
        vectors = np.lib.format.open_memmap(emb_file, mode="w+", dtype=np.float32, shape=(n, 1024))
        start = 0
        progress = dict(data_signature=signature, settings=settings, compatibility=check,
                        completed_metadata_documents=0, extra_details_complete=False)
        write_json(progress_file, progress)
    t0 = time.monotonic()
    for off in range(start, n, args.chunk_size):
        end = min(off + args.chunk_size, n)
        batch = encode(model, [r["metadata_text"] for r in index[off:end]], args.batch_size)
        vectors[off:end] = batch
        vectors.flush()
        progress["completed_metadata_documents"] = end
        progress["batch_size"] = args.batch_size
        write_json(progress_file, progress)
        elapsed = time.monotonic() - t0
        rate = (end - start) / max(elapsed, .001)
        eta = (n - end) / rate / 60
        print(f"Metadata: {end:,}/{n:,} | {rate:.1f} docs/sec | ETA {eta:.1f} min", flush=True)
    del vectors

    extra = [r for r in index if r["description_source"] == "EXISTING_CACHE"]
    extra_path = OUT / "extra_detail_embeddings.npy"
    if not progress["extra_details_complete"]:
        print(f"Encoding {len(extra)} existing non-gold descriptions...", flush=True)
        # Same 1024-token limit; few full descriptions use a small batch.
        lengths = [len(model.tokenizer(r["available_text"], add_special_tokens=True,
                                       truncation=False, verbose=False)["input_ids"]) for r in extra]
        if max(lengths) > model.max_seq_length:
            raise ValueError("A cached description exceeds the common token limit.")
        arr = encode(model, [r["available_text"] for r in extra], 4)
        np.save(extra_path, arr)
        write(OUT / "extra_detail_index.csv", [
            dict(extra_embedding_row=i, embedding_row=r["embedding_row"],
                 language=r["language"], source_contentid=r["source_contentid"])
            for i, r in enumerate(extra)
        ])
        progress["extra_details_complete"] = True
        progress["max_extra_detail_tokens"] = max(lengths)
        write_json(progress_file, progress)
    assert np.load(extra_path, mmap_mode="r", allow_pickle=False).shape == (63, 1024)
    print("Full corpus vectors are ready.", flush=True)


if __name__ == "__main__":
    main()
