"""Development-only BGE-M3 baseline, same-language TourAPI retrieval."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/common_enrichment_corpus_variants.csv'
QUERIES = ROOT / 'results/common_pilot/main_dev_direct_queries.csv'
SPLITS = ROOT / 'data/processed/common_place_splits_balanced.csv'
OUT = ROOT / 'results/common_pilot'

def load(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--revision', default=None, help='Optional BGE-M3 model revision')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import torch
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    split_map = {r['place_id']: r['split'] for r in load(SPLITS)}
    docs = [dict(r, description=r['description_natural'] if split_map[r['place_id']] == 'dev' else r['description_baseline']) for r in load(DATA) if r['source'] == 'tourapi']
    queries = load(QUERIES)
    splits = {r['place_id']: r['split'] for r in load(SPLITS)}
    keys = [(r['place_id'], r['language']) for r in docs]
    assert len(docs) == 414 and len(set(keys)) == 414
    assert len({r['query_id'] for r in queries}) == len(queries)
    assert all(r['split'] == 'dev' and splits[r['place_id']] == 'dev' and r['query'].strip() for r in queries)
    assert all((r['place_id'], r['language']) in set(keys) for r in queries)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; use --device cpu')
    rank_path = OUT / 'main_dev_direct_format_natural.csv'
    details_path = OUT / 'main_dev_direct_format_natural_details.csv'
    meta_path = OUT / 'main_dev_direct_format_natural_metadata.json'
    if any(p.exists() for p in (rank_path, details_path, meta_path)):
        raise FileExistsError('Output exists; preserve or move all three output files before running.')
    kwargs = {'device': args.device, 'local_files_only': args.local_files_only}
    if args.revision: kwargs['revision'] = args.revision
    cache = Path.home() / '.cache/huggingface/hub/models--BAAI--bge-m3'
    ref = cache / 'refs/main'
    local_model = cache / 'snapshots' / ref.read_text().strip() if ref.exists() else None
    model = SentenceTransformer(str(local_model) if args.local_files_only and local_model and local_model.exists() else 'BAAI/bge-m3', **kwargs)
    model.max_seq_length = 2048
    doc_texts = [r['title'].strip() + '\n' + r['description'].strip() for r in docs]
    query_texts = [r['query'].strip() for r in queries]
    # Count lengths before truncation, including special tokens.
    doc_lengths = [len(model.tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in doc_texts]
    query_lengths = [len(model.tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in query_texts]
    print('Encoding documents...', flush=True)
    dv = model.encode(doc_texts, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    print('Encoding queries...', flush=True)
    qv = model.encode(query_texts, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    if not np.isfinite(dv).all() or not np.isfinite(qv).all(): raise ValueError('Non-finite embeddings')
    norms = np.concatenate([np.linalg.norm(dv, axis=1), np.linalg.norm(qv, axis=1)])
    if not np.allclose(norms, 1.0, atol=1e-3): raise ValueError('Embedding norms are invalid; rankings not saved')
    scores = torch.from_numpy(qv) @ torch.from_numpy(dv).T
    scores = scores.numpy()
    if not np.isfinite(scores).all() or np.max(np.abs(scores)) > 1.001: raise ValueError('Invalid cosine scores; rankings not saved')
    result, details = [], []
    for i, q in enumerate(queries):
        indices = [j for j, d in enumerate(docs) if d['language'] == q['language']]
        indices.sort(key=lambda j: (-float(scores[i, j]), docs[j]['place_id']))
        rank = next(k for k, j in enumerate(indices, 1) if docs[j]['place_id'] == q['place_id'])
        result.append(dict(query_id=q['query_id'], language=q['language'], field=q['field'], split='dev', rank_before=rank))
        details.append(dict(query_id=q['query_id'], place_id=q['place_id'], intent_id=q['intent_id'], query=q['query'], candidate_count=len(indices), top1_place_id=docs[indices[0]]['place_id'], top1_title=docs[indices[0]]['title'], rank_before=rank, relevance_review=q['relevance_review'], language_review=q['language_review']))
    for path, rows in ((rank_path, result), (details_path, details)):
        with path.open('x', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    metadata = {'model':'BAAI/bge-m3','requested_revision':args.revision,'device':args.device,'max_seq_length':2048,'document_input':'title + newline + natural-language full available enrichment; no address','candidate_scope':'138 same-language TourAPI documents, including test places as unchanged distractors','query_set':'directly enrichable development query drafts, previously inspected','document_count':len(docs),'query_count':len(queries),'truncated_documents':sum(n>2048 for n in doc_lengths),'truncated_queries':sum(n>2048 for n in query_lengths),'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (DATA,QUERIES,SPLITS)},'torch_version':torch.__version__,'sentence_transformers_version':sentence_transformers.__version__,'rank_tie_break':'place_id ascending','note':'Development only; not independent gold evaluation.'}
    with meta_path.open('x', encoding='utf-8') as f: json.dump(metadata,f,ensure_ascii=False,indent=2)
    print('Done:',rank_path,'queries:',len(result),'truncated documents:',metadata['truncated_documents'])

if __name__ == '__main__': main()
