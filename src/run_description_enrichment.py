"""Paired intervention: add confirmed omitted source facts to four documents.

This is an exploratory, error-selected intervention, not a held-out test.
The same corpus, 360 queries, model, and candidate pools are reused.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_bge_m3 import DATA, MODEL, ROOT, encode, evaluate, metrics_by_group, load_baseline_model

STANDARD=ROOT/'data/processed/tourapi_standardized.csv'
BASE=ROOT/'results/bge_m3_dense_full'
OUT=ROOT/'results/tourapi_description_enrichment'
STANDARD_AUG=ROOT/'data/processed/tourapi_standardized_enriched_experiment.csv'

# Each sentence paraphrases an omitted fact in that place's supplied Korean
# description. We deliberately exclude English Owen: its reference to a white
# building conflicts with the Korean grey-brick description.
SUPPLEMENTS={
 ('125724','en'):('The cave takes its name from the Go family, who are said to have taken shelter here during the Japanese invasions of Korea in the late sixteenth century.','국문: 임진왜란 당시 고씨가족이 피난하였던 곳이라 하여 붙여진 이름'),
 ('125724','ja'):('高氏窟という名称は、壬辰倭乱（文禄・慶長の役）の時に高氏一族がこの洞窟に身を隠したという伝承に由来します。','국문: 임진왜란 당시 고씨가족이 피난하였던 곳이라 하여 붙여진 이름'),
 ('2605528','ja'):('現在は1970～80年代の街並みが再現され、線路沿いの古い商店で昔懐かしいお菓子などを楽しめます。当時の制服を着て記念撮影する体験も人気です。','국문: 1970~80년대 풍경 재현, 오래된 상점과 달고나, 옛 교복 사진 체험'),
 ('127074','ja'):('建物は灰色のレンガをオランダ式に積んだ二階建てで、礼拝のための説教壇を中心に左右対称の構造を備えています。','국문: 회색 벽돌의 2층 건물, 네덜란드식 벽돌쌓기, 설교단 중심 좌우대칭'),
}

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def write(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    docs=read(DATA)
    standard=read(STANDARD)
    queries=read(BASE/'queries.csv')
    meta=json.loads((BASE/'run_metadata.json').read_text())
    assert meta['model']==MODEL and meta['max_seq_length']==1024
    assert meta['corpus_sha256']==hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert meta['queries_sha256']==hashlib.sha256((BASE/'queries.csv').read_bytes()).hexdigest()
    assert len(docs)==len(standard)==300 and len(queries)==360
    assert all((d['place_id'],d['language'],d['title'],d['overview'])==(s['place_id'],s['language'],s['title'],s['description']) for d,s in zip(docs,standard))
    doc_vec=np.load(BASE/'document_embeddings.npy'); query_vec=np.load(BASE/'query_embeddings.npy')
    assert doc_vec.shape==(300,1024) and query_vec.shape==(360,1024)
    original=evaluate(docs,queries,doc_vec,query_vec,np)
    original_csv=read(BASE/'query_results.csv')
    assert len(original)==len(original_csv)==1440
    assert all(str(a['rank'])==b['rank'] and a['query_id']==b['query_id'] and a['target_language']==b['target_language'] for a,b in zip(original,original_csv))
    enriched=[]; changes=[]; indices=[]
    for i,(d,s) in enumerate(zip(docs,standard)):
        out=dict(s); k=(d['place_id'],d['language'])
        if k in SUPPLEMENTS:
            supplement,evidence=SUPPLEMENTS[k]
            out['description']+=' '+supplement
            changes.append(dict(place_id=k[0],language=k[1],title=d['title'],source_evidence_ko=evidence,added_text=supplement,original_description_chars=len(s['description']),enriched_description_chars=len(out['description'])))
            indices.append(i)
        enriched.append(out)
    assert len(changes)==4 and len(enriched)==300
    assert all(a[k]==b[k] for a,b in zip(standard,enriched) for k in ('source','place_id','language','title'))
    assert all(b['description'].startswith(a['description']) for a,b in zip(standard,enriched))
    assert sum(a!=b for a,b in zip(standard,enriched))==4
    write(STANDARD_AUG,enriched)
    write(OUT/'description_changes.csv',changes)
    model=load_baseline_model(meta)
    changed_text=[docs[i]['search_text']+' '+SUPPLEMENTS[docs[i]['place_id'],docs[i]['language']][0] for i in indices]
    assert all(len(model.tokenizer(t,add_special_tokens=True,truncation=False)['input_ids'])<=1024 for t in changed_text)
    after_vec=doc_vec.copy()
    after_vec[indices]=encode(model,changed_text,4,'enriched documents',np)
    untouched=[i for i in range(len(docs)) if i not in indices]
    assert np.array_equal(after_vec[untouched],doc_vec[untouched])
    after=evaluate(docs,queries,after_vec,query_vec,np)
    np.save(OUT/'document_embeddings_enriched.npy',after_vec)
    write(OUT/'query_results_enriched.csv',after)
    comparison=[]
    modified_places={p for p,l in SUPPLEMENTS}
    for a,b in zip(original,after):
        assert a['query_id']==b['query_id'] and a['target_language']==b['target_language']
        relevant=(a['place_id'],a['target_language']) in SUPPLEMENTS or (a['target_language']=='ALL_PLACES' and a['place_id'] in modified_places)
        delta=a['rank']-b['rank']
        comparison.append(dict(query_id=a['query_id'],query_type=a['query_type'],query_language=a['query_language'],target_language=a['target_language'],place_id=a['place_id'],query=a['query'],target_is_enriched='YES' if relevant else 'NO',rank_before=a['rank'],rank_after=b['rank'],rank_gain=delta,change='IMPROVED' if delta>0 else 'WORSE' if delta<0 else 'UNCHANGED',top1_before=a['top1_title'],top1_after=b['top1_title'],correct_score_before=a['correct_score'],correct_score_after=b['correct_score']))
    write(OUT/'all_rank_comparison.csv',comparison)
    targeted=[r for r in comparison if r['query_type']=='SEMANTIC' and r['target_is_enriched']=='YES']
    assert len(targeted)==21 # 4 document-language groups x 3 queries + 3 places x 3 mixed queries
    write(OUT/'targeted_semantic_comparison.csv',targeted)
    before_metrics=metrics_by_group(original); after_metrics=metrics_by_group(after)
    metrics=[]
    for a,b in zip(before_metrics,after_metrics):
        assert all(a[k]==b[k] for k in ('query_type','query_language','target_language','queries'))
        out={k:a[k] for k in ('query_type','query_language','target_language','queries')}
        for k in ('recall_at_1','recall_at_5','recall_at_10','mrr'):
            out.update({k+'_before':a[k],k+'_after':b[k],k+'_delta':round(b[k]-a[k],6)})
        metrics.append(out)
    write(OUT/'metrics_comparison.csv',metrics)
    summary=[]
    for name,subset in [('TARGETED_SINGLE_LANGUAGE',[r for r in targeted if r['target_language']!='ALL_PLACES']),('TARGETED_MIXED',[r for r in targeted if r['target_language']=='ALL_PLACES']),('ALL_SEMANTIC_MIXED',[r for r in comparison if r['query_type']=='SEMANTIC' and r['target_language']=='ALL_PLACES']),('ALL_SEMANTIC_SINGLE_LANGUAGE',[r for r in comparison if r['query_type']=='SEMANTIC' and r['target_language']!='ALL_PLACES'])]:
        n=len(subset)
        out=dict(group=name,queries=n,improved=sum(r['change']=='IMPROVED' for r in subset),worse=sum(r['change']=='WORSE' for r in subset),unchanged=sum(r['change']=='UNCHANGED' for r in subset))
        for suffix in ('before','after'):
            ranks=[r['rank_'+suffix] for r in subset]
            out['top1_'+suffix]=sum(r==1 for r in ranks)
            out['recall1_'+suffix]=round(sum(r==1 for r in ranks)/n,6)
            out['recall5_'+suffix]=round(sum(r<=5 for r in ranks)/n,6)
            out['mrr_'+suffix]=round(sum(1/r for r in ranks)/n,6)
        summary.append(out)
    write(OUT/'summary.csv',summary)
    (OUT/'metadata.json').write_text(json.dumps(dict(baseline=meta,changed_documents=len(changes),unchanged_embeddings=296,query_embeddings_reused=True,method='Append omitted details paraphrased from Korean descriptions to four English/Japanese docs. Same candidate pool, query vectors, max length, and remaining document vectors.',limitation='Error-selected pilot on evaluation queries; no independent test set, sham control, or causal proof. Added content not externally independently verified.'),ensure_ascii=False,indent=2),encoding='utf-8')
    for s in summary: print(s,flush=True)
    print(OUT,flush=True)

if __name__=='__main__': main()
