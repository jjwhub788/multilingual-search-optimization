"""Evaluate 600 realistic AI-generated queries on frozen original/enriched corpora."""
import csv
import hashlib
import json
from collections import Counter
import numpy as np
from run_bge_m3 import DATA, MODEL, ROOT, encode, evaluate, metrics_by_group, load_baseline_model
from run_description_enrichment import SUPPLEMENTS

QUERY_FILE=ROOT/'data/processed/tourapi_ai_generated_queries_600_natural.csv'
BASE=ROOT/'results/bge_m3_dense_full'
AUG=ROOT/'results/tourapi_description_enrichment'
OUT=ROOT/'results/tourapi_natural_query_eval_600'
LANGS=('ko','en','ja')

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(rows,group):
    n=len(rows)
    out=dict(group=group,query_evaluations=n,improved=sum(r['rank_after']<r['rank_before'] for r in rows),worse=sum(r['rank_after']>r['rank_before'] for r in rows),unchanged=sum(r['rank_after']==r['rank_before'] for r in rows))
    for suffix in ('before','after'):
        ranks=[r['rank_'+suffix] for r in rows]
        for k in (1,5,10):
            hits=sum(r<=k for r in ranks)
            out[f'hits_at_{k}_{suffix}']=hits
            out[f'recall_at_{k}_{suffix}']=round(hits/n,6)
        out['mrr_'+suffix]=round(sum(1/r for r in ranks)/n,6)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    docs=read(DATA); input_rows=read(QUERY_FILE)
    meta=json.loads((BASE/'run_metadata.json').read_text())
    assert meta['corpus_sha256']==hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert meta['model']==MODEL and meta['max_seq_length']==1024
    doc_by_place={d['place_id']:d for d in docs if d['language']=='ko'}
    index=read(BASE/'embedding_index.csv')
    assert len(docs)==len(index)==300 and all((a['place_id'],a['language'],a['source_contentid'])==(b['place_id'],b['language'],b['source_contentid']) for a,b in zip(docs,index))
    before_vec=np.load(BASE/'document_embeddings.npy');after_vec=np.load(AUG/'document_embeddings_enriched.npy')
    assert before_vec.shape==after_vec.shape==(300,1024)
    changed={(d['place_id'],d['language']) for i,d in enumerate(docs) if not np.array_equal(before_vec[i],after_vec[i])}
    assert changed==set(SUPPLEMENTS)
    assert len(input_rows)==200 and set(Counter(r['place_id'] for r in input_rows).values())=={2}
    assert set(r['place_id'] for r in input_rows)==set(doc_by_place)
    queries=[]
    for r in input_rows:
        doc=doc_by_place[r['place_id']]
        assert (r['title_ko'],r['reference_description_ko'])==(doc['title'],doc['overview'])
        kind='REPRESENTATIVE' if r['intent_id'].endswith('-1') else 'DETAIL'
        for lang in LANGS:
            queries.append(dict(query_id=r['intent_id']+'-'+lang,query_type=kind,query_language=lang,place_id=r['place_id'],query=r['query_'+lang]))
    assert len(queries)==600 and len({r['query_id'] for r in queries})==600 and len({r['query'] for r in queries})==600
    write(OUT/'queries.csv',queries)
    model=load_baseline_model(meta)
    qvec=encode(model,[q['query'] for q in queries],4,'natural queries',np)
    np.save(OUT/'query_embeddings.npy',qvec)
    before=evaluate(docs,queries,before_vec,qvec,np);after=evaluate(docs,queries,after_vec,qvec,np)
    assert len(before)==len(after)==2400
    write(OUT/'query_results_original.csv',before);write(OUT/'query_results_enriched.csv',after)
    comp=[]; modified_places={p for p,l in SUPPLEMENTS}
    for a,b in zip(before,after):
        assert (a['query_id'],a['target_language'])==(b['query_id'],b['target_language'])
        gain=a['rank']-b['rank']
        comp.append(dict(query_id=a['query_id'],query_type=a['query_type'],query_language=a['query_language'],target_language=a['target_language'],place_id=a['place_id'],correct_title_ko=doc_by_place[a['place_id']]['title'],query=a['query'],target_enriched='YES' if (a['place_id'],a['target_language']) in SUPPLEMENTS or (a['place_id'] in modified_places and a['target_language']=='ALL_PLACES') else 'NO',rank_before=a['rank'],rank_after=b['rank'],rank_gain=gain,change='IMPROVED' if gain>0 else 'WORSE' if gain<0 else 'UNCHANGED',top1_place_id_before=a['top1_place_id'],top1_place_id_after=b['top1_place_id'],top1_before=a['top1_title'],top1_after=b['top1_title'],correct_score_before=a['correct_score'],correct_score_after=b['correct_score']))
    write(OUT/'rank_comparison.csv',comp)
    write(OUT/'original_top1_misses.csv',[r for r in comp if r['rank_before']!=1])
    metrics=[]
    for a,b in zip(metrics_by_group(before),metrics_by_group(after)):
        out={k:a[k] for k in ('query_type','query_language','target_language','queries')}
        assert all(a[k]==b[k] for k in out)
        for k in ('recall_at_1','recall_at_5','recall_at_10','mrr'):
            out.update({k+'_before':a[k],k+'_after':b[k],k+'_delta':round(b[k]-a[k],6)})
        metrics.append(out)
    write(OUT/'language_metrics_comparison.csv',metrics)
    groups=[]
    for kind in ('REPRESENTATIVE','DETAIL','ALL'):
        subset=[r for r in comp if kind=='ALL' or r['query_type']==kind]
        for label,filter_fn in [
            ('MIXED',lambda r:r['target_language']=='ALL_PLACES'),
            ('SINGLE_LANGUAGE',lambda r:r['target_language'] in LANGS),
            ('SAME_LANGUAGE',lambda r:r['target_language']==r['query_language']),
            ('CROSS_LANGUAGE',lambda r:r['target_language'] in LANGS and r['target_language']!=r['query_language']),
            ('ENRICHED_SINGLE_TARGETS',lambda r:(r['place_id'],r['target_language']) in SUPPLEMENTS),
            ('ENRICHED_MIXED_TARGETS',lambda r:r['target_language']=='ALL_PLACES' and r['place_id'] in modified_places),
        ]:
            group=[r for r in subset if filter_fn(r)]
            if group: groups.append(summarize(group,kind+'_'+label))
    write(OUT/'summary.csv',groups)
    (OUT/'metadata.json').write_text(json.dumps(dict(query_sha256=hashlib.sha256(QUERY_FILE.read_bytes()).hexdigest(),source_query_count=600,place_count=100,model=MODEL,original_documents=300,enriched_documents=4,embedding_revision=meta,limitations=['AI-generated, Korean-description-grounded queries, not independent human queries.','One designated target per query; broad phrases may admit other relevant places. One-target metrics measure recovery of assigned place, not exhaustive relevance.','Representative/detail query intents differ; any performance difference is not caused solely by query length/style.','The baseline input includes title, address and description from earlier experiment; same input schema is used in both arms.']),ensure_ascii=False,indent=2),encoding='utf-8')
    for g in groups:
        print(g['group'],g['query_evaluations'],'R@1',g['recall_at_1_before'],g['recall_at_1_after'],'R@5',g['recall_at_5_before'],g['recall_at_5_after'],'MRR',g['mrr_before'],g['mrr_after'],'improved/worse',g['improved'],g['worse'],flush=True)
    print(OUT,flush=True)

if __name__=='__main__':main()
