"""Compare original, selective supplementation, full MT and scope-matched full MT.

All methods reuse the same BGE-M3 query vectors, titles, addresses, and ranks
among the same 100 places. Only the intended descriptions change.
"""
import csv
import hashlib
import html
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from run_bge_m3 import DATA,MODEL,ROOT,encode,evaluate,metrics_by_group,load_baseline_model
from statistical_validate_enrichment import paired_test,holm

BASE=ROOT/'results/bge_m3_dense_full'
NAT=ROOT/'results/tourapi_natural_query_eval_600'
SELECTIVE=ROOT/'results/tourapi_full_feature_enrichment'
FULL=ROOT/'results/tourapi_full_translation'
OUT=ROOT/'results/tourapi_enrichment_method_comparison'
RAW=ROOT/'data/processed/tourapi_standardized.csv'
AUG=ROOT/'data/processed/tourapi_standardized_enriched_68_experiment.csv'
MT=ROOT/'data/processed/tourapi_standardized_full_translation_experiment.csv'
METHODS=('BASELINE','SELECTIVE','SELECTIVE_MT_68','FULL_MT_68','FULL_MT_200')
LANGS=('ko','en','ja')

# Source sentence selection by missing feature, checked against the Korean
# segments before comparing rankings. Using exactly the NLLB outputs of the
# full translation makes SELECTIVE_MT_68 vs FULL_MT_68 engine- and scope-matched.
SOURCE_SEGMENTS={
 'B01-1':(4,), 'B02-2':(6,9), 'B03-2':(5,6), 'B07-2':(2,),
 'B09-2':(4,), 'B10-2':(7,9), 'B11-1':(1,), 'B11-2':(3,4),
 'B12-1':(3,4), 'B12-2':(4,), 'B14-2':(2,), 'B18-2':(7,),
 'B19-2':(1,), 'B26-2':(5,), 'B30-2':(1,), 'B32-2':(4,),
 'B40-1':(2,4), 'B42-2':(1,), 'B43-1':(5,), 'B44-2':(2,),
 'B46-1':(1,), 'B46-2':(2,), 'B48-2':(3,), 'B53-2':(4,),
 'B57-2':(1,), 'B58-2':(1,), 'B60-1':(4,), 'B61-1':(1,),
 'B62-2':(2,), 'B64-2':(2,), 'B65-2':(2,), 'B68-2':(1,),
 'B72-1':(1,), 'B73-2':(2,), 'B74-1':(2,), 'B76-2':(2,),
 'B77-1':(3,), 'B80-1':(1,2), 'B82-1':(1,), 'B83-2':(5,),
 'B84-2':(2,4), 'B87-2':(2,4), 'B88-2':(2,), 'B90-2':(3,7),
 'B91-2':(4,), 'B94-1':(6,), 'B98-2':(5,), 'B100-2':(2,4,5),
}

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def clean(value):
    return re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',html.unescape(value))).strip()

def metrics(rows):
    ranks=[int(r['rank']) for r in rows];n=len(rows)
    assert n and max(ranks)<=100
    out=dict(evaluations=n,distinct_places=len({r['place_id'] for r in rows}))
    for k in (1,5,10):
        hits=sum(r<=k for r in ranks);out[f'hits_at_{k}']=hits;out[f'recall_at_{k}']=round(hits/n,6)
    out['mrr']=round(statistics.mean(1/r for r in ranks),6)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    docs=read(DATA);raw=read(RAW);aug=read(AUG);mt=read(MT);queries=read(NAT/'queries.csv')
    assert len(docs)==len(raw)==len(aug)==len(mt)==300 and len(queries)==600
    for i,(d,a,b,c) in enumerate(zip(docs,raw,aug,mt)):
        assert (d['place_id'],d['language'],d['title'],d['overview'])==(a['place_id'],a['language'],a['title'],a['description'])
        assert all(a[k]==b[k]==c[k] for k in ('source','place_id','language','title'))
        assert b['description'].startswith(a['description']) and c['description'].strip()
    selected=[i for i,(a,b) in enumerate(zip(raw,aug)) if a!=b];translated=[i for i,(a,c) in enumerate(zip(raw,mt)) if a!=c]
    assert len(selected)==68 and len(translated)==200 and set(selected)<=set(translated)
    assert {docs[i]['language'] for i in translated}=={'en','ja'}
    base_vec=np.load(BASE/'document_embeddings.npy');select_vec=np.load(SELECTIVE/'document_embeddings_enriched.npy');query_vec=np.load(NAT/'query_embeddings.npy')
    assert base_vec.shape==select_vec.shape==(300,1024) and query_vec.shape==(600,1024)
    assert {i for i in range(300) if not np.array_equal(base_vec[i],select_vec[i])}==set(selected)
    meta=json.loads((BASE/'run_metadata.json').read_text());assert meta['corpus_sha256']==hashlib.sha256(DATA.read_bytes()).hexdigest()
    changes=read(SELECTIVE/'description_changes.csv')
    assert {r['intent_id'] for r in changes}==set(SOURCE_SEGMENTS)
    segment_by_key={(r['place_id'],r['language'],int(r['segment_index'])):r for r in read(FULL/'translation_segments.csv')}
    selected_segments={};source_log=[]
    for c in changes:
        key=(c['place_id'],c['language']);ids=SOURCE_SEGMENTS[c['intent_id']]
        for i in ids:
            rec=segment_by_key[key+(i,)]
            assert rec['translation'].strip() and rec['output_capped']=='False'
            selected_segments.setdefault(key,{})[i]=rec
            source_log.append(dict(intent_id=c['intent_id'],place_id=key[0],language=key[1],segment_index=i,source_text=rec['source_text'],translation=rec['translation']))
    assert len(selected_segments)==68
    write(OUT/'selective_mt_source_segments.csv',source_log)
    matched_mt_additions={key:' '.join(group[i]['translation'] for i in sorted(group)) for key,group in selected_segments.items()}
    mt_selective=[]
    for s in raw:
        row=dict(s);add=matched_mt_additions.get((s['place_id'],s['language']))
        if add:row['description']+=' '+add
        mt_selective.append(row)
    assert sum(a!=b for a,b in zip(raw,mt_selective))==68
    write(ROOT/'data/processed/tourapi_standardized_selective_mt_68_experiment.csv',mt_selective)
    model=load_baseline_model(meta)
    def tok(t):return len(model.tokenizer(t,add_special_tokens=False,truncation=False)['input_ids'])
    translated_text=[clean(' '.join([docs[i]['title'],docs[i]['addr1'],docs[i]['addr2'],mt[i]['description']])) for i in translated]
    token_lengths=[tok(t)+2 for t in translated_text]
    n_truncated=sum(x>1024 for x in token_lengths)
    # Report rather than silently truncate. The original corpus had no truncation.
    assert n_truncated==0, f'{n_truncated} translated search documents exceed 1024 tokens; rerun all conditions with a larger common limit'
    mt_vec=base_vec.copy();mt_vec[translated]=encode(model,translated_text,4,'full translation documents',np)
    scope_vec=base_vec.copy();scope_vec[selected]=mt_vec[selected]
    mt_select_text=[docs[i]['search_text']+' '+matched_mt_additions[docs[i]['place_id'],docs[i]['language']] for i in selected]
    assert max(tok(s)+2 for s in mt_select_text)<=1024
    select_mt_vec=base_vec.copy();select_mt_vec[selected]=encode(model,mt_select_text,4,'same-engine selective documents',np)
    np.save(OUT/'document_embeddings_full_mt_200.npy',mt_vec)
    np.save(OUT/'document_embeddings_full_mt_68.npy',scope_vec)
    np.save(OUT/'document_embeddings_selective_mt_68.npy',select_mt_vec)
    vectors={'BASELINE':base_vec,'SELECTIVE':select_vec,'SELECTIVE_MT_68':select_mt_vec,'FULL_MT_68':scope_vec,'FULL_MT_200':mt_vec}
    result={m:evaluate(docs,queries,v,query_vec,np) for m,v in vectors.items()}
    for m,rows in result.items(): assert len(rows)==2400
    original=read(NAT/'query_results_original.csv');first=result['BASELINE']
    assert all(a['query_id']==b['query_id'] and a['target_language']==b['target_language'] and str(a['rank'])==b['rank'] for a,b in zip(first,original))
    prior=read(SELECTIVE/'rank_comparison.csv');assert all(a['query_id']==b['query_id'] and a['target_language']==b['target_language'] and str(a['rank'])==b['rank_after'] for a,b in zip(result['SELECTIVE'],prior))
    enriched_keys={(r['intent_id'],r['language']) for r in read(SELECTIVE/'description_changes.csv')};assert len(enriched_keys)==73
    title_ko={d['place_id']:d['title'] for d in raw if d['language']=='ko'}
    rows=[]
    for i,a in enumerate(first):
        intent=a['query_id'].rsplit('-',1)[0];target=a['target_language']
        row=dict(intent_id=intent,query_id=a['query_id'],query_type=a['query_type'],query_language=a['query_language'],target_language=target,place_id=a['place_id'],title_ko=title_ko[a['place_id']],query=a['query'],gap_target='YES' if (intent,target) in enriched_keys else 'NO')
        for m in METHODS:
            b=result[m][i];assert (a['query_id'],target)==(b['query_id'],b['target_language'])
            row['rank_'+m.lower()]=b['rank'];row['top1_'+m.lower()]=b['top1_title'];row['score_'+m.lower()]=b['correct_score']
        rows.append(row)
    write(OUT/'query_rank_comparison.csv',rows)
    summary=[]
    groups={}
    for kind in ('ALL','REPRESENTATIVE','DETAIL'):
        for scope,pred in (
            ('MIXED',lambda r:r['target_language']=='ALL_PLACES'),
            ('SAME_LANGUAGE',lambda r:r['target_language']==r['query_language']),
            ('CROSS_LANGUAGE',lambda r:r['target_language'] in LANGS and r['target_language']!=r['query_language']),
            ('FOREIGN_SINGLE',lambda r:r['target_language'] in ('en','ja')),
            ('GAP_TARGETS',lambda r:r['gap_target']=='YES'),
            ('GAP_TARGETS_CROSS',lambda r:r['gap_target']=='YES' and r['target_language']!=r['query_language']),
        ):
            indices=[i for i,r in enumerate(rows) if (kind=='ALL' or r['query_type']==kind) and pred(r)]
            if not indices:continue
            groups[(kind,scope)]=indices
            for m in METHODS:
                summary.append(dict(query_type=kind,scope=scope,method=m,**metrics([result[m][i] for i in indices])))
    write(OUT/'method_metrics.csv',summary)
    # Wide compact comparison: one row per evaluation setting.
    wide=[]
    for (kind,scope),indices in groups.items():
        rec=dict(query_type=kind,scope=scope,evaluations=len(indices))
        for m in METHODS:
            s=metrics([result[m][i] for i in indices]);
            for k in ('recall_at_1','recall_at_5','recall_at_10','mrr'):rec[m.lower()+'_'+k]=s[k]
        wide.append(rec)
    write(OUT/'method_comparison_summary.csv',wide)
    lm=[]
    for m in METHODS:
        for r in metrics_by_group(result[m]):lm.append(dict(method=m,**r))
    write(OUT/'language_metrics.csv',lm)
    workload=[]
    translated_set=set(translated)
    for m,idxs in [('BASELINE',[]),('SELECTIVE',selected),('SELECTIVE_MT_68',selected),('FULL_MT_68',selected),('FULL_MT_200',translated)]:
        outputs=[aug[i]['description'][len(raw[i]['description']):].strip() if m=='SELECTIVE' else mt_selective[i]['description'][len(raw[i]['description']):].strip() if m=='SELECTIVE_MT_68' else mt[i]['description'] for i in idxs]
        n=sum(tok(t) for t in outputs)
        workload.append(dict(method=m,foreign_documents_changed=len(idxs),foreign_documents_total=200,changed_places=len({docs[i]['place_id'] for i in idxs}),newly_translated_output_chars=sum(len(t) for t in outputs),newly_translated_output_bge_tokens=n,workload_note='Output text volume is a proxy, not monetary or human-time cost. SELECTIVE uses AI-authored translation; MT variants use the same NLLB engine.' if idxs else 'Original texts reused'))
    write(OUT/'translation_workload.csv',workload)
    # Paired exploratory comparisons: direct method comparisons in matched target
    # scope as well as whole corpus. A positive effect favors SELECTIVE.
    rng=np.random.default_rng(20261120);tests=[]
    for method,compare_to in [('SELECTIVE','FULL_MT_68'),('SELECTIVE','FULL_MT_200'),('SELECTIVE_MT_68','FULL_MT_68')]:
        for scope in ('GAP_TARGETS','MIXED','CROSS_LANGUAGE'):
            inds=groups['ALL',scope]
            paired=[dict(place_id=rows[i]['place_id'],rank_before=rows[i]['rank_'+compare_to.lower()],rank_after=rows[i]['rank_'+method.lower()]) for i in inds]
            tests.append(paired_test(method+'_MINUS_'+compare_to+'_'+scope,paired,'R@1',rng))
    adjusted=holm([x['raw_p_value'] for x in tests])
    for x,p in zip(tests,adjusted):x['holm_adjusted_p_value']=round(p,8);x['raw_p_value']=round(x['raw_p_value'],8)
    write(OUT/'paired_method_tests.csv',tests)
    (OUT/'metadata.json').write_text(json.dumps(dict(model=MODEL,source_documents=300,queries=600,original_corpus_sha256=meta['corpus_sha256'],translation_model_metadata=json.loads((FULL/'translation_metadata.json').read_text()),same_scope_full_mt_documents=68,full_mt_documents=200,selective_documents=68,selective_mt_documents=68,foreign_doc_truncation_count=n_truncated,max_translated_document_tokens=max(token_lengths),limitations=['Full MT uses sentence-level NLLB with greedy decoding and recovery. SELECTIVE uses AI-authored source-grounded text; SELECTIVE_MT_68 instead appends the same NLLB translations of selected Korean segments, controlling translation engine and scope.','NLLB generated errors in historical names and architectural terminology; this is a specific research MT baseline, not high-quality validated full translation.','Same 600 Korean-reference-grounded AI queries inform feature audit and selective method; no independent holdout.','Single designated target per query; others may be relevant.','Output token volume is not a direct cost/time estimate.','Paired re-sampling tests are exploratory and do not establish equivalence or noninferiority.']),ensure_ascii=False,indent=2),encoding='utf-8')
    for r in wide:
        if r['query_type']=='ALL':print(r,flush=True)
    print('workload',workload,flush=True);print(OUT,flush=True)

if __name__=='__main__':main()
