"""Evaluate a severity-based hybrid enrichment policy against fixed strategies.

Full translation is selected when at least one audited intent is MISSING;
selective supplementation is used for PARTIAL-only documents. No ranking
outcome is used for the assignments. Reuse frozen same-translator embeddings.
"""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from run_bge_m3 import DATA,MODEL,ROOT,evaluate

OUT=ROOT/'results/tourapi_hybrid_policy'
BASE=ROOT/'results/bge_m3_dense_full'
NAT=ROOT/'results/tourapi_natural_query_eval_600'
MT=ROOT/'results/tourapi_enrichment_method_comparison'
QC=ROOT/'results/tourapi_quality_controlled_sample'
CHANGES=ROOT/'results/tourapi_full_feature_enrichment/description_changes.csv'
RAW=ROOT/'data/processed/tourapi_standardized.csv'
LANGS=('ko','en','ja')
RANDOM_TRIALS=300
SEED=20261120

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(ranks):
    a=np.asarray(ranks);n=len(a)
    return dict(evaluations=n,recall_at_1=round(float((a<=1).mean()),6),recall_at_5=round(float((a<=5).mean()),6),recall_at_10=round(float((a<=10).mean()),6),mrr=round(float((1/a).mean()),6))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    docs=read(DATA);raw=read(RAW);queries=read(NAT/'queries.csv');changes=read(CHANGES)
    db={(r['place_id'],r['language']):i for i,r in enumerate(docs)};assert len(db)==300
    assert len(docs)==len(raw)==300 and len(queries)==600
    base=np.load(BASE/'document_embeddings.npy');qvec=np.load(NAT/'query_embeddings.npy')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    def tok(text):return len(tokenizer(text,add_special_tokens=False,truncation=False)['input_ids'])
    cfg=[
      ('NLLB_68',MT/'document_embeddings_selective_mt_68.npy',MT/'document_embeddings_full_mt_68.npy',ROOT/'data/processed/tourapi_standardized_selective_mt_68_experiment.csv',ROOT/'data/processed/tourapi_standardized_full_translation_experiment.csv',None),
      ('SOURCE_CHECKED_20',QC/'document_embeddings_source_checked_selective_20.npy',QC/'document_embeddings_source_checked_full_20.npy',QC/'tourapi_source_checked_selective_20.csv',QC/'tourapi_source_checked_full_20.csv',QC/'sample_documents.csv'),
    ]
    place_ids=sorted({d['place_id'] for d in docs});place_ix={p:i for i,p in enumerate(place_ids)}
    place_doc_ids=np.array([[db[p,l] for l in LANGS] for p in place_ids])
    query_place=np.array([place_ix[q['place_id']] for q in queries]);query_lang=np.array([LANGS.index(q['query_language']) for q in queries])
    q_index={q['query_id']:i for i,q in enumerate(queries)}
    summary=[];decision=[];random_rows=[];random_summary=[];rank_rows=[];rng=np.random.default_rng(SEED)
    for setting,select_path,full_path,select_data,full_data,sample_file in cfg:
        pool={(r['place_id'],r['language']) for r in changes}
        if sample_file:pool&={(r['place_id'],r['language']) for r in read(sample_file)}
        full_keys={k for k in pool if any((r['place_id'],r['language'])==k and r['original_coverage']=='MISSING' for r in changes)}
        idx=np.array(sorted(db[k] for k in pool));full_idx=np.array(sorted(db[k] for k in full_keys))
        sv=np.load(select_path);fv=np.load(full_path)
        assert sv.shape==fv.shape==base.shape==(300,1024)
        # Apply only the scope selected by the setting. In both inputs the
        # corresponding foreign documents already contain the desired method.
        assert {i for i in range(300) if not np.array_equal(base[i],sv[i])}==set(idx)
        assert {i for i in range(300) if not np.array_equal(base[i],fv[i])}==set(idx)
        hybrid=sv.copy();hybrid[full_idx]=fv[full_idx]
        np.save(OUT/f'{setting.lower()}_hybrid_embeddings.npy',hybrid)
        sel_data=read(select_data);full_docs=read(full_data)
        assert len(sel_data)==len(full_docs)==300
        costs_s={};costs_f={}
        for i in idx:
            a,b,c=raw[i],sel_data[i],full_docs[i];assert a['description']==raw[i]['description'] and b['description'].startswith(a['description'])
            costs_s[i]=tok(b['description'][len(a['description']):].strip());costs_f[i]=tok(c['description'])
            key=(a['place_id'],a['language']);rs=[r for r in changes if (r['place_id'],r['language'])==key]
            decision.append(dict(setting=setting,place_id=key[0],language=key[1],title=a['title'],audited_statuses='|'.join(sorted({r['original_coverage'] for r in rs})),decision='FULL' if key in full_keys else 'SELECTIVE',selective_output_tokens=costs_s[i],full_output_tokens=costs_f[i],policy_output_tokens=costs_f[i] if key in full_keys else costs_s[i]))
        cost={ 'BASELINE':0,'SELECTIVE':sum(costs_s.values()),'FULL':sum(costs_f.values()),'HYBRID':sum(costs_f[i] if i in set(full_idx) else costs_s[i] for i in idx)}
        methods={'BASELINE':base,'SELECTIVE':sv,'FULL':fv,'HYBRID':hybrid}
        gap_keys={(r['intent_id'],r['language']) for r in changes if (r['place_id'],r['language']) in pool}
        results={m:evaluate(docs,queries,v,qvec,np) for m,v in methods.items()}
        ref=results['BASELINE'];assert len(ref)==2400
        groups={}
        for scope,predicate in (
            ('MIXED',lambda r:r['target_language']=='ALL_PLACES'),
            ('CROSS_LANGUAGE',lambda r:r['target_language'] in LANGS and r['target_language']!=r['query_language']),
            ('GAP_TARGETS',lambda r:(r['query_id'].rsplit('-',1)[0],r['target_language']) in gap_keys),
            ('GAP_TARGETS_CROSS',lambda r:(r['query_id'].rsplit('-',1)[0],r['target_language']) in gap_keys and r['target_language']!=r['query_language']),
        ):
            inds=[i for i,r in enumerate(ref) if predicate(r)];assert inds
            groups[scope]=inds
            for m in methods:
                summary.append(dict(setting=setting,scope=scope,method=m,full_docs=len(pool) if m=='FULL' else len(full_keys) if m=='HYBRID' else 0,selective_docs=len(pool) if m=='SELECTIVE' else len(pool)-len(full_keys) if m=='HYBRID' else 0,translated_output_tokens=cost[m],**stats([results[m][i]['rank'] for i in inds])))
        for i,a in enumerate(ref):
            row=dict(setting=setting,query_id=a['query_id'],place_id=a['place_id'],query_type=a['query_type'],query_language=a['query_language'],target_language=a['target_language'],query=a['query'],gap_target='YES' if (a['query_id'].rsplit('-',1)[0],a['target_language']) in gap_keys else 'NO')
            for m in methods:row['rank_'+m.lower()]=results[m][i]['rank']
            rank_rows.append(row)
        # Random assignments use the same full-document count. This is a
        # comparison to random allocation, not a post-hoc optimized policy.
        scores_s=qvec@sv.T;scores_f=qvec@fv.T
        gi=groups['GAP_TARGETS'];gq=np.array([q_index[ref[i]['query_id']] for i in gi]);gl=np.array([LANGS.index(ref[i]['target_language']) for i in gi]);gp=query_place[gq]
        target_doc=place_doc_ids[gp,gl]
        g_candidates=place_doc_ids[:,gl].T
        def score_ranks(score):
            # Stable place_id-order tie break matches evaluate().
            vals=score[gq[:,None],g_candidates];correct=score[gq,target_doc][:,None]
            ranks=1+np.sum((vals>correct)|((vals==correct)&(np.arange(100)[None,:]<gp[:,None])),axis=1)
            pooled=score[:,place_doc_ids].max(axis=2);c=pooled[np.arange(len(queries)),query_place][:,None]
            mixed_rank=1+np.sum((pooled>c)|((pooled==c)&(np.arange(100)[None,:]<query_place[:,None])),axis=1)
            return ranks,mixed_rank
        # Cross-check vectorized ranks against canonical evaluator on the hybrid.
        candidate=scores_s.copy();candidate[:,full_idx]=scores_f[:,full_idx]
        actual_g,actual_m=score_ranks(candidate)
        assert np.array_equal(actual_g,[results['HYBRID'][i]['rank'] for i in gi])
        assert np.array_equal(actual_m,[results['HYBRID'][i]['rank'] for i in groups['MIXED']])
        trials=[]
        for trial in range(RANDOM_TRIALS):
            choose=rng.choice(idx,size=len(full_idx),replace=False)
            score=scores_s.copy();score[:,choose]=scores_f[:,choose]
            gr,mr=score_ranks(score);budget=cost['SELECTIVE']+sum(costs_f[i]-costs_s[i] for i in choose)
            actual=dict(setting=setting,trial=trial+1,full_doc_count=len(choose),translated_output_tokens=budget,gap_target_r1=float((gr==1).mean()),gap_target_r5=float((gr<=5).mean()),gap_target_mrr=float((1/gr).mean()),mixed_r1=float((mr==1).mean()))
            trials.append(actual);random_rows.append(actual)
        # Distinguish equal count from approximately equal token budget.
        close=[r for r in trials if abs(r['translated_output_tokens']-cost['HYBRID'])<=0.10*cost['HYBRID']]
        random_summary.append(dict(setting=setting,hybrid_full_docs=len(full_idx),hybrid_output_tokens=cost['HYBRID'],hybrid_gap_r1=float((actual_g==1).mean()),hybrid_mixed_r1=float((actual_m==1).mean()),random_trials=len(trials),random_mean_gap_r1=statistics.mean(r['gap_target_r1'] for r in trials),random_median_gap_r1=statistics.median(r['gap_target_r1'] for r in trials),random_mean_mixed_r1=statistics.mean(r['mixed_r1'] for r in trials),random_gap_at_or_below_hybrid=sum(r['gap_target_r1']<=float((actual_g==1).mean()) for r in trials)/len(trials),budget_within_10pct_trials=len(close),budget_matched_mean_gap_r1=statistics.mean(r['gap_target_r1'] for r in close) if close else '',budget_matched_mean_mixed_r1=statistics.mean(r['mixed_r1'] for r in close) if close else ''))
    write(OUT/'policy_performance.csv',summary);write(OUT/'policy_document_decisions.csv',decision);write(OUT/'rank_comparison.csv',rank_rows);write(OUT/'random_assignment_trials.csv',random_rows);write(OUT/'random_allocation_comparison.csv',random_summary)
    (OUT/'metadata.json').write_text(json.dumps(dict(rule='At least one MISSING intent -> full translation; PARTIAL-only -> selective',comparison_scopes=['NLLB_68','SOURCE_CHECKED_20'],random_trials_per_scope=RANDOM_TRIALS,seed=SEED,limitations=['Post-hoc exploratory rule based on feature coverage, not ranking outcomes.','Random allocations have same number of full translations; near-budget subset uses +/-10% output token volume.','Output tokens are a workload proxy, not actual cost.','AI-generated queries and gap-selected document pool; no independent confirmation.']),ensure_ascii=False,indent=2),encoding='utf-8')
    for x in summary:
        if x['scope'] in ('GAP_TARGETS','MIXED','CROSS_LANGUAGE'):print(x,flush=True)
    print('random allocation',random_summary,flush=True)

if __name__=='__main__':main()
