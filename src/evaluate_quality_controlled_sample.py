"""Scope- and translator-matched comparison on 20 randomly selected gap docs."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from run_bge_m3 import DATA,MODEL,ROOT,encode,evaluate,load_baseline_model
from compare_enrichment_methods import SOURCE_SEGMENTS,clean
from statistical_validate_enrichment import paired_test,holm

OUT=ROOT/'results/tourapi_quality_controlled_sample'
BASE=ROOT/'results/bge_m3_dense_full'
NAT=ROOT/'results/tourapi_natural_query_eval_600'
MT=ROOT/'results/tourapi_enrichment_method_comparison'
RAW=ROOT/'data/processed/tourapi_standardized.csv'
METHODS=('BASELINE','SOURCE_CHECKED_SELECTIVE_20','SOURCE_CHECKED_FULL_20','NLLB_SELECTIVE_20','NLLB_FULL_20')
LANGS=('ko','en','ja')

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(rows):
    ranks=[int(r['rank']) for r in rows];n=len(rows)
    out=dict(evaluations=n,distinct_places=len({r['place_id'] for r in rows}))
    for k in (1,5,10):
        hits=sum(r<=k for r in ranks);out[f'hits_at_{k}']=hits;out[f'recall_at_{k}']=round(hits/n,6)
    out['mrr']=round(statistics.mean(1/r for r in ranks),6)
    return out

def main():
    selected=read(OUT/'sample_documents.csv');seg=read(OUT/'source_checked_translation_segments.csv');docs=read(DATA);raw=read(RAW);queries=read(NAT/'queries.csv')
    sample_keys={(x['place_id'],x['language']) for x in selected};assert len(sample_keys)==20 and len(seg)==147
    assert len(docs)==len(raw)==300 and len(queries)==600
    assert all((d['place_id'],d['language'],d['overview'])==(a['place_id'],a['language'],a['description']) for d,a in zip(docs,raw))
    groups=defaultdict(dict)
    for r in seg: groups[r['place_id'],r['language']][int(r['segment_index'])]=r
    assert set(groups)==sample_keys
    # Reuse exactly the same translated sentences in the selected and full arms.
    add_text={};full_text={};log=[];gap_keys=set()
    for d in selected:
        key=d['place_id'],d['language'];ids=d['intent_ids'].split('|')
        relevant={n for iid in ids for n in SOURCE_SEGMENTS[iid]}
        assert relevant<=set(groups[key])
        add_text[key]=' '.join(groups[key][n]['translation'] for n in sorted(relevant))
        full_text[key]=' '.join(groups[key][n]['translation'] for n in sorted(groups[key]))
        assert add_text[key] and full_text[key]
        for iid in ids:gap_keys.add((iid,key[1]))
        for n,r in sorted(groups[key].items()):
            log.append(dict(**r,selected_for_append='YES' if n in relevant else 'NO'))
    write(OUT/'translation_selection.csv',log)
    indices=[i for i,d in enumerate(raw) if (d['place_id'],d['language']) in sample_keys];assert len(indices)==20
    qs=[];qf=[]
    for r in raw:
        s,f=dict(r),dict(r);key=r['place_id'],r['language']
        if key in sample_keys:s['description']+=' '+add_text[key];f['description']=full_text[key]
        qs.append(s);qf.append(f)
    assert sum(a!=b for a,b in zip(raw,qs))==sum(a!=b for a,b in zip(raw,qf))==20
    write(OUT/'tourapi_source_checked_selective_20.csv',qs);write(OUT/'tourapi_source_checked_full_20.csv',qf)
    model=load_baseline_model(json.loads((BASE/'run_metadata.json').read_text()))
    def tokens(t):return len(model.tokenizer(t,add_special_tokens=False,truncation=False)['input_ids'])
    texts_s=[docs[i]['search_text']+' '+add_text[docs[i]['place_id'],docs[i]['language']] for i in indices]
    texts_f=[clean(' '.join([docs[i]['title'],docs[i]['addr1'],docs[i]['addr2'],full_text[docs[i]['place_id'],docs[i]['language']]])) for i in indices]
    assert max(tokens(t)+2 for t in texts_s+texts_f)<=1024
    baseline=np.load(BASE/'document_embeddings.npy');qvec=np.load(NAT/'query_embeddings.npy')
    nllbselect=np.load(MT/'document_embeddings_selective_mt_68.npy');nllbfull=np.load(MT/'document_embeddings_full_mt_68.npy')
    assert baseline.shape==nllbselect.shape==nllbfull.shape==(300,1024) and qvec.shape==(600,1024)
    def substitute(v):
        o=baseline.copy();o[indices]=v;return o
    source_select=substitute(encode(model,texts_s,4,'source-checked selective 20',np))
    source_full=substitute(encode(model,texts_f,4,'source-checked full 20',np))
    vectors=dict(BASELINE=baseline,SOURCE_CHECKED_SELECTIVE_20=source_select,SOURCE_CHECKED_FULL_20=source_full,NLLB_SELECTIVE_20=substitute(nllbselect[indices]),NLLB_FULL_20=substitute(nllbfull[indices]))
    np.save(OUT/'document_embeddings_source_checked_selective_20.npy',source_select);np.save(OUT/'document_embeddings_source_checked_full_20.npy',source_full)
    result={m:evaluate(docs,queries,v,qvec,np) for m,v in vectors.items()}
    assert all(len(r)==2400 for r in result.values())
    titles={d['place_id']:d['title'] for d in docs if d['language']=='ko'}
    rows=[];target_ids=set(x['place_id'] for x in selected)
    for i,r in enumerate(result['BASELINE']):
        iid=r['query_id'].rsplit('-',1)[0];target=r['target_language']
        row=dict(intent_id=iid,query_id=r['query_id'],query_type=r['query_type'],query_language=r['query_language'],target_language=target,place_id=r['place_id'],title_ko=titles[r['place_id']],query=r['query'],gap_target='YES' if (iid,target) in gap_keys else 'NO',sample_document='YES' if (r['place_id'],target) in sample_keys else 'NO')
        for m in METHODS:row['rank_'+m.lower()]=result[m][i]['rank']
        rows.append(row)
    write(OUT/'rank_comparison.csv',rows)
    sets={
        'ALL_MIXED':[i for i,r in enumerate(rows) if r['target_language']=='ALL_PLACES'],
        'ALL_CROSS_LANGUAGE':[i for i,r in enumerate(rows) if r['target_language'] in LANGS and r['target_language']!=r['query_language']],
        'SAMPLED_DOCUMENT_TARGETS':[i for i,r in enumerate(rows) if r['sample_document']=='YES'],
        'SAMPLE_GAP_TARGETS':[i for i,r in enumerate(rows) if r['gap_target']=='YES'],
        'SAMPLE_GAP_CROSS':[i for i,r in enumerate(rows) if r['gap_target']=='YES' and r['target_language']!=r['query_language']],
    }
    assert len(sets['SAMPLED_DOCUMENT_TARGETS'])==120 and len(sets['SAMPLE_GAP_TARGETS'])==3*len(gap_keys)
    out=[]
    for scope,inds in sets.items():
        for m in METHODS:out.append(dict(scope=scope,method=m,**summarize([result[m][i] for i in inds])))
    write(OUT/'method_performance.csv',out)
    work=[]
    nllb_t=read(MT/'selective_mt_source_segments.csv')
    nllb_segments=read(ROOT/'results/tourapi_full_translation/translation_segments.csv')
    nt=defaultdict(dict);nf=defaultdict(dict)
    for r in nllb_t:
        k=r['place_id'],r['language']
        if k in sample_keys:nt[k][int(r['segment_index'])]=r['translation']
    for r in nllb_segments:
        k=r['place_id'],r['language']
        if k in sample_keys:nf[k][int(r['segment_index'])]=r['translation']
    for method,bydoc,nseg in [('SOURCE_CHECKED_SELECTIVE_20',add_text,sum(r['selected_for_append']=='YES' for r in log)),('SOURCE_CHECKED_FULL_20',full_text,len(log)),('NLLB_SELECTIVE_20',{k:' '.join(v[n] for n in sorted(v)) for k,v in nt.items()},sum(len(v) for v in nt.values())),('NLLB_FULL_20',{k:' '.join(v[n] for n in sorted(v)) for k,v in nf.items()},sum(len(v) for v in nf.values()))]:
        assert set(bydoc)==sample_keys
        work.append(dict(method=method,changed_documents=20,distinct_places=len(target_ids),translated_segments=nseg,new_output_chars=sum(len(t) for t in bydoc.values()),new_output_bge_tokens=sum(tokens(t) for t in bydoc.values())))
    write(OUT/'translation_workload.csv',work)
    tests=[];rng=np.random.default_rng(20261120)
    for scope in ('SAMPLE_GAP_TARGETS','SAMPLE_GAP_CROSS','ALL_MIXED','ALL_CROSS_LANGUAGE'):
        inds=sets[scope]
        for measure in ('R@1','MRR'):
            paired=[dict(place_id=rows[i]['place_id'],rank_before=rows[i]['rank_source_checked_full_20'],rank_after=rows[i]['rank_source_checked_selective_20']) for i in inds]
            tests.append(paired_test('SOURCE_CHECKED_SELECTIVE_MINUS_FULL_'+scope,paired,measure,rng))
    for r,p in zip(tests,holm([r['raw_p_value'] for r in tests])):r['holm_adjusted_p_value']=round(p,8);r['raw_p_value']=round(r['raw_p_value'],8)
    write(OUT/'paired_method_tests.csv',tests)
    (OUT/'metadata.json').write_text(json.dumps(dict(sample_documents=20,documents_per_language=10,unique_places=len(target_ids),gap_intent_language_pairs=len(gap_keys),translated_segments=147,selected_translated_segments=sum(r['selected_for_append']=='YES' for r in log),model=MODEL,source_checked_translator='AI assistant; same translated source segments in both arms',limitation='Stratified random subset of previously identified gap docs, not representative of all 200 foreign docs. Translation source consistency checked by AI, not independent human quality assessment. Retrieval queries informed gap selection and were not independently held out. No equivalence/noninferiority margin prespecified.'),ensure_ascii=False,indent=2),encoding='utf-8')
    for r in out:
        if r['scope'] in ('SAMPLE_GAP_TARGETS','ALL_MIXED','ALL_CROSS_LANGUAGE'):print(r,flush=True)
    print('workload',work,flush=True)

if __name__=='__main__':main()
