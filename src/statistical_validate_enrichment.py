"""Exploratory place-cluster bootstrap and paired sign-flip tests.

Repeated queries/directions for one tourist place are kept together.
Intervals and p-values describe a resampling model for this curated benchmark,
not independently sampled real-world search traffic or causal confirmation.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'results/tourapi_full_feature_enrichment'
ASSOC=ROOT/'results/tourapi_coverage_retrieval_analysis/coverage_rank_joined.csv'
OUT=ROOT/'results/tourapi_statistical_validation'
B=20000
PERM=100000
SEED=20261120

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def score(rank,measure):
    rank=int(rank)
    if measure=='R@1':return float(rank==1)
    if measure=='R@5':return float(rank<=5)
    if measure=='MRR':return 1/rank
    raise ValueError(measure)

def p_sign_flip(contributions,rng):
    """Two-sided paired cluster randomization under symmetric change null."""
    arr=np.asarray(contributions,dtype=float)
    arr=arr[np.abs(arr)>1e-14]
    obs=abs(arr.sum())
    if len(arr)==0:return 1.0,'EXACT',1,0
    if len(arr)<=18:
        total=1<<len(arr); extreme=0
        for start in range(0,total,4096):
            idx=np.arange(start,min(total,start+4096),dtype=np.uint32)
            signs=1-2*((idx[:,None]>>np.arange(len(arr),dtype=np.uint32))&1).astype(float)
            vals=signs@arr
            extreme+=int(np.sum(np.abs(vals)>=obs-1e-12))
        return extreme/total,'EXACT',total,len(arr)
    extreme=0
    for start in range(0,PERM,5000):
        n=min(5000,PERM-start)
        signs=2*rng.integers(0,2,size=(n,len(arr)),dtype=np.int8)-1
        extreme+=int(np.sum(np.abs(signs@arr)>=obs-1e-12))
    return (extreme+1)/(PERM+1),'MONTE_CARLO',PERM,len(arr)

def paired_test(name,rows,measure,rng,primary=False):
    # Columns are per-place number of evaluations, before sum, after sum.
    clusters=defaultdict(lambda:[0,0.,0.])
    improved=worse=unchanged=0
    for r in rows:
        before=score(r['rank_before'],measure);after=score(r['rank_after'],measure)
        rec=clusters[r['place_id']];rec[0]+=1;rec[1]+=before;rec[2]+=after
        if after>before+1e-12:improved+=1
        elif after<before-1e-12:worse+=1
        else:unchanged+=1
    vals=np.array([clusters[p] for p in sorted(clusters)],dtype=float)
    n=len(vals); totals=vals.sum(axis=0); before=totals[1]/totals[0];after=totals[2]/totals[0]
    diff=after-before
    boot=rng.integers(0,n,size=(B,n)); bs=vals[boot].sum(axis=1)
    delta=(bs[:,2]-bs[:,1])/bs[:,0]
    lo,hi=np.quantile(delta,[0.025,0.975])
    p,method,draws,active=p_sign_flip((vals[:,2]-vals[:,1])/totals[0],rng)
    factor=100 if measure.startswith('R@') else 1
    return dict(test=name,measure=measure,role='PRIMARY_DESCRIPTIVE' if primary else 'EXPLORATORY_SECONDARY',evaluation_count=len(rows),distinct_places=n,active_change_clusters=active,before=round(before,8),after=round(after,8),effect=round(diff*factor,6),effect_unit='percentage_points' if factor==100 else 'MRR',ci95_lower=round(float(lo)*factor,6),ci95_upper=round(float(hi)*factor,6),improved_evaluations=improved,worse_evaluations=worse,unchanged_evaluations=unchanged,raw_p_value=p,permutation_method=method,permutation_draws=draws)

def gap_test(name,rows,rank_index,rng):
    # Per place: n_present, delta hits present, before hits present,
    # n_gap, delta hits gap, before hits gap.
    clusters=defaultdict(lambda:np.zeros(6,dtype=float))
    for a in rows:
        r=rank_index[a['query_id'],a['target_language']]
        assert a['rank_before']==r['rank_before']
        before=score(r['rank_before'],'R@1');after=score(r['rank_after'],'R@1')
        offset=0 if a['coverage_group']=='PRESENT' else 3
        v=clusters[a['place_id']];v[offset]+=1;v[offset+1]+=after-before;v[offset+2]+=before
    v=np.array([clusters[x] for x in sorted(clusters)]);n=len(v);tot=v.sum(axis=0)
    present_before=tot[2]/tot[0];gap_before=tot[5]/tot[3]
    present_after=(tot[2]+tot[1])/tot[0];gap_after=(tot[5]+tot[4])/tot[3]
    gap0=present_before-gap_before;gap1=present_after-gap_after;effect=gap0-gap1
    boot=rng.integers(0,n,size=(B,n));bs=v[boot].sum(axis=1);valid=bs[(bs[:,0]>0)&(bs[:,3]>0)]
    effect_boot=valid[:,4]/valid[:,3]-valid[:,1]/valid[:,0]
    lo,hi=np.quantile(effect_boot,[0.025,0.975]);p,method,draws,active=p_sign_flip(v[:,4]/tot[3]-v[:,1]/tot[0],rng)
    return dict(test=name,measure='R@1_GAP_REDUCTION',role='EXPLORATORY_SECONDARY',evaluation_count=len(rows),distinct_places=n,active_change_clusters=active,before=round(gap0,8),after=round(gap1,8),effect=round(effect*100,6),effect_unit='percentage_points_reduction',ci95_lower=round(float(lo)*100,6),ci95_upper=round(float(hi)*100,6),improved_evaluations='',worse_evaluations='',unchanged_evaluations='',raw_p_value=p,permutation_method=method,permutation_draws=draws)

def holm(p_values):
    m=len(p_values);idx=sorted(range(m),key=lambda i:p_values[i]);out=[0.]*m;prev=0.
    for rank,i in enumerate(idx):
        prev=max(prev,min(1.,(m-rank)*p_values[i]));out[i]=prev
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ranks=read(EXP/'rank_comparison.csv');assoc=[r for r in read(ASSOC) if r['eligible_primary']=='YES']
    assert len(ranks)==2400 and len(assoc)==1185 and len({r['place_id'] for r in ranks})==100
    idx={(r['query_id'],r['target_language']):r for r in ranks};assert len(idx)==2400
    targeted=[r for r in ranks if r['gap_target']=='YES'];assert len(targeted)==219 and len({r['place_id'] for r in targeted})==45
    cross=[r for r in ranks if r['target_language'] in ('ko','en','ja') and r['target_language']!=r['query_language']]
    mixed=[r for r in ranks if r['target_language']=='ALL_PLACES']
    same=[r for r in ranks if r['target_language']==r['query_language']]
    assert len(cross)==1200 and len(mixed)==600 and len(same)==600
    rng=np.random.default_rng(SEED)
    specs=[
      ('GAP_TARGETS_R1',targeted,'R@1',True),
      ('GAP_TARGETS_R5',targeted,'R@5',False),
      ('GAP_TARGETS_MRR',targeted,'MRR',False),
      ('GAP_TARGETS_CROSS_R1',[r for r in targeted if r['target_language']!=r['query_language']],'R@1',False),
      ('DETAIL_GAP_TARGETS_R1',[r for r in targeted if r['query_type']=='DETAIL'],'R@1',False),
      ('REPRESENTATIVE_GAP_TARGETS_R1',[r for r in targeted if r['query_type']=='REPRESENTATIVE'],'R@1',False),
      ('ALL_MIXED_R1',mixed,'R@1',False),
      ('ALL_CROSS_LANGUAGE_R1',cross,'R@1',False),
      ('ALL_SAME_LANGUAGE_R1',same,'R@1',False),
    ]
    tests=[paired_test(name,rows,measure,rng,primary) for name,rows,measure,primary in specs]
    tests.append(gap_test('FEATURE_COVERAGE_GAP_REDUCTION',assoc,idx,rng))
    tests.append(gap_test('DETAIL_FEATURE_COVERAGE_GAP_REDUCTION',[r for r in assoc if r['query_type']=='DETAIL'],idx,rng))
    adjusted=holm([x['raw_p_value'] for x in tests])
    for row,padj in zip(tests,adjusted):
        row['holm_adjusted_p_value']=round(padj,8)
        row['raw_p_value']=round(row['raw_p_value'],8)
        row['raw_p_lt_0_05']='YES' if row['raw_p_value']<0.05 else 'NO'
        row['holm_p_lt_0_05']='YES' if padj<0.05 else 'NO'
        row['ci_excludes_zero']='YES' if row['ci95_lower']>0 or row['ci95_upper']<0 else 'NO'
    write(OUT/'statistical_tests.csv',tests)
    (OUT/'methods.json').write_text(json.dumps(dict(seed=SEED,bootstrap_replicates=B,confidence_interval='95% percentile cluster bootstrap, sample place_id as units with replacement; weighted evaluation proportions within resample',test='two-sided paired cluster sign-flip under symmetric/exchangeable before-after change null; exact enumeration for <=18 active clusters, otherwise 100,000 simulations with plus-one correction',effect='percent-point difference after minus before, except feature coverage gap reduction = prior-present vs prior-gap difference before minus after; MRR in absolute units',holm_family_size=len(tests),alpha=0.05,limitations=['Exploratory post-hoc analysis, not preregistered; label PRIMARY_DESCRIPTIVE refers to focal endpoint, not a prespecified confirmatory endpoint.','AI-generated queries derived from Korean descriptions, and enrichment selected using those intents. No independent held-out human evaluation.','Cluster by place accounts for repeated query outcomes within a place, not correlations across places due to shared ranking corpus.','Places are purposively selected; resampling intervals do not establish population-wide generalizability or causality.','Raw and Holm-adjusted p-values are both reported; intervals are unadjusted descriptive 95% intervals.']),ensure_ascii=False,indent=2),encoding='utf-8')
    for r in tests: print(r['test'],'n',r['evaluation_count'],'places',r['distinct_places'],'effect',r['effect'],'CI',r['ci95_lower'],r['ci95_upper'],'p',r['raw_p_value'],'Holm',r['holm_adjusted_p_value'])

if __name__=='__main__':main()
