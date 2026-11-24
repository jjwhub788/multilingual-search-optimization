"""Compare fixed pre-intervention feature coverage groups before/after enrichment."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ASSOC=ROOT/'results/tourapi_coverage_retrieval_analysis/coverage_rank_joined.csv'
OUT=ROOT/'results/tourapi_full_feature_enrichment'

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    original=[x for x in read(ASSOC) if x['eligible_primary']=='YES']
    ranks={(x['query_id'],x['target_language']):x for x in read(OUT/'rank_comparison.csv')}
    assert len(original)==1185 and len(ranks)==2400
    results=[]
    for kind in ('ALL','REPRESENTATIVE','DETAIL'):
        for relation in ('ALL','SAME','CROSS'):
            subset=[r for r in original if (kind=='ALL' or r['query_type']==kind) and (relation=='ALL' or r['language_relation']==relation)]
            rec=dict(query_type=kind,language_relation=relation)
            for label in ('PRESENT','GAP'):
                group=[r for r in subset if r['coverage_group']==label]
                n=len(group);rec[label.lower()+'_evaluations']=n
                for stage in ('before','after'):
                    hits=sum(ranks[x['query_id'],x['target_language']]['rank_'+stage]=='1' for x in group)
                    rec[label.lower()+'_top1_'+stage]=hits/n
            for stage in ('before','after'):
                rec['gap_pp_'+stage]=round(100*(rec['present_top1_'+stage]-rec['gap_top1_'+stage]),4)
            rec['gap_reduction_pp']=round(rec['gap_pp_before']-rec['gap_pp_after'],4)
            results.append(rec)
    with (OUT/'coverage_gap_reduction.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    print(results[0])

if __name__=='__main__':main()
