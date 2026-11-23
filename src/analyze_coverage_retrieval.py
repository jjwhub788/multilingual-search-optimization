"""Join audited core-feature coverage to original BGE-M3 retrieval ranks.

Observational analysis. The same 600 queries are repeated across target
languages; counts below are evaluations, not independent query samples.
"""
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'results/tourapi_feature_coverage/feature_coverage_audit.csv'
RANKS=ROOT/'results/tourapi_natural_query_eval_600/rank_comparison.csv'
OUT=ROOT/'results/tourapi_coverage_retrieval_analysis'
LANGS=('ko','en','ja')

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def metrics(rows, rank_key='rank_before'):
    ranks=[int(r[rank_key]) for r in rows]
    assert ranks and max(ranks)<=100 and min(ranks)>=1
    n=len(ranks)
    out=dict(evaluations=n,distinct_places=len({r['place_id'] for r in rows}),distinct_intents=len({r['intent_id'] for r in rows}),distinct_queries=len({r['query_id'] for r in rows}))
    for k in (1,5,10):
        hits=sum(r<=k for r in ranks)
        out[f'hits_at_{k}']=hits;out[f'recall_at_{k}']=round(hits/n,6)
    out.update(mrr=round(statistics.mean(1/r for r in ranks),6),mean_rank=round(statistics.mean(ranks),3),median_rank=statistics.median(ranks))
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=read(AUDIT); ranks=read(RANKS)
    ix={(r['intent_id'],r['language']):r for r in audit}
    assert len(ix)==600 and len(ranks)==2400
    joined=[]
    for r in ranks:
        target=r['target_language']
        if target not in LANGS: continue  # mixed ranks have no unique target-language document
        intent=r['query_id'].rsplit('-',1)[0]
        a=ix[intent,target]; q=ix[intent,r['query_language']];ko=ix[intent,'ko']
        assert a['place_id']==q['place_id']==ko['place_id']==r['place_id']
        assert a['query_type']==q['query_type']==r['query_type'] and q['query']==r['query']
        status=a['description_coverage']
        # Primary contrast: target-language EN/JA, reference KO feature present,
        # and nonconflicting target status. Exclusions remain in joined CSV.
        reason=''
        if target=='ko':reason='KOREAN_REFERENCE'
        elif ko['description_coverage']!='PRESENT':reason='REFERENCE_FEATURE_NOT_FULLY_PRESENT'
        elif status not in ('PRESENT','PARTIAL','MISSING'):reason='UNRESOLVED_OR_CONFLICT'
        joined.append(dict(intent_id=intent,**r,language_relation='SAME' if target==r['query_language'] else 'CROSS',reference_ko_coverage=ko['description_coverage'],description_coverage=status,coverage_group='PRESENT' if status=='PRESENT' else 'GAP' if status in ('PARTIAL','MISSING') else 'EXCLUDED',coverage_method=a['decision_method'],core_features=a['core_features'],coverage_note=a['note'],description_evidence=a['description_evidence'],title_address_evidence=a['title_address_evidence'],eligible_primary='NO' if reason else 'YES',primary_exclusion_reason=reason))
    assert len(joined)==1800 and len({(r['intent_id'],r['query_language'],r['target_language']) for r in joined})==1800
    write(OUT/'coverage_rank_joined.csv',joined)
    primary=[r for r in joined if r['eligible_primary']=='YES']
    assert len(primary)==1185
    summaries=[];gaps=[]
    for kind in ('ALL','REPRESENTATIVE','DETAIL'):
        for target in ('FOREIGN_ALL','en','ja'):
            for relation in ('ALL','SAME','CROSS'):
                cohort=[r for r in primary if (kind=='ALL' or r['query_type']==kind) and (target=='FOREIGN_ALL' or r['target_language']==target) and (relation=='ALL' or r['language_relation']==relation)]
                for coverage in ('PRESENT','PARTIAL','MISSING','GAP'):
                    group=[r for r in cohort if r['coverage_group']=='GAP'] if coverage=='GAP' else [r for r in cohort if r['description_coverage']==coverage]
                    if group:summaries.append(dict(query_type=kind,target_language=target,language_relation=relation,coverage=coverage,**metrics(group)))
                present=[r for r in cohort if r['coverage_group']=='PRESENT'];gap=[r for r in cohort if r['coverage_group']=='GAP']
                if present and gap:
                    p=metrics(present);g=metrics(gap)
                    gaps.append(dict(query_type=kind,target_language=target,language_relation=relation,present_evaluations=p['evaluations'],gap_evaluations=g['evaluations'],present_places=p['distinct_places'],gap_places=g['distinct_places'],present_recall1=p['recall_at_1'],gap_recall1=g['recall_at_1'],recall1_difference_pp=round((p['recall_at_1']-g['recall_at_1'])*100,3),present_recall5=p['recall_at_5'],gap_recall5=g['recall_at_5'],recall5_difference_pp=round((p['recall_at_5']-g['recall_at_5'])*100,3),present_mrr=p['mrr'],gap_mrr=g['mrr'],mrr_difference=round(p['mrr']-g['mrr'],6)))
    write(OUT/'coverage_performance_summary.csv',summaries)
    write(OUT/'coverage_gap_comparison.csv',gaps)
    # Pair the two foreign documents on the same query to control query/intent
    # identity when exactly one target document has a gap. Language still varies.
    idx={(r['intent_id'],r['query_language'],r['target_language']):r for r in primary}
    pairs=[]
    for a in primary:
        if a['target_language']!='en':continue
        b=idx.get((a['intent_id'],a['query_language'],'ja'))
        if not b or a['coverage_group']==b['coverage_group']:continue
        present,gap=(a,b) if a['coverage_group']=='PRESENT' else (b,a)
        rp,rg=int(present['rank_before']),int(gap['rank_before'])
        pairs.append(dict(intent_id=a['intent_id'],place_id=a['place_id'],correct_title_ko=a['correct_title_ko'],query_id=a['query_id'],query_type=a['query_type'],query_language=a['query_language'],query=a['query'],present_language=present['target_language'],gap_language=gap['target_language'],gap_status=gap['description_coverage'],present_rank=rp,gap_rank=rg,gap_rank_minus_present=rg-rp,present_top1='YES' if rp==1 else 'NO',gap_top1='YES' if rg==1 else 'NO',which_rank_better='PRESENT' if rp<rg else 'GAP' if rg<rp else 'TIE',gap_note=gap['coverage_note']))
    assert pairs
    write(OUT/'paired_foreign_rank_comparison.csv',pairs)
    psummary=[]
    for kind in ('ALL','REPRESENTATIVE','DETAIL'):
        for lang in ('ALL','en','ja'):
            group=[r for r in pairs if (kind=='ALL' or r['query_type']==kind) and (lang=='ALL' or r['gap_language']==lang)]
            if not group:continue
            n=len(group);c=Counter(r['which_rank_better'] for r in group)
            psummary.append(dict(query_type=kind,gap_language=lang,paired_evaluations=n,distinct_intents=len({r['intent_id'] for r in group}),distinct_places=len({r['place_id'] for r in group}),present_better=c['PRESENT'],gap_better=c['GAP'],tie=c['TIE'],present_recall1=round(sum(r['present_rank']==1 for r in group)/n,6),gap_recall1=round(sum(r['gap_rank']==1 for r in group)/n,6),present_mrr=round(statistics.mean(1/r['present_rank'] for r in group),6),gap_mrr=round(statistics.mean(1/r['gap_rank'] for r in group),6),mean_rank_gap=round(statistics.mean(r['gap_rank_minus_present'] for r in group),4)))
    write(OUT/'paired_foreign_summary.csv',psummary)
    (OUT/'analysis_notes.json').write_text(json.dumps(dict(source_coverage='feature_coverage_audit.csv',source_ranks='rank_comparison.csv (rank_before only)',joined_evaluations=len(joined),primary_evaluations=len(primary),primary_exclusions=dict(Counter(r['primary_exclusion_reason'] for r in joined if r['eligible_primary']=='NO')),primary_definition='English/Japanese target only; Korean reference PRESENT; target PRESENT/PARTIAL/MISSING; exclude conflict. GAP=PARTIAL+MISSING. Each intent-target has three query-language evaluations.',limitations=['Coverage is AI-assisted; PRESENT may include only lexical evidence, and independent review is not completed.','Queries are AI-generated and Korean-description-grounded; one designated target per query.','Repeated evaluations are not independent; this is descriptive and no p-values are reported.','Rank model input includes titles and addresses, not only descriptions.','Associations are not evidence of causal impact. Paired English/Japanese comparison controls query identity but not target-language/model effects.']),ensure_ascii=False,indent=2),encoding='utf-8')
    for s in gaps:
        if s['target_language']=='FOREIGN_ALL' and s['language_relation'] in ('ALL','CROSS'):
            print(s)
    print('paired',psummary[0]);print('exclusions',Counter(r['primary_exclusion_reason'] for r in joined if r['eligible_primary']=='NO'))
    print(OUT)

if __name__=='__main__':main()
