"""Exploratory RQ1 comparison using already-run AI-drafted queries."""
import csv,collections
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'data/processed';P=R/'results/common_pilot';O=R/'results/rq1_analysis'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
gap={(x['place_id'],x['language'],x['field']):x for x in read(D/'common_gap_review_final.csv')}
inputs=[('development','dev_semantic_expanded_review.csv','dev_priority_pilot_ranks.csv'),('holdout_exploratory','holdout_exploratory_query_draft.csv','holdout_priority_exploratory_ranks.csv'),('fresh_holdout','fresh_holdout_queries_review.csv','fresh_priority_check_ranks.csv')]
rows=[]
for dataset,qfile,rfile in inputs:
 qs=read(P/qfile); rr={x['query_id']:x for x in read(P/rfile) if x['policy']=='baseline'}
 assert len(rr)==len(qs) and {x['query_id'] for x in qs}==set(rr)
 for q in qs:
  g=gap[(q['place_id'],q['language'],q['field'])];rank=int(rr[q['query_id']]['rank'])
  rows.append(dict(dataset=dataset,query_id=q['query_id'],intent_id=q['intent_id'],place_id=q['place_id'],title_ko=q['title_ko'],language=q['language'],field=q['field'],query=q['query'],information_status=g['tourapi_final_status'],information_review=g['presence_review_provenance'],judgment_note=g['review_note'],rank_before=rank,query_review='PENDING_IN_SOURCE',interpretation='exploratory_not_independent'))
assert len({x['query_id'] for x in rows})==len(rows)
# Keep sets separate: they were created under different selection conditions.
groups=collections.defaultdict(list)
for x in rows:groups[(x['dataset'],x['information_status'])].append(x)
summary=[]
for (dataset,status),rr in sorted(groups.items()):
 ranks=[x['rank_before'] for x in rr]
 summary.append(dict(dataset=dataset,information_status=status,queries=len(rr),places=len({x['place_id'] for x in rr}),r1=sum(x==1 for x in ranks)/len(ranks),r5=sum(x<=5 for x in ranks)/len(ranks),mrr=sum(1/x for x in ranks)/len(ranks)))
write(O/'existing_query_status_and_ranks.csv',rows);write(O/'existing_group_summary.csv',summary)
for x in summary:print(x)
