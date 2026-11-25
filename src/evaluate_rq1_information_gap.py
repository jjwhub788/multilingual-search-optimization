"""RQ1: query-specific labels, matched place, cross-language baseline search."""
import csv,json,hashlib,random,statistics
from pathlib import Path
from collections import defaultdict
R=Path(__file__).resolve().parents[1];D=R/'data/processed';O=R/'results/rq1_analysis'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import numpy as np,torch
 from sentence_transformers import SentenceTransformer
 docs=[x for x in read(D/'common_dataset.csv') if x['source']=='tourapi'];qs=read(O/'rq1_queries.csv');labels={(x['intent_id'],x['document_language']):x for x in read(O/'rq1_intent_labels.csv')}
 assert len(docs)==414 and len(qs)==30 and len(labels)==30
 assert len({(d['place_id'],d['language']) for d in docs})==414
 paths=[O/'rq1_rank_results.csv',O/'rq1_group_summary.csv',O/'rq1_paired_summary.csv',O/'rq1_metadata.json']
 if any(p.exists() for p in paths):raise FileExistsError('Output already exists')
 cache=Path.home()/'.cache/huggingface/hub/models--BAAI--bge-m3';rev=(cache/'refs/main').read_text().strip();model=SentenceTransformer(str(cache/'snapshots'/rev),device='cpu',local_files_only=True);model.max_seq_length=2048
 texts=[d['title'].strip()+'\n'+d['description'].strip() for d in docs];qtext=[q['query'] for q in qs]
 lengths=[len(model.tokenizer.encode(t,truncation=False,add_special_tokens=True)) for t in texts+qtext]
 print('문서 임베딩',len(docs),flush=True)
 dv=model.encode(texts,batch_size=4,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype(np.float32)
 print('검색어 임베딩',len(qs),flush=True)
 qv=model.encode(qtext,batch_size=4,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype(np.float32)
 norms=np.concatenate([np.linalg.norm(dv,axis=1),np.linalg.norm(qv,axis=1)])
 if not np.isfinite(norms).all() or not np.allclose(norms,1,atol=.001):raise ValueError('invalid vectors')
 scores=(torch.from_numpy(qv)@torch.from_numpy(dv).T).numpy()
 if not np.isfinite(scores).all() or np.max(np.abs(scores))>1.001:raise ValueError('invalid similarities')
 rows=[]
 for i,q in enumerate(qs):
  for lang in ('ko','en','ja'):
   candidates=[j for j,d in enumerate(docs) if d['language']==lang];candidates.sort(key=lambda j:(-float(scores[i,j]),docs[j]['place_id']))
   rank=next(k for k,j in enumerate(candidates,1) if docs[j]['place_id']==q['place_id']);label=labels[(q['intent_id'],lang)]
   rows.append(dict(intent_id=q['intent_id'],query_id=q['query_id'],place_id=q['place_id'],split=q['split'],query_language=q['language'],document_language=lang,same_language='YES' if q['language']==lang else 'NO',information_status=label['query_specific_status'],rank=rank,top1_place_id=docs[candidates[0]]['place_id'],candidate_count=len(candidates),human_review='PENDING'))
 write(paths[0],rows)
 summary=[]
 for subset in ('same_language','all_language_pairs'):
  chosen=[x for x in rows if subset=='all_language_pairs' or x['same_language']=='YES']
  for status in ('PRESENT','MISSING'):
   v=[x for x in chosen if x['information_status']==status];r=[x['rank'] for x in v]
   summary.append(dict(subset=subset,information_status=status,rows=len(v),places=len({x['place_id'] for x in v}),recall_at_1=sum(x==1 for x in r)/len(r),recall_at_5=sum(x<=5 for x in r)/len(r),mrr=sum(1/x for x in r)/len(r),median_rank=statistics.median(r)))
 write(paths[1],summary)
 # Compare present vs missing document languages for the exact same query.
 # Average within each query first, then resample places to preserve repeated queries.
 perquery=defaultdict(list)
 for x in rows:perquery[x['query_id']].append(x)
 clusters=defaultdict(list)
 for qid,v in perquery.items():
  a=[x for x in v if x['information_status']=='PRESENT'];b=[x for x in v if x['information_status']=='MISSING'];assert a and b
  a_r1=sum(x['rank']==1 for x in a)/len(a);b_r1=sum(x['rank']==1 for x in b)/len(b)
  a_mrr=sum(1/x['rank'] for x in a)/len(a);b_mrr=sum(1/x['rank'] for x in b)/len(b)
  clusters[v[0]['place_id']].append((a_r1-b_r1,a_mrr-b_mrr))
 pairs=[x for z in clusters.values() for x in z];rng=random.Random(2026);ids=sorted(clusters);boot=[[],[]]
 for _ in range(5000):
  sampled=[rng.choice(ids) for _ in ids];rr=[x for pid in sampled for x in clusters[pid]]
  for k in (0,1):boot[k].append(sum(x[k] for x in rr)/len(rr))
 paired=[]
 for k,name in enumerate(('recall_at_1','mrr')):
  v=sorted(boot[k]);paired.append(dict(metric=name,present_minus_missing=sum(x[k] for x in pairs)/len(pairs),ci95_low=v[124],ci95_high=v[4874],places=len(ids),queries=len(pairs),bootstrap_iterations=5000,seed=2026))
 write(paths[2],paired)
 meta={'status':'EXPLORATORY_AI_LABELED_NOT_INDEPENDENT','selection':'10 places deliberately selected for presence-missing contrast; some test places previously used, no holdout claims','query_design':'each query asks for an activity and contains location/context; each of 30 queries searched in all three language-specific corpora','grouping':'same query ranked against each 138-document language corpus, then within-query present-minus-missing, bootstrap by place','label_warning':'query-specific labels manually AI-reviewed against snapshot, not independently human reviewed; MISSING means not mentioned in document only','interpretation':'correlation not causal; document language, content, candidate sets, query formulation, and model language behavior are confounded; 3 queries per place not independent','model':'BAAI/bge-m3','revision':rev,'max_seq_length':2048,'truncated_inputs':sum(x>2048 for x in lengths),'hashes':{str(p.relative_to(R)):sha(p) for p in [D/'common_dataset.csv',O/'rq1_queries.csv',O/'rq1_intent_labels.csv']}}
 with paths[3].open('x',encoding='utf-8') as f:json.dump(meta,f,ensure_ascii=False,indent=2)
 print('summary',summary,'paired',paired,flush=True)
if __name__=='__main__':main()
