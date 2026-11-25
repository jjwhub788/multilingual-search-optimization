"""Development-only equal-fact-budget priority comparison; AI query drafts, not gold."""
import argparse,csv,hashlib,json,random
from collections import Counter, defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'data/processed';O=R/'results/common_pilot'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,rows):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 import numpy as np,torch
 from sentence_transformers import SentenceTransformer
 P=argparse.ArgumentParser();P.add_argument('--device',default='cpu',choices=['cpu','mps']);P.add_argument('--batch-size',type=int,default=4);P.add_argument('--budget',type=int,default=15);args=P.parse_args()
 paths=[D/'common_dataset.csv',D/'common_enrichment_item_pool.csv',D/'common_gap_review_final.csv',D/'common_place_splits_balanced.csv',O/'dev_semantic_expanded_review.csv']
 docs,items,gaps,splits,queries=[read(p) for p in paths];sm={r['place_id']:r['split'] for r in splits}
 docs=[r for r in docs if r['source']=='tourapi']; pool=[r for r in items if sm[r['place_id']]=='dev'];docs.sort(key=lambda x:(x['language'],x['place_id']));pool.sort(key=lambda x:x['fact_id'])
 assert len(docs)==414 and len({(x['place_id'],x['language']) for x in docs})==414 and all(sm[x['place_id']]=='dev' and x['split']=='dev' for x in queries)
 assert len({x['query_id'] for x in queries})==len(queries) and len({x['fact_id'] for x in pool})==len(pool)
 if not 0<args.budget<=len(pool):raise ValueError('invalid budget')
 targets=[O/x for x in ['dev_priority_transferable_ranks.csv','dev_priority_transferable_summary.csv','dev_priority_transferable_allocations.csv','dev_priority_transferable_metadata.json']]
 if any(p.exists() for p in targets):raise FileExistsError('Output exists; preserve or move pilot outputs before rerun')
 shortage=Counter((r['place_id'],r['language']) for r in gaps if r['tourapi_final_status'] in ('MISSING','PARTIAL'))
 # Query prevalence, not target-place relevance: selection uses only development queries.
 prevalence=Counter(r['field'] for r in queries)
 allocations={}
 for seed in range(10):allocations['random_'+str(seed)]=random.Random(seed).sample(pool,args.budget)
 allocations['shortage']=sorted(pool,key=lambda r:(-shortage[(r['place_id'],r['language'])],r['fact_id']))[:args.budget]
 tie=lambda r: hashlib.sha256(('seed42:'+r['fact_id']).encode()).hexdigest()
 allocations['frequency_deficit']=sorted(pool,key=lambda r:(-(prevalence[r['field']]/len(queries)) * (1 + shortage[(r['place_id'],r['language'])]/5),tie(r)))[:args.budget]
 assert all(len(v)==args.budget and len({r['fact_id'] for r in v})==args.budget for v in allocations.values())
 # Embed baseline plus unique changed documents, then reuse vectors across all queries.
 texts=[r['title'].strip()+'\n'+r['description'].strip() for r in docs]; lookup={(r['place_id'],r['language']):i for i,r in enumerate(docs)};idx={};allocated=[]
 for policy,facts in allocations.items():
  bydoc=defaultdict(list)
  for r in facts:bydoc[(r['place_id'],r['language'])].append(r);allocated.append({'policy':policy,'fact_id':r['fact_id'],'place_id':r['place_id'],'language':r['language'],'field':r['field'],'cost':1})
  for key,ff in bydoc.items():
   j=lookup[key];ff.sort(key=lambda r:r['fact_id']);text=texts[j]+'\n'+'\n'.join(r['natural_text'].strip() for r in ff)
   idx[(policy,j)]=len(texts);texts.append(text)
 cache=Path.home()/'.cache/huggingface/hub/models--BAAI--bge-m3';ref=(cache/'refs/main').read_text().strip();model=SentenceTransformer(str(cache/'snapshots'/ref),device=args.device,local_files_only=True);model.max_seq_length=2048
 lengths=[len(model.tokenizer.encode(t,add_special_tokens=True,truncation=False)) for t in texts]; qtext=[r['query'] for r in queries];qlens=[len(model.tokenizer.encode(t,add_special_tokens=True,truncation=False)) for t in qtext]
 print('encoding corpus variants',len(texts),flush=True)
 dv=model.encode(texts,batch_size=args.batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype(np.float32)
 print('encoding development queries',len(qtext),flush=True)
 qv=model.encode(qtext,batch_size=args.batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype(np.float32)
 norms=np.concatenate([np.linalg.norm(dv,axis=1),np.linalg.norm(qv,axis=1)])
 if not np.isfinite(norms).all() or not np.allclose(norms,1,atol=1e-3):raise ValueError('invalid embeddings')
 scores=(torch.from_numpy(qv) @ torch.from_numpy(dv).T).numpy()
 if not np.isfinite(scores).all() or abs(scores).max()>1.001:raise ValueError('invalid cosine scores')
 ranks=[]
 for pol in ['baseline']+list(allocations):
  for i,q in enumerate(queries):
   ind=[j for j,d in enumerate(docs) if d['language']==q['language']]
   def score(j):return float(scores[i,idx.get((pol,j),j)])
   ind.sort(key=lambda j:(-score(j),docs[j]['place_id']))
   rank=next(k for k,j in enumerate(ind,1) if docs[j]['place_id']==q['place_id'])
   ranks.append({'policy':pol,'query_id':q['query_id'],'intent_id':q['intent_id'],'place_id':q['place_id'],'language':q['language'],'field':q['field'],'rank':rank,'candidate_count':len(ind),'review_status':'PENDING'})
 summary=[]
 for pol in ['baseline']+list(allocations):
  rr=[int(x['rank']) for x in ranks if x['policy']==pol];summary.append({'policy':pol,'budget_facts':0 if pol=='baseline' else args.budget,'queries':len(rr),'places':len({x['place_id'] for x in queries}),'r_at_1':sum(x==1 for x in rr)/len(rr),'r_at_5':sum(x<=5 for x in rr)/len(rr),'mrr':sum(1/x for x in rr)/len(rr)})
 write(targets[0],ranks);write(targets[1],summary);write(targets[2],allocated)
 meta={'status':'EXPLORATORY_AI_QUERY_DRAFTS_NOT_REVIEWED','seed':list(range(10)),'budget_unit':'equal count of fact-language items; token cost is not equal','budget':args.budget,'candidate_scope':'138 same-language original documents; only development documents changed','frequency_deficit':'development query field frequency multiplied by (1 + missing_or_partial_fields/5), deterministic hash tie-break; uses no query target identity','shortage':'count missing/partial fields in each place-language document, using AI-reviewed gap table','random_trials':10,'max_seq_length':2048,'model_revision_snapshot':ref,'input_sha256':{str(p.relative_to(R)):sha(p) for p in paths},'truncated_documents_including_variants':sum(n>2048 for n in lengths),'truncated_queries':sum(n>2048 for n in qlens),'warning':'No statistical claim; relevance, language, and facts require human review; multiple relevant places possible; independent test queries not used.'}
 with targets[3].open('x',encoding='utf-8') as f:json.dump(meta,f,indent=2,ensure_ascii=False)
 print('completed',summary,flush=True)
if __name__=='__main__':main()
