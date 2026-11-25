"""RQ4: identical 30 selected facts, same development query set, 3 retrieval layouts.
Field condition: max(query·base_doc, query·each independent field vector).
"""
import csv,hashlib,json,random,statistics
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'data/processed';P=R/'results/common_pilot';O=R/'results/rq4_analysis'
def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,rows):
 with path.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 import numpy as np,torch
 from sentence_transformers import SentenceTransformer
 paths=[D/'common_dataset.csv',D/'common_enrichment_item_pool.csv',D/'common_place_splits_balanced.csv',P/'main_dev_direct_queries.csv',P/'main_dev_direct_priority_allocations.csv']
 docs=[r for r in read(paths[0]) if r['source']=='tourapi'];pool={x['fact_id']:x for x in read(paths[1])};splits={x['place_id']:x['split'] for x in read(paths[2])};queries=read(paths[3]);allocation=[x for x in read(paths[4]) if x['policy']=='shortage']
 docs.sort(key=lambda x:(x['language'],x['place_id']));assert len(docs)==414 and len(allocation)==30 and len({x['fact_id'] for x in allocation})==30
 assert all(splits[x['place_id']]=='dev' and x['split']=='dev' for x in queries)
 bydoc=defaultdict(list)
 for a in allocation:
  f=pool[a['fact_id']]; assert splits[f['place_id']]=='dev';structured=json.loads(f['structured_field_json']);assert structured['fact_id']==f['fact_id'] and structured['field_text'].strip()==f['key_value_text'].strip()
  bydoc[(f['place_id'],f['language'])].append(f)
 outputs=[O/'rq4_rank_results.csv',O/'rq4_summary.csv',O/'rq4_bootstrap.csv',O/'rq4_metadata.json']
 if any(x.exists() for x in outputs):raise FileExistsError('Output already exists')
 texts=[d['title'].strip()+'\n'+d['description'].strip() for d in docs];base_count=len(texts);idx={};lookup={(d['place_id'],d['language']):i for i,d in enumerate(docs)}
 for key,fs in sorted(bydoc.items()):
  fs.sort(key=lambda x:x['fact_id']);j=lookup[key]
  for typ,col in [('natural','natural_text'),('key_value','key_value_text')]:
   idx[(typ,j)]=len(texts);texts.append(texts[j]+'\n'+'\n'.join(x[col].strip() for x in fs))
  for fact in fs:
   idx[('field',j,fact['fact_id'])]=len(texts);texts.append(json.loads(fact['structured_field_json'])['field_text'].strip())
 cache=Path.home()/'.cache/huggingface/hub/models--BAAI--bge-m3';rev=(cache/'refs/main').read_text().strip();m=SentenceTransformer(str(cache/'snapshots'/rev),device='cpu',local_files_only=True);m.max_seq_length=2048
 lens=[len(m.tokenizer.encode(t,truncation=False,add_special_tokens=True)) for t in texts+[x['query'] for x in queries]]
 print('임베딩할 문서·보완 필드',len(texts),flush=True);v=m.encode(texts,batch_size=4,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype(np.float32)
 print('검색어',len(queries),flush=True);q=m.encode([x['query'] for x in queries],batch_size=4,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype(np.float32)
 norms=np.concatenate([np.linalg.norm(v,axis=1),np.linalg.norm(q,axis=1)])
 if not np.isfinite(norms).all() or not np.allclose(norms,1,atol=1e-3):raise ValueError('invalid vectors')
 score=(torch.from_numpy(q)@torch.from_numpy(v).T).numpy()
 if not np.isfinite(score).all() or abs(score).max()>1.001:raise ValueError('invalid similarities')
 rows=[]
 for cond in ('baseline','natural','key_value','field_max'):
  for i,query in enumerate(queries):
   ids=[j for j,d in enumerate(docs) if d['language']==query['language']]
   def get_score(j):
    if cond=='baseline':return float(score[i,j])
    if cond in ('natural','key_value'):return float(score[i,idx.get((cond,j),j)])
    facts=bydoc.get((docs[j]['place_id'],docs[j]['language']),[])
    return max([float(score[i,j])]+[float(score[i,idx[('field',j,f['fact_id'])]]) for f in facts])
   ids.sort(key=lambda j:(-get_score(j),docs[j]['place_id']));rank=next(k for k,j in enumerate(ids,1) if docs[j]['place_id']==query['place_id'])
   rows.append(dict(condition=cond,query_id=query['query_id'],intent_id=query['intent_id'],place_id=query['place_id'],language=query['language'],field=query['field'],rank=rank,top1_place_id=docs[ids[0]]['place_id'],candidate_count=len(ids)))
 write(outputs[0],rows)
 summ=[]
 for cond in ('baseline','natural','key_value','field_max'):
  rr=[int(x['rank']) for x in rows if x['condition']==cond];summ.append(dict(condition=cond,budget_facts=0 if cond=='baseline' else 30,queries=len(rr),places=len({x['place_id'] for x in queries}),r1=sum(x==1 for x in rr)/len(rr),r5=sum(x<=5 for x in rr)/len(rr),mrr=sum(1/x for x in rr)/len(rr),median_rank=statistics.median(rr)))
 write(outputs[1],summ)
 g=defaultdict(dict)
 for x in rows:g[x['condition']][x['query_id']]=x
 boot=[]
 for cond in ('natural','key_value','field_max'):
  clusters=defaultdict(list)
  for qid,b in g['baseline'].items():clusters[b['place_id']].append((int(g[cond][qid]['rank'])==1)-(int(b['rank'])==1))
  ids=sorted(clusters);rng=random.Random(2026);v=[]
  for _ in range(5000):
   sample=[rng.choice(ids) for _ in ids];x=[z for pid in sample for z in clusters[pid]];v.append(sum(x)/len(x))
  v.sort();flat=[z for zs in clusters.values() for z in zs];boot.append(dict(condition=cond,delta_r1=sum(flat)/len(flat),ci95_low=v[124],ci95_high=v[4874],places=len(ids),queries=len(flat),iterations=5000,seed=2026))
 write(outputs[2],boot)
 meta={'status':'DEVELOPMENT_EXPLORATORY','corpus':'138 same-language TourAPI documents; unselected documents unchanged','selection':'fixed 30 shortage-priority fact-language items from existing allocation; same facts in all three conditions','field_max':'encode each field_text separately; document score is max of base document cosine and each field cosine; changes retrieval scoring as well as text structure','query_set':'57 AI-drafted development queries; selected by direct enrichment availability and already observed results; no independence','budget_unit':'30 facts, not equal tokens or costs','model':'BAAI/bge-m3','model_revision':rev,'max_seq_length':2048,'device':'cpu','truncated_inputs':sum(x>2048 for x in lens),'hashes':{str(p.relative_to(R)):sha(p) for p in paths},'limitations':'labels require independent human review; multiple relevant places unjudged; document-length and max score effects; results not causal or independently generalizable'}
 with outputs[3].open('x',encoding='utf-8') as f:json.dump(meta,f,ensure_ascii=False,indent=2)
 print('SUMMARY',summ,'BOOT',boot,flush=True)
if __name__=='__main__':main()
