"""Paired place-cluster bootstrap on exploratory heldout results, not confirmatory."""
import csv,random
from collections import defaultdict
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'results/common_pilot'
def read(name):
 with (P/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
ranks=read('holdout_priority_exploratory_ranks.csv'); grouped=defaultdict(dict)
for row in ranks:grouped[row['policy']][row['query_id']]=row
base=grouped['baseline'];pairs={}
for policy in ('shortage','frequency_deficit'):
 clusters=defaultdict(list)
 for q,b in base.items():
  a=grouped[policy][q];clusters[b['place_id']].append((int(a['rank'])==1)-(int(b['rank'])==1))
 rng=random.Random(2026);ids=sorted(clusters);samples=[]
 for _ in range(5000):
  chosen=[rng.choice(ids) for _ in ids];d=[x for pid in chosen for x in clusters[pid]];samples.append(sum(d)/len(d))
 samples.sort();allv=[v for vs in clusters.values() for v in vs]
 pairs[policy]=dict(policy=policy,delta_recall1=sum(allv)/len(allv),ci95_low=samples[124],ci95_high=samples[4874],places=len(ids),queries=len(allv),iterations=5000,seed=2026,status='exploratory_not_independent_not_confirmatory')
out=P/'holdout_priority_exploratory_bootstrap.csv'
with out.open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(next(iter(pairs.values()))));w.writeheader();w.writerows(pairs.values())
print(out)
