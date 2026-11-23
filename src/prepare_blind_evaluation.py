"""Create an annotation template that does not disclose intervention labels or scores."""
import csv
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/tourapi_standardized.csv'
OUT=ROOT/'data/processed/tourapi_blind_query_template.csv'
PRIOR=ROOT/'results/bge_m3_dense_full/queries.csv'
RETEST=ROOT/'results/tourapi_description_enrichment_retest/new_queries.csv'
AUG=ROOT/'results/tourapi_description_enrichment/description_changes.csv'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def main():
    docs={r['place_id']:r for r in read(DATA) if r['language']=='ko'}
    changed={r['place_id'] for r in read(AUG)}
    previous={r['place_id'] for r in read(PRIOR) if r['query_type']=='SEMANTIC'} | {r['place_id'] for r in read(RETEST)}
    assert len(docs)==100 and len(changed)==3
    rng=random.Random(20261120)
    ids=sorted(changed)+rng.sample(sorted(set(docs)-previous-changed),12)
    rng.shuffle(ids)
    rows=[]
    for i,pid in enumerate(ids,1):
        for j in (1,2):
            rows.append(dict(intent_id=f'B{i:02d}-{j}',place_id=pid,title_ko=docs[pid]['title'],reference_description_ko=docs[pid]['description'],query_ko='',query_en='',query_ja='',annotator_note=''))
    assert len(rows)==30 and len({r['place_id'] for r in rows})==15
    assert len({r['intent_id'] for r in rows})==30
    with OUT.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert read(OUT)==rows
    print(OUT, 'places',len(ids),'intents',len(rows),'query slots',len(rows)*3)

if __name__=='__main__': main()
