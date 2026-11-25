"""Produce reviewable TourAPI–AI Hub candidate links and source documents.

Exact normalized Korean names generate candidates; address agreement is evidence,
not proof. No source files are modified. Run from any working directory.
"""
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
OUT = ROOT / 'results/common_poi_matching'
OUT.mkdir(parents=True, exist_ok=True)

def read(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def norm(s):
    return re.sub(r'[^가-힣a-z0-9]', '', unicodedata.normalize('NFKC', s).lower())

def variants(s):
    # Parenthetical aliases are only candidate-generation evidence.
    parts = [s, re.sub(r'\s*[（(][^）)]*[）)]', '', s)]
    return {norm(p) for p in parts if len(norm(p)) >= 2}

def write(name, rows, fields):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

ai = read('aihub_poi.csv')
tour = [r for r in read('tourapi_clean.csv') if r['language'] == 'ko']
index = defaultdict(set)
for i, a in enumerate(ai):
    for name in (a['title_ko_en'], a['title_ko_ja']):
        for v in variants(name): index[v].add(i)
pairs = []
for t in tour:
    found = set()
    for v in variants(t['title']): found.update(index.get(v, ()))
    for i in sorted(found):
        a = ai[i]
        taddr = norm(t['addr1'] + ' ' + t['addr2'])
        addresses = [norm(a[k]) for k in ('address_ko_en','address_ko_ja') if a[k].strip()]
        # At least 8 characters avoids classifying a shared city alone as an address match.
        agreement = any(len(x) >= 8 and (x in taddr or taddr in x and len(taddr) >= 8) for x in addresses)
        exact = norm(t['title']) in {norm(a['title_ko_en']), norm(a['title_ko_ja'])} - {''}
        pairs.append(dict(tourapi_id=t['source_contentid'], aihub_id=a['poi_id'], tourapi_title=t['title'],
          aihub_title_en_source=a['title_ko_en'], aihub_title_ja_source=a['title_ko_ja'],
          tourapi_address=' '.join(x for x in (t['addr1'],t['addr2']) if x),
          aihub_address_en_source=a['address_ko_en'], aihub_address_ja_source=a['address_ko_ja'],
          name_match='exact_normalized' if exact else 'parenthesis_variant',
          address_match='normalized_containment' if agreement else 'unverified',
          aihub_has_en=a['has_en'], aihub_has_ja=a['has_ja'], aihub_name_mismatch=a['name_mismatch'],
          review_status='PENDING', review_note=''))
tc=Counter(p['tourapi_id'] for p in pairs); ac=Counter(p['aihub_id'] for p in pairs)
for p in pairs:
    p['tourapi_candidate_count']=tc[p['tourapi_id']]; p['aihub_candidate_count']=ac[p['aihub_id']]
    p['priority']='high' if p['name_match']=='exact_normalized' and p['address_match']=='normalized_containment' and tc[p['tourapi_id']]==1 and ac[p['aihub_id']]==1 else 'review'
pairs.sort(key=lambda p:(p['priority']!='high',p['tourapi_title'],p['aihub_id']))
fields=['priority','review_status','review_note','tourapi_id','aihub_id','tourapi_title','aihub_title_en_source','aihub_title_ja_source','tourapi_address','aihub_address_en_source','aihub_address_ja_source','name_match','address_match','tourapi_candidate_count','aihub_candidate_count','aihub_has_en','aihub_has_ja','aihub_name_mismatch']
write('candidate_pairs.csv',pairs,fields)
# Preserve each source's documents separately; candidate IDs are provisional.
clean=read('aihub_clean.csv'); detail=read('tourapi_clean.csv'); links=read('tourapi_final_matches.csv')
ai_by=defaultdict(list); tour_by={(r['language'],r['source_contentid']):r for r in detail}; link_by={r['id_ko']:r for r in links if r['match_status']=='MATCH'}
for r in clean: ai_by[r['poi_id']].append(r)
docs=[]
for p in pairs:
    tid=p['tourapi_id']; aid=p['aihub_id']; group='tourapi:'+tid+'|aihub:'+aid
    ids={'ko':tid}; link=link_by.get(tid)
    if link: ids.update({l:link['id_'+l] for l in ('en','ja') if link['id_'+l]})
    for lang,id_ in ids.items():
        r=tour_by.get((lang,id_))
        if r: docs.append(dict(candidate_pair=group,priority=p['priority'],source='tourapi',source_place_id=tid,source_document_id=id_,language=lang,title=r['title'],address=' '.join(x for x in (r['addr1'],r['addr2']) if x),description=r['overview'],attributes='',source_split=''))
    # Keep both Korean versions if they differ, without silently choosing one.
    ko_seen=set()
    for r in ai_by[aid]:
        docs.append(dict(candidate_pair=group,priority=p['priority'],source='aihub',source_place_id=aid,source_document_id=aid+':'+r['language'],language=r['language'],title=r['title'],address=r['address'],description='',attributes=r['attributes'],source_split=r['source_split']))
        key=(r['title_ko'],r['address_ko'],r['attributes_ko'])
        if key not in ko_seen:
            ko_seen.add(key)
            docs.append(dict(candidate_pair=group,priority=p['priority'],source='aihub',source_place_id=aid,source_document_id=aid+':ko_from_'+r['language'],language='ko',title=r['title_ko'],address=r['address_ko'],description='',attributes=r['attributes_ko'],source_split=r['source_split']))
write('candidate_documents.csv',docs,['candidate_pair','priority','source','source_place_id','source_document_id','language','title','address','description','attributes','source_split'])
print('pairs',len(pairs),'high',sum(p['priority']=='high' for p in pairs),'tour places',len(tc),'ai places',len(ac),'documents',len(docs))
print('high with linked tour en/ja',sum(p['priority']=='high' and p['tourapi_id'] in link_by for p in pairs))
