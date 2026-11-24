import csv
from collections import Counter, defaultdict
from review_tourapi import ROOT, OUT, read, distance

review_path = OUT
dest = ROOT / 'tourapi_final_matches.csv'
high = read(ROOT / 'tourapi_high_confidence.csv')
reviewed = read(review_path)
candidates = read(ROOT / 'tourapi_match_candidates.csv')
clean = read(ROOT / 'tourapi_clean.csv')
langs = ('ko', 'en', 'ja')
base = list(high[0])
assert len(high) == 279 and len(reviewed) == 63 and len(candidates) == 342
assert all(r['match_en'] == r['match_ja'] == 'True' for r in high)
assert Counter(r['review_status'] for r in reviewed) == Counter(MATCH=52, REVIEW=6, UNMATCH=5)
assert not ({r['coord'] for r in high} & {r['coord'] for r in reviewed})
assert {tuple(r[k] for k in candidates[0]) for r in high+reviewed} == {tuple(r.values()) for r in candidates}
idx = defaultdict(list)
coord_index = defaultdict(list)
for r in clean:
    idx[r['language'], r['source_contentid']].append(r)
    try:
        x,y = float(r['mapx']),float(r['mapy'])
        if 124<=x<=132 and 33<=y<=39.5:
            coord_index[f'{round(x,5)}_{round(y,5)}'].append((r['language'],r['source_contentid']))
    except ValueError:
        pass

selection = {r['coord']:r for r in reviewed if r['review_status']=='MATCH'}
rows = []
for c in high + list(selection.values()):
    row = {k:c[k] for k in base}
    source = 'SOURCE_REVIEW' if c['coord'] in selection else 'AUTO_NAME_MATCH'
    row.update(match_status='MATCH', match_source=source,
               review_note=c.get('review_note','기존 한·영·일 이름 자동 일치'),
               address_status=c.get('address_status','NOT_SEMANTICALLY_REVIEWED'),
               address_note=c.get('address_note',''),
               sample_review_status='PENDING' if source=='AUTO_NAME_MATCH' else 'AI_REVIEWED',
               external_verification='NOT_PERFORMED')
    recs = {}
    assert Counter(coord_index[c['coord']])==Counter((l,c['id_'+l]) for l in langs),c
    for l in langs:
        matches=idx[l,c['id_'+l]]
        assert len(matches)==1 and matches[0]['title']==c['title_'+l],c
        recs[l]=matches[0]
        for k in ('addr1','addr2','mapx','mapy'):
            row[k+'_'+l]=matches[0][k]
    row['max_distance_m']=f"{max(distance(recs[a],recs[b]) for a,b in [('ko','en'),('ko','ja'),('en','ja')]):.6f}"
    rows.append(row)

# Original candidate order, rather than grouping rows by origin.
rank = {r['coord']:i for i,r in enumerate(candidates)}
rows.sort(key=lambda r:rank[r['coord']])
assert len(rows)==331 and len({r['coord'] for r in rows})==331
assert len({r['id_'+l] for r in rows for l in langs})==993
assert not ({r['coord'] for r in rows} & {r['coord'] for r in reviewed if r['review_status']!='MATCH'})
with dest.open('x', encoding='utf-8-sig', newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
assert read(dest)==rows
print('file', dest)
print('count', len(rows), 'origin', Counter(r['match_source'] for r in rows))
print('unique IDs',len({r['id_'+l] for r in rows for l in langs}))
print('unique coordinates',len({r['coord'] for r in rows}))
print('maximum distance', max(float(r['max_distance_m']) for r in rows))
