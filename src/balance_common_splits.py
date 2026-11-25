"""Ensure exploratory experience coverage in the holdout before query creation."""
import csv
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'

def read(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

rows = read('common_place_splits.csv')
items = read('common_enrichment_item_pool.csv')
fields = defaultdict(set)
for item in items:
    fields[item['place_id']].add(item['field'])
by_id = {r['place_id']: r for r in rows}
assert len(by_id) == len(rows) and all(x['place_id'] in by_id for x in items)
experience_dev = sorted(r['place_id'] for r in rows if r['split'] == 'dev' and 'experience' in fields[r['place_id']])
experience_test = [r for r in rows if r['split'] == 'test' and 'experience' in fields[r['place_id']]]
if experience_test or len(experience_dev) < 2:
    raise ValueError('Review experience distribution manually; no swap made.')
# Deterministically swap one experience place with a test place that has the
# same remaining field categories. No search outcomes or queries are used.
move_dev = None
move_test = None
for dev in experience_dev:
    matches = sorted(r['place_id'] for r in rows if r['split'] == 'test' and fields[r['place_id']] == fields[dev] - {'experience'})
    if matches:
        move_dev, move_test = dev, matches[0]
        break
if not move_dev:
    raise ValueError('No matching field-category swap found; review manually.')
by_id[move_dev]['split'] = 'test'
by_id[move_test]['split'] = 'dev'
output = DATA / 'common_place_splits_balanced.csv'
with output.open('x', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['place_id', 'title_ko', 'split'])
    writer.writeheader(); writer.writerows(rows)
print('최종 평가용으로 이동:', by_id[move_dev]['title_ko'], move_dev)
print('개발용으로 이동:', by_id[move_test]['title_ko'], move_test)
counts = Counter((by_id[i['place_id']]['split'], i['field']) for i in items)
for key, n in sorted(counts.items()): print(*key, n)
print('저장:', output)
