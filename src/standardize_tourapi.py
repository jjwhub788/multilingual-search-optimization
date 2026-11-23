"""Export the collected TourAPI pilot to the agreed AI Hub/TourAPI schema."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data/processed'
INPUT = PROCESSED / 'tourapi_pilot_details_300.csv'
OUTPUT = PROCESSED / 'tourapi_standardized.csv'
FIELDS = ['source', 'place_id', 'language', 'title', 'description']

with INPUT.open(encoding='utf-8-sig', newline='') as f:
    records = list(csv.DictReader(f))

assert len(records) == 300
assert all(r['collection_status'] == 'COMPLETE' for r in records)
rows = [dict(source='tourapi', place_id=r['place_id'], language=r['language'],
             title=r['title'], description=r['overview']) for r in records]
assert all(r[k].strip() for r in rows for k in FIELDS)
assert len({(r['source'], r['place_id'], r['language']) for r in rows}) == len(rows)
groups = defaultdict(set)
for r in rows:
    groups[r['place_id']].add(r['language'])
assert len(groups) == 100 and all(v == {'ko', 'en', 'ja'} for v in groups.values())

# Preserve the original cleaned, detailed and raw source files.
with OUTPUT.open('x', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

with OUTPUT.open(encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    assert reader.fieldnames == FIELDS and list(reader) == rows
print(OUTPUT)
print('관광지', len(groups), '문서', len(rows), '언어별', dict(Counter(r['language'] for r in rows)))
print('결측, 중복, 언어 누락: 0')
