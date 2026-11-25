import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "data/raw/tourapi"))

import collect_matched_details as collector

DATA = ROOT / "data/processed"

links = collector.read_csv(DATA / "common_poi_links.csv")
tour = collector.read_csv(DATA / "tourapi_clean.csv")

index = {
    (r["language"], r["source_contentid"]): r
    for r in tour
}

cache = collector.load_cache()

result = []

for place in links:
    for lang in ["ko", "en", "ja"]:

        cid = place[f"tourapi_{lang}_id"]
        r = index[(lang, cid)]
        overview = cache.get((lang, cid), "")

        result.append({
            "source": "tourapi",
            "place_id": place["place_id"],
            "language": lang,
            "source_contentid": cid,
            "title": r["title"],
            "address": r["addr1"],
            "description": overview,
            "collection_status": (
                "COMPLETE" if overview
                else "EMPTY" if (lang, cid) in cache
                else "PENDING"
            )
        })

collector.write_csv(
    DATA / "common_tourapi_details.csv",
    result
)

print("전체 문서:", len(result))
print(Counter(r["collection_status"] for r in result))