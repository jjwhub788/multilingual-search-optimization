import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results/joint_poi_scan"

with (FOLDER / "joint_trilingual_high_candidates.csv").open(
    encoding="utf-8-sig"
) as f:
    rows = list(csv.DictReader(f))

new = [
    r for r in rows
    if r["tourapi_trilingual_status"] == "NEW_ALIAS_COORD_REVIEW"
]

for r in new:
    distance = max(
        float(r["new_en_coord_distance_m"]),
        float(r["new_ja_coord_distance_m"])
    )

    r["max_distance_m"] = distance
    r["coord_priority"] = (
        "COORD_MATCH" if distance == 0 else "NEAR_COORD"
    )

new.sort(key=lambda r: r["max_distance_m"])

with (FOLDER / "new_trilingual_review.csv").open(
    "x", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=new[0].keys())
    writer.writeheader()
    writer.writerows(new)

print("추가 관광지 후보:", len(new))
print(Counter(r["coord_priority"] for r in new))