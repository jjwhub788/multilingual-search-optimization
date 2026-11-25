import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

old = read_csv(
    ROOT / "results/common_poi_matching/priority_review_round2_checked.csv"
)

new = read_csv(
    ROOT / "results/joint_poi_scan/new_trilingual_review_web_verified.csv"
)

tour = read_csv(
    ROOT / "data/processed/tourapi_final_matches.csv"
)

links = {r["id_ko"]: r for r in tour if r["match_status"] == "MATCH"}

result = []

# 기존 31곳
for r in old:
    if r["review_status"] != "MATCH":
        continue

    t = links[r["tourapi_id"]]

    result.append({
        "place_id": t["id_ko"],
        "tourapi_ko_id": t["id_ko"],
        "tourapi_en_id": t["id_en"],
        "tourapi_ja_id": t["id_ja"],
        "aihub_poi_id": r["aihub_id"],
        "title_ko": r["tourapi_title"],
        "review_note": r["review_note"]
    })

# 추가 107곳
for r in new:
    if r["team_final_decision"] != "MATCH":
        continue

    result.append({
        "place_id": r["tourapi_ko_id"],
        "tourapi_ko_id": r["tourapi_ko_id"],
        "tourapi_en_id": r["tourapi_en_id"],
        "tourapi_ja_id": r["tourapi_ja_id"],
        "aihub_poi_id": r["aihub_poi_id"],
        "title_ko": r["tourapi_title_ko"],
        "review_note": r["team_review_note"]
    })

for key in ["tourapi_ko_id", "tourapi_en_id",
            "tourapi_ja_id", "aihub_poi_id"]:
    assert all(r[key] for r in result)
    assert len(set(r[key] for r in result)) == len(result)

output = ROOT / "data/processed/common_poi_links.csv"

with output.open("x", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result[0].keys())
    writer.writeheader()
    writer.writerows(result)

print("공통 관광지:", len(result))