import csv
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RESULT = ROOT / "results/common_poi_matching"


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize(text):
    return re.sub(r"\W+", "", text).lower()


# 공통 관광지 후보 및 TourAPI 한·영·일 연결 자료
candidates = read_csv(RESULT / "candidate_pairs.csv")
matches = read_csv(DATA / "tourapi_final_matches.csv")

links = {
    r["id_ko"]: r
    for r in matches
    if r["match_status"] == "MATCH"
    and r["id_en"]
    and r["id_ja"]
}

review_rows = []

for row in candidates:

    # 양쪽 출처에 필요한 언어 자료가 있는 후보만 선별
    if row["tourapi_id"] not in links:
        continue

    if row["aihub_has_en"] != "True" or row["aihub_has_ja"] != "True":
        continue

    names = [
        normalize(row[k])
        for k in [
            "tourapi_title",
            "aihub_title_en_source",
            "aihub_title_ja_source",
        ]
    ]

    addresses = [
        normalize(row[k])
        for k in [
            "tourapi_address",
            "aihub_address_en_source",
            "aihub_address_ja_source",
        ]
    ]

    if (
        all(names)
        and len(set(names)) == 1
        and all(addresses)
        and len(set(addresses)) == 1
        and row["priority"] == "high"
    ):
        row["auto_decision"] = "AUTO_MATCH_CANDIDATE"
        row["auto_reason"] = "이름 및 주소 일치, 일대일 후보"
    else:
        row["auto_decision"] = "REVIEW"
        row["auto_reason"] = "추가 확인 필요"

    review_rows.append(row)

review_rows.sort(key=lambda r: r["auto_decision"])

output = RESULT / "priority_review.csv"

with output.open("x", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=review_rows[0].keys())
    writer.writeheader()
    writer.writerows(review_rows)

print("전체 검토 후보:", len(review_rows))
print(Counter(r["auto_decision"] for r in review_rows))