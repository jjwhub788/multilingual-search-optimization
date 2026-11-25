import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/common_poi_matching"


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


rows = read_csv(RESULT / "priority_review_checked.csv")

# 기존 주소 정규화 분석 결과
expanded = read_csv(
    ROOT / "results/joint_poi_scan/joint_trilingual_high_candidates.csv"
)

index = {
    (r["tourapi_ko_id"], r["aihub_poi_id"]): r
    for r in expanded
}

for row in rows:
    key = (row["tourapi_id"], row["aihub_id"])
    evidence = index.get(key)

    # 직접 검토한 결과는 유지
    if row["review_status"] == "MATCH":
        continue

    if evidence:
        row["auto_decision"] = "AUTO_ADDRESS_CANDIDATE"
        row["auto_reason"] = (
            "정규화한 이름과 상세 주소 일치"
        )

output = RESULT / "priority_review_round2.csv"

with output.open("x", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("직접 검토 결과:", Counter(r["review_status"] for r in rows))
print("자동 분류 결과:", Counter(r["auto_decision"] for r in rows))