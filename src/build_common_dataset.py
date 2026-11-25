import csv
import unicodedata
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize(text):
    return "".join(
        unicodedata.normalize("NFKC", text).split()
    )


links = read_csv(DATA / "common_poi_links.csv")
tour = read_csv(DATA / "common_tourapi_details.csv")

aihub_path = next(DATA.glob("aihub_clean*.csv"))
aihub = read_csv(aihub_path)

index = {
    (r["poi_id"], r["language"]): r
    for r in aihub
}

result = []
conflicts = []

# TourAPI 원본 정보
for r in tour:
    result.append({
        "source": "tourapi",
        "place_id": r["place_id"],
        "language": r["language"],
        "title": r["title"],
        "description": r["description"],
        "attributes": "",
        "status": "COMPLETE"
    })

# AI Hub 정보 연결
for link in links:

    aid = link["aihub_poi_id"]
    en = index[(aid, "en")]
    ja = index[(aid, "ja")]

    for r in [en, ja]:
        result.append({
            "source": "aihub",
            "place_id": link["place_id"],
            "language": r["language"],
            "title": r["title"],
            "description": "",
            "attributes": r["attributes"],
            "status": "COMPLETE"
        })

    same = normalize(en["attributes_ko"]) == normalize(
        ja["attributes_ko"]
    )

    # 한국어 원문 간 불일치 별도 기록
    if not same:
        conflicts.append({
            "place_id": link["place_id"],
            "title_ko": link["title_ko"],
            "attributes_ko_from_en": en["attributes_ko"],
            "attributes_ko_from_ja": ja["attributes_ko"]
        })

    result.append({
        "source": "aihub",
        "place_id": link["place_id"],
        "language": "ko",
        "title": en["title_ko"],
        "description": "",
        "attributes": en["attributes_ko"] if same else "",
        "status": "COMPLETE" if same else "REVIEW"
    })


def save(name, rows):
    if not rows:
        return

    with (DATA / name).open(
        "x", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)


save("common_dataset.csv", result)
save("common_aihub_ko_review.csv", conflicts)

print("전체 데이터:", len(result))
print("출처별 데이터:", Counter(r["source"] for r in result))
print("AI Hub 한국어 원문 검토 필요:", len(conflicts))