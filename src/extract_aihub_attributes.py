import csv
import unicodedata
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"

# 정보 유형, 공통 항목명, 한국어·영어·일본어 원본 항목명
FIELDS = [
    ("activity", "tourism_type", "관광타입", "Tourism Type", "観光タイプ"),
    ("activity", "experience", "체험프로그램", "Experience Program(s)", "体験プログラム"),
    ("operation", "opening_hours", "이용시간", "Opening Hours", "利用時間"),
    ("operation", "holidays", "휴무일", "Holidays", "定休日"),
    ("cost", "admission_fee", "입장료/시설이용료", "Admission Fee", "入場料/施設利用料"),
    ("accessibility", "parking", "주차시설 유무", "Parking Available", "駐車施設有無"),
]


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_attributes(text):
    fields = {}
    for part in text.split(" | "):
        key, sep, value = part.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


aihub = [r for r in read_csv("common_dataset.csv") if r["source"] == "aihub"]
links = {r["place_id"]: r["aihub_poi_id"] for r in read_csv("common_poi_links.csv")}
variants = {
    r["place_id"]: (
        parse_attributes(r["attributes_ko_from_en"]),
        parse_attributes(r["attributes_ko_from_ja"]),
    )
    for r in read_csv("common_aihub_ko_review.csv")
}

result = []

for row in aihub:
    place_id = row["place_id"]
    lang = row["language"]
    attrs = parse_attributes(row["attributes"])

    for kind, field, ko, en, ja in FIELDS:
        key = {"ko": ko, "en": en, "ja": ja}[lang]
        value = attrs.get(key, "")
        from_en = from_ja = ""
        needs_review = False

        if lang == "ko" and place_id in variants:
            en_attrs, ja_attrs = variants[place_id]
            from_en = en_attrs.get(ko, "")
            from_ja = ja_attrs.get(ko, "")
            needs_review = normalize(from_en) != normalize(from_ja)
            value = "" if needs_review else from_en

        result.append({
            "place_id": place_id,
            "aihub_poi_id": links[place_id],
            "language": lang,
            "title": row["title"],
            "information_type": kind,
            "field": field,
            "raw_value": value,
            "status": (
                "REVIEW" if needs_review
                else "RAW_PRESENT" if value
                else "MISSING_FIELD"
            ),
            "ko_value_from_en": from_en,
            "ko_value_from_ja": from_ja,
        })

assert len(result) == len(aihub) * len(FIELDS)
assert len({(r["place_id"], r["language"], r["field"]) for r in result}) == len(result)

output = DATA / "common_aihub_attributes.csv"
with output.open("x", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result[0].keys())
    writer.writeheader()
    writer.writerows(result)

print("전체 정보 항목:", len(result))
print("상태별 개수:", Counter(r["status"] for r in result))