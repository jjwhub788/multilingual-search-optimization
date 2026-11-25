import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"

# TourAPI 설명에서 먼저 읽어볼 부분을 찾는 단서.
# 단서가 있다는 것만으로 해당 정보를 제공한다고 확정하지 않는다.
CUES = {
    "experience": {
        "ko": r"체험|프로그램|강습|워크숍|템플\s*스테이",
        "en": r"\b(?:experiences?|program(?:me)?s?|workshops?|lessons?|hands-on|templestay)\b",
        "ja": r"体験|プログラム|ワークショップ|レッスン|テンプルステイ",
    },
    "opening_hours": {
        "ko": r"(?:운영|이용|개방|관람)\s*시간|상시\s*개방|24\s*시간",
        "en": r"\b(?:opening|operating|business) hours\b|\bopen (?:from|daily|24 hours)\b",
        "ja": r"営業時間|利用時間|開館時間|常時開放|24\s*時間",
    },
    "holidays": {
        "ko": r"휴무|휴관|연중무휴|매주\s*[월화수목금토일]요일|공휴일",
        "en": r"\bclosed\b|\bholidays?\b|\byear[- ]round\b|\bopen (?:daily|every day)\b",
        "ja": r"休業|休館|定休日|年中無休|毎週[月火水木金土日]曜|祝日",
    },
    "admission_fee": {
        "ko": r"입장료|이용료|관람료|무료\s*입장|입장\s*무료|[\d,]+\s*원",
        "en": r"\badmission\b|\b(?:entrance|entry) fees?\b|\bfree of charge\b|\bKRW\b",
        "ja": r"入場料|利用料|観覧料|入場無料|[\d,]+\s*ウォン",
    },
    "parking": {
        "ko": r"주차",
        "en": r"\bparking\b|\bcar parks?\b",
        "ja": r"駐車",
    },
}

TIME = r"(?<!\d)(?:[01]?\d|2[0-4]):[0-5]\d(?!\d)"
UNSPECIFIED = {
    "없음", "전화문의", "상황에 따라 다름",
    "none", "please call", "telephone inquiries",
    "depends on the situation", "subject to change",
    "なし", "無し", "電話での問い合わせ", "電話でのお問い合わせ",
    "状況により異なる", "状況によって異なる",
}


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


tour = {
    (r["place_id"], r["language"]): r
    for r in read_csv("common_dataset.csv")
    if r["source"] == "tourapi"
}

result = []

for attr in read_csv("common_aihub_attributes.csv"):
    field = attr["field"]
    if field not in CUES:
        continue

    lang = attr["language"]
    doc = tour[(attr["place_id"], lang)]
    desc = doc["description"]
    text = unicodedata.normalize("NFKC", desc)

    pattern = CUES[field][lang]
    if field == "opening_hours":
        pattern = f"{pattern}|{TIME}"

    hit = re.search(pattern, text, flags=re.IGNORECASE)
    hint = text[max(0, hit.start() - 55):hit.end() + 75] if hit else ""

    raw = attr["raw_value"]
    normalized = " ".join(
        unicodedata.normalize("NFKC", raw).casefold().split()
    ).rstrip(".。")

    value_status = attr["status"]
    if value_status == "RAW_PRESENT":
        value_status = (
            "AMBIGUOUS_VALUE" if normalized in UNSPECIFIED
            else "VALUE_CANDIDATE"
        )

    result.append({
        "place_id": attr["place_id"],
        "language": lang,
        "tourapi_title": doc["title"],
        "information_type": attr["information_type"],
        "field": field,
        "tourapi_description": desc,
        "tourapi_evidence_hint": hint,
        "tourapi_scan_status": "CUE_FOUND" if hit else "NO_CUE",
        "aihub_raw_value": raw,
        "aihub_value_status": value_status,
        "ko_value_from_en": attr["ko_value_from_en"],
        "ko_value_from_ja": attr["ko_value_from_ja"],
        "tourapi_final_status": "",
        "review_note": "",
    })

assert len(result) == 2070

output = DATA / "common_gap_review.csv"
with output.open("x", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result[0].keys())
    writer.writeheader()
    writer.writerows(result)

print("전체 비교 항목:", len(result))
print("TourAPI 단서:", Counter(r["tourapi_scan_status"] for r in result))
print("AI Hub 값 상태:", Counter(r["aihub_value_status"] for r in result))