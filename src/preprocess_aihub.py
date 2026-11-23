
import csv
import json
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("data/raw/aihub/AI_Hub/01-1.정식개방데이터")
OUTPUT = Path("data/processed/aihub_clean.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

def clean(value):
    if value is None:
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()

columns = [
    "source", "source_split", "poi_id", "language",
    "title_ko", "title", "address_ko", "address",
    "attributes_ko", "attributes"
]

file_counts = Counter()
field_counts = Counter()
missing_names = Counter()
failed_files = 0

# 같은 POI의 한국어 관광지명 비교
korean_names = defaultdict(dict)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()

    for split in ["Training", "Validation"]:
        label_dir = ROOT / split / "02.라벨링데이터"

        for folder in label_dir.iterdir():
            folder_name = clean(folder.name)

            if "관광지" not in folder_name:
                continue

            if "영어" in folder_name:
                lang = "en"
            elif "일본어" in folder_name:
                lang = "ja"
            else:
                continue

            for path in folder.glob("*.json"):
                try:
                    with path.open(encoding="utf-8") as f:
                        obj = json.load(f)

                    poi_id = clean(obj.get("data", {}).get("POI_id"))
                    if not poi_id:
                        failed_files += 1
                        continue

                    annotations = obj.get("annotations", [])
                    fields = {
                        clean(a.get("k_column")): a
                        for a in annotations
                        if isinstance(a, dict)
                    }

                    field_counts.update(fields.keys())

                    name = fields.get("관광지명", {})
                    address = {}

                    for key in ["주소(도로명)", "주소(지번주소)", "주소"]:
                        item = fields.get(key, {})

                        if clean(item.get("k_context")) and clean(item.get("t_context")):
                            address = item
                            break


                    title_ko = clean(name.get("k_context"))
                    title = clean(name.get("t_context"))

                    if not title_ko or not title:
                        missing_names[lang] += 1

                    korean_names[poi_id][lang] = title_ko

                    # 이름과 주소 외의 항목을 보존
                    excluded = {"관광지명", "주소(도로명)"}
                    attrs_ko = []
                    attrs_translated = []

                    for key, item in fields.items():
                        if key in excluded:
                            continue

                        ko_value = clean(item.get("k_context"))
                        tr_value = clean(item.get("t_context"))
                        tr_label = clean(item.get("t_column")) or key

                        if ko_value:
                            attrs_ko.append(f"{key}: {ko_value}")
                        if tr_value:
                            attrs_translated.append(
                                f"{tr_label}: {tr_value}"
                            )

                    writer.writerow({
                        "source": "aihub",
                        "source_split": split,
                        "poi_id": poi_id,
                        "language": lang,
                        "title_ko": title_ko,
                        "title": title,
                        "address_ko": clean(address.get("k_context")),
                        "address": clean(address.get("t_context")),
                        "attributes_ko": " | ".join(attrs_ko),
                        "attributes": " | ".join(attrs_translated)
                    })

                    file_counts[(split, lang)] += 1

                except (OSError, ValueError, TypeError):
                    failed_files += 1

print("\n===== 추출 결과 =====")
for key, count in sorted(file_counts.items()):
    print(key, count)

print("총 CSV 행 수:", sum(file_counts.values()))
print("읽기 실패 또는 ID 누락:", failed_files)
print("관광지명 누락:", dict(missing_names))

common = [
    names for names in korean_names.values()
    if "en" in names and "ja" in names
]

mismatches = sum(
    names["en"] != names["ja"]
    for names in common
)

print("\n===== 한국어 관광지명 일치 검사 =====")
print("영어·일본어 공통 POI:", len(common))
print("한국어 관광지명 불일치:", mismatches)

print("\n===== 주요 필드별 등장 횟수 =====")
for field, count in field_counts.most_common(20):
    print(field, count)

print("\n저장 완료:", OUTPUT)
