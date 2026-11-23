
import json
import unicodedata
from pathlib import Path
from collections import Counter

ROOT = Path("data/raw/aihub/AI_Hub/01-1.정식개방데이터")

# 각 데이터 구분별 관광지 ID 저장
poi_counts = {}
place_names = {}

for split in ["Training", "Validation"]:
    label_dir = ROOT / split / "02.라벨링데이터"

    for folder in label_dir.iterdir():
        folder_name = unicodedata.normalize("NFC", folder.name)

        if "관광지" not in folder_name:
            continue

        if "영어" in folder_name:
            lang = "en"
        elif "일본어" in folder_name:
            lang = "ja"
        else:
            continue

        key = (split, lang)
        counts = Counter()
        names = {}
        errors = 0

        for path in folder.glob("*.json"):
            try:
                with path.open(encoding="utf-8") as f:
                    obj = json.load(f)
            except (OSError, ValueError):
                errors += 1
                continue

            poi_id = str(obj.get("data", {}).get("POI_id", "")).strip()
            if not poi_id:
                errors += 1
                continue

            counts[poi_id] += 1

            for item in obj.get("annotations", []):
                if item.get("k_column") == "관광지명":
                    names[poi_id] = item.get("k_context", "")
                    break

        poi_counts[key] = counts
        place_names[key] = names

        print(f"\n[{split} / {lang}]")
        print("JSON 파일 수:", sum(counts.values()))
        print("고유 관광지 수:", len(counts))
        print("중복 ID에 해당하는 추가 파일 수:",
              sum(n - 1 for n in counts.values()))
        print("읽기 실패 또는 ID 누락:", errors)

# 같은 split 내부의 영어·일본어 매칭
for split in ["Training", "Validation"]:
    en_ids = set(poi_counts.get((split, "en"), {}))
    ja_ids = set(poi_counts.get((split, "ja"), {}))
    common = en_ids & ja_ids

    print(f"\n===== {split} 언어 매칭 =====")
    print("영어·일본어 공통 POI:", len(common))
    print("영어에만 존재:", len(en_ids - ja_ids))
    print("일본어에만 존재:", len(ja_ids - en_ids))

    for poi_id in sorted(common)[:3]:
        print("매칭 예시:", poi_id,
              place_names[(split, "en")].get(poi_id),
              "|", place_names[(split, "ja")].get(poi_id))

# Training/Validation 사이의 관광지 중복 확인
train_ids = set().union(
    *[set(poi_counts.get(("Training", l), {})) for l in ["en", "ja"]]
)
valid_ids = set().union(
    *[set(poi_counts.get(("Validation", l), {})) for l in ["en", "ja"]]
)

print("\nTraining/Validation 공통 POI:", len(train_ids & valid_ids))


# Training과 Validation을 교차해서 비교

en_train = set(poi_counts.get(("Training", "en"), {}))
en_valid = set(poi_counts.get(("Validation", "en"), {}))
ja_train = set(poi_counts.get(("Training", "ja"), {}))
ja_valid = set(poi_counts.get(("Validation", "ja"), {}))

print("\n===== 언어별 교차 매칭 =====")
print("영어 Train ↔ 일본어 Train:", len(en_train & ja_train))
print("영어 Train ↔ 일본어 Valid:", len(en_train & ja_valid))
print("영어 Valid ↔ 일본어 Train:", len(en_valid & ja_train))
print("영어 Valid ↔ 일본어 Valid:", len(en_valid & ja_valid))

all_en = en_train | en_valid
all_ja = ja_train | ja_valid

print("\n===== 전체 데이터 기준 =====")
print("영어 전체 고유 POI:", len(all_en))
print("일본어 전체 고유 POI:", len(all_ja))
print("한·영·일 구성 가능한 POI 후보:", len(all_en & all_ja))
print("영어에만 존재:", len(all_en - all_ja))
print("일본어에만 존재:", len(all_ja - all_en))

print("\n===== 데이터 분할 중복 =====")
print("영어 Train/Valid 공통 POI:", len(en_train & en_valid))
print("일본어 Train/Valid 공통 POI:", len(ja_train & ja_valid))
