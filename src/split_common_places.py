import csv
import random
from collections import Counter
from pathlib import Path

# 프로젝트의 기본 경로
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

# 읽어올 파일
LINKS_FILE = DATA / "common_poi_links.csv"
ITEMS_FILE = DATA / "common_enrichment_item_pool.csv"

# 새로 만들 파일
OUTPUT_FILE = DATA / "common_place_splits.csv"

# 나중에 다시 실행해도 같은 결과가 나오도록 고정
RANDOM_SEED = 42

# 개발용 70%, 최종 평가용 약 30%
DEV_RATIO = 0.7


def read_csv(path):
    """CSV 파일을 읽어서 행 목록으로 반환합니다."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


# 1. 관광지 목록과 보완 가능 정보 읽기
links = read_csv(LINKS_FILE)
items = read_csv(ITEMS_FILE)

# 2. 공통 관광지 ID 확인
place_ids = [row["place_id"] for row in links]

if len(place_ids) != len(set(place_ids)):
    raise ValueError("관광지 ID가 중복되어 있어요. 원본 파일을 확인해 주세요.")

# 3. 관광지 순서를 섞기
# 정렬 후 섞어야 원본 파일 순서가 달라도 같은 결과를 얻을 수 있어요.
place_ids = sorted(place_ids)

random_generator = random.Random(RANDOM_SEED)
random_generator.shuffle(place_ids)

# 4. 관광지 단위로 개발용과 최종 평가용 나누기
dev_count = int(len(place_ids) * DEV_RATIO)

dev_ids = set(place_ids[:dev_count])
test_ids = set(place_ids[dev_count:])

if dev_ids & test_ids:
    raise ValueError("개발용과 최종 평가용에 같은 관광지가 들어 있어요.")

# 관광지 이름을 ID로 찾기 위한 표
titles = {
    row["place_id"]: row["title_ko"]
    for row in links
}

# 5. 결과 파일을 새로 저장
# 'x' 모드는 기존 파일이 있으면 덮어쓰지 않고 멈춰요.
with OUTPUT_FILE.open("x", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["place_id", "title_ko", "split"]
    )
    writer.writeheader()

    for place_id in sorted(place_ids):
        split = "dev" if place_id in dev_ids else "test"

        writer.writerow({
            "place_id": place_id,
            "title_ko": titles[place_id],
            "split": split,
        })

# 6. 각 그룹에 어떤 보완 정보가 있는지 확인
distribution = Counter()
unknown_ids = set()

for item in items:
    place_id = item["place_id"]

    if place_id in dev_ids:
        split = "dev"
    elif place_id in test_ids:
        split = "test"
    else:
        unknown_ids.add(place_id)
        continue

    distribution[(split, item["field"])] += 1

# 7. 결과 출력
print("관광지 분리 완료!")
print(f"전체 관광지: {len(place_ids)}곳")
print(f"개발용 관광지: {len(dev_ids)}곳")
print(f"최종 평가용 관광지: {len(test_ids)}곳")

print("\n그룹별 보완 가능 정보 건수:")
for (split, field), count in sorted(distribution.items()):
    print(f"  {split} / {field}: {count}건")

print(f"\n관광지 목록에 없는 보완 정보 ID: {len(unknown_ids)}개")
print(f"저장 위치: {OUTPUT_FILE}")