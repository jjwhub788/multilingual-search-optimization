
import pandas as pd
from pathlib import Path

input_path = Path("data/processed/aihub_clean.csv")
output_path = Path("data/processed/aihub_poi.csv")

df = pd.read_csv(
    input_path,
    dtype={"poi_id": str},
    keep_default_na=False
)

# 같은 POI와 언어 조합이 중복되지 않는지 확인
assert not df.duplicated(["poi_id", "language"]).any()

# 한 관광지를 한 행으로 변환
fields = [
    "source_split", "title_ko", "title",
    "address_ko", "address", "attributes"
]

wide = df.pivot(
    index="poi_id",
    columns="language",
    values=fields
)

wide.columns = [
    f"{field}_{lang}" for field, lang in wide.columns
]

wide = wide.reset_index().fillna("")

# 언어별 데이터 존재 여부
wide["has_en"] = wide["source_split_en"] != ""
wide["has_ja"] = wide["source_split_ja"] != ""

# 양쪽 언어의 한국어 명칭이 다른 관광지 표시
wide["name_mismatch"] = (
    wide["has_en"]
    & wide["has_ja"]
    & (wide["title_ko_en"] != wide["title_ko_ja"])
)

# 원래 Training/Validation이 교차된 관광지 표시
wide["cross_split"] = (
    wide["has_en"]
    & wide["has_ja"]
    & (wide["source_split_en"] != wide["source_split_ja"])
)

wide.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("전체 고유 POI:", len(wide))
print("영어·일본어 모두 존재:",
      (wide["has_en"] & wide["has_ja"]).sum())
print("영어에만 존재:",
      (wide["has_en"] & ~wide["has_ja"]).sum())
print("일본어에만 존재:",
      (~wide["has_en"] & wide["has_ja"]).sum())
print("한국어 이름 불일치:",
      wide["name_mismatch"].sum())
print("언어 간 Training/Validation 교차:",
      wide["cross_split"].sum())
print("저장 완료:", output_path)
