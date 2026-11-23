
import pandas as pd
from pathlib import Path

folder = Path("data/processed")

df = pd.read_csv(
    folder / "aihub_clean.csv",
    dtype={"poi_id": str},
    keep_default_na=False
)

# 1. 관광지명이 누락된 데이터
missing_names = df[
    (df["title_ko"].str.strip() == "") |
    (df["title"].str.strip() == "")
].copy()

missing_names.to_csv(
    folder / "aihub_missing_names.csv",
    index=False,
    encoding="utf-8-sig"
)

# 2. 주소 누락 확인
missing_address = df[
    (df["address_ko"].str.strip() == "") |
    (df["address"].str.strip() == "")
]

# 3. 같은 POI의 한국어 이름 불일치
names = df.pivot(
    index="poi_id",
    columns="language",
    values="title_ko"
)

mismatches = names[
    names["en"].notna() &
    names["ja"].notna() &
    (names["en"] != names["ja"])
].copy()

mismatches.to_csv(
    folder / "aihub_name_mismatches.csv",
    encoding="utf-8-sig"
)

print("===== 데이터 품질 점검 =====")
print("전체 행 수:", len(df))
print("이름 누락 행 수:", len(missing_names))
print("주소 누락 행 수:", len(missing_address))
print("한국어 이름 불일치 POI 수:", len(mismatches))

print("\n검수 파일 저장 완료")
