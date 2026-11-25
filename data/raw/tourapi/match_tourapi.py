
import pandas as pd
from pathlib import Path

# ===== 파일 경로 =====
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"

INPUT = DATA_DIR / "tourapi_clean.csv"
OUTPUT = DATA_DIR / "tourapi_match_candidates.csv"

# ===== 데이터 불러오기 =====
df = pd.read_csv(
    INPUT,
    dtype=str,
    keep_default_na=False
)

print("전체 관광정보:", len(df))

# ===== 좌표 데이터 정리 =====
df["lon"] = pd.to_numeric(df["mapx"], errors="coerce")
df["lat"] = pd.to_numeric(df["mapy"], errors="coerce")

df = df.dropna(subset=["lon", "lat"]).copy()

# 한국 지역의 정상적인 좌표만 사용
df = df[
    df["lon"].between(124, 132)
    & df["lat"].between(33, 39.5)
].copy()

print("유효 좌표 데이터:", len(df))

# ===== 동일 좌표 후보 생성 =====
df["coord"] = (
    df["lon"].round(5).astype(str)
    + "_"
    + df["lat"].round(5).astype(str)
)

counts = (
    df.groupby(["coord", "language"])
    .size()
    .unstack(fill_value=0)
)

for lang in ["ko", "en", "ja"]:
    if lang not in counts.columns:
        counts[lang] = 0

# 세 언어가 모두 존재하는 좌표
common = counts[
    (counts["ko"] > 0)
    & (counts["en"] > 0)
    & (counts["ja"] > 0)
]

print("3개 언어 공통 좌표:", len(common))

# 같은 좌표에 언어별 관광지가 1개씩인 경우
unique = common[
    (common["ko"] == 1)
    & (common["en"] == 1)
    & (common["ja"] == 1)
]

print("일대일 매칭 후보:", len(unique))

# ===== 언어별 데이터 분리 =====
matched = df[df["coord"].isin(unique.index)].copy()

def select_language(lang):
    subset = matched[matched["language"] == lang]
    subset = subset[
        ["coord", "source_contentid", "title"]
    ]

    return subset.rename(columns={
        "source_contentid": f"id_{lang}",
        "title": f"title_{lang}"
    })

ko = select_language("ko")
en = select_language("en")
ja = select_language("ja")

# ===== 세 언어 결합 =====
result = (
    ko.merge(en, on="coord")
      .merge(ja, on="coord")
)

# ===== 저장 =====
result.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 매칭 결과 =====")
print("매칭 후보 관광지:", len(result))
print("저장 위치:", OUTPUT)

print("\n===== 관광지 이름 비교 =====")
print(
    result[
        ["title_ko", "title_en", "title_ja"]
    ].head(10).to_string(index=False)
)
