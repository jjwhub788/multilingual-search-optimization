
import pandas as pd
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"

INPUT = DATA_DIR / "tourapi_match_candidates.csv"

df = pd.read_csv(
    INPUT,
    dtype=str,
    keep_default_na=False
)

# 비교를 위해 공백·특수문자 제거
def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"[^가-힣0-9]", "", text)


# 외국어 관광지 이름의 괄호 속 한국어 추출
def get_korean_names(title):
    title = unicodedata.normalize("NFKC", str(title))
    parts = re.findall(r"\(([^()]*)\)", title)

    return [
        normalize(part)
        for part in parts
        if re.search(r"[가-힣]", part)
    ]


def check_match(korean_title, foreign_title):
    target = normalize(korean_title)
    return bool(target) and target in get_korean_names(foreign_title)


# 영어·일본어 이름 각각 검증
df["match_en"] = df.apply(
    lambda r: check_match(r["title_ko"], r["title_en"]),
    axis=1
)

df["match_ja"] = df.apply(
    lambda r: check_match(r["title_ko"], r["title_ja"]),
    axis=1
)

# 두 언어 모두 일치하는 경우
verified = df[
    df["match_en"] & df["match_ja"]
].copy()

# 추가 확인이 필요한 후보
review = df[
    ~(df["match_en"] & df["match_ja"])
].copy()

verified.to_csv(
    DATA_DIR / "tourapi_high_confidence.csv",
    index=False,
    encoding="utf-8-sig"
)

review.to_csv(
    DATA_DIR / "tourapi_manual_review.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 관광지 이름 매칭 검증 =====")
print("전체 후보:", len(df))
print("영어 이름 일치:", df["match_en"].sum())
print("일본어 이름 일치:", df["match_ja"].sum())
print("두 언어 모두 일치:", len(verified))
print("추가 검토 필요:", len(review))

print("\n===== 추가 검토 대상 예시 =====")
print(
    review[
        ["title_ko", "title_en", "title_ja"]
    ].head(10).to_string(index=False)
)
