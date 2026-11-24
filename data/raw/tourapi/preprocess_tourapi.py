
import pandas as pd
import re
import html
import unicodedata
from pathlib import Path

# ===== 경로 설정 =====
RAW_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RAW_DIR.parent.parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "ko": "tour_kr.csv",
    "en": "tour_en.csv",
    "ja": "tour_jp.csv",
}

# ===== 텍스트 정제 =====
def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


all_data = []

# ===== 언어별 처리 =====
for lang, filename in FILES.items():

    path = RAW_DIR / filename

    if not path.exists():
        print(f"[{lang}] 파일 없음: {filename}")
        continue

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False
    )

    before = len(df)

    # 필요한 컬럼 확보
    columns = [
        "contentid",
        "title",
        "addr1",
        "addr2",
        "mapx",
        "mapy",
        "overview",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = ""

    # 텍스트 전처리
    for col in ["contentid", "title", "addr1",
                "addr2", "overview"]:
        df[col] = df[col].apply(clean_text)

    # 관광지 ID 또는 이름이 없는 데이터 제거
    df = df[
        (df["contentid"] != "") &
        (df["title"] != "")
    ].copy()

    # 동일 언어 내 ID 중복 제거
    df = df.drop_duplicates(
        subset=["contentid"],
        keep="first"
    )

    # 언어 및 원본 ID 기록
    df["language"] = lang
    df["source_contentid"] = df["contentid"]

    # BGE-M3 검색에 사용할 텍스트
    df["search_text"] = (
        df["title"] + " " +
        df["addr1"] + " " +
        df["addr2"] + " " +
        df["overview"]
    ).apply(clean_text)

    final_columns = [
        "language",
        "source_contentid",
        "title",
        "addr1",
        "addr2",
        "mapx",
        "mapy",
        "overview",
        "search_text",
    ]

    df = df[final_columns]

    print(
        f"[{lang}] 원본 {before}건 → "
        f"전처리 후 {len(df)}건"
    )

    all_data.append(df)

# ===== 통합 데이터 저장 =====
if not all_data:
    raise RuntimeError(
        "CSV 파일을 찾지 못했습니다."
    )

combined = pd.concat(
    all_data,
    ignore_index=True
)

output_path = OUTPUT_DIR / "tourapi_clean.csv"

combined.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 전처리 완료 =====")
print("전체 데이터:", len(combined))
print("언어별 데이터 수:")
print(combined["language"].value_counts())
print("저장 위치:", output_path)
