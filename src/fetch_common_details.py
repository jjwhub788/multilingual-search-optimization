import sys
import os
import json
import time
import getpass
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/raw/tourapi"))

import collect_matched_details as collector

OUTPUT = ROOT / "data/processed/common_tourapi_details.csv"
CACHE = ROOT / "data/raw/tourapi/common_detail_overviews.jsonl"

rows = collector.read_csv(OUTPUT)

index = {
    (r["language"], r["source_contentid"]): r
    for r in rows
}

# 이전 수집 결과 복원
if CACHE.exists():
    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                key = (item["language"], item["source_contentid"])

                if key in index:
                    index[key]["description"] = item["overview"]
                    index[key]["collection_status"] = (
                        "COMPLETE" if item["overview"] else "EMPTY"
                    )
            except (ValueError, KeyError):
                continue

# API 인증키 입력
default_key = (
    os.environ.get("TOURAPI_KEY")
    or getpass.getpass("국문 API 인증키: ").strip()
)

keys = {
    "ko": default_key,
    "en": getpass.getpass("영문 API 인증키 (같으면 Enter): ").strip()
          or default_key,
    "ja": getpass.getpass("일문 API 인증키 (같으면 Enter): ").strip()
          or default_key
}

try:
    with CACHE.open("a", encoding="utf-8") as f:
        for lang in ["ko", "en", "ja"]:

            targets = [
                r for r in rows
                if r["language"] == lang
                and r["collection_status"] == "PENDING"
            ]

            for i, r in enumerate(targets, 1):
                cid = r["source_contentid"]

                try:
                    overview = collector.fetch_with_retry(
                        lang, cid, keys[lang]
                    )
                except Exception as e:
                    print(
                        f"[{lang}] 수집 중단:",
                        collector.error_description(e)
                    )
                    break

                r["description"] = overview
                r["collection_status"] = (
                    "COMPLETE" if overview else "EMPTY"
                )

                f.write(json.dumps({
                    "language": lang,
                    "source_contentid": cid,
                    "overview": overview
                }, ensure_ascii=False) + "\n")
                f.flush()

                if i % 10 == 0 or i == len(targets):
                    print(f"[{lang}] {i}/{len(targets)} 처리")

                time.sleep(1.2)

finally:
    collector.write_csv(OUTPUT, rows)
    print("최종 상태:", Counter(
        r["collection_status"] for r in rows
    ))