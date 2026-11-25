"""Apply AI-assisted identity-review judgments to a separate CSV copy.

The specific exceptions below are AI-assisted assessments of the supplied
matched records, not human verification or independently checked addresses.
Existing reviewer decisions and notes are preserved unchanged.
"""

import csv
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "joint_poi_scan"
INPUT = FOLDER / "new_trilingual_review_checked.csv"
OUTPUT = FOLDER / "new_trilingual_review_ai_completed.csv"

# Korean TourAPI IDs, rather than spreadsheet row numbers, anchor the judgments.
REVIEW_REASONS = {
    "2024432": ("가우도", "TourAPI 영어 주소 월곶로 473과 한국어·일본어·AI Hub 주소 신기리 산31-2의 동일 장소 여부 확인 필요"),
    "2754982": ("능포항", "TourAPI 영·일 주소가 능포동까지만 기재됨; 한국어·AI Hub 지번 480-42와의 동일 위치 확인 필요"),
    "2563697": ("지산유원지", "TourAPI 영어 지호로164번길 35-1과 한국어·일본어·AI Hub 지호로164번길 23 불일치; 같은 관광지 내 시설인지 확인 필요"),
    "2617724": ("마린시티", "TourAPI 영·일 주소가 우동까지만 기재·한국어와 AI Hub는 우동 1411; 넓은 지역 내 동일 대상 확인 필요"),
    "2661528": ("조천연꽃공원", "AI Hub 영어 주소 비로봉길 94와 한국어·일본어 주소 번암리 226의 대응 여부 확인 필요"),
    "2715684": ("내장산 단풍생태공원", "한국어·AI Hub 내장동 364-2, TourAPI 일본어 내장동 560, 영어 내장호반로 273-17로 상세 주소 불일치"),
    "2661736": ("행주산성 역사공원", "TourAPI 영어 주소가 행주외동까지만 기재됨; 한국어·일본어·AI Hub 지번 140-8과의 대응 확인 필요"),
    "126516": ("보신각터", "보신각터와 영·일 제목의 보신각 종각이 동일 관광지 범위를 가리키는지 확인 필요"),
    "585605": ("금진해변", "TourAPI 한국어·AI Hub 헌화로 271과 영·일 금진리 176-2가 동일 장소인지 확인 필요"),
    "125800": ("강릉 선교장", "AI Hub 일본어 명칭 江陵宣教場과 TourAPI 江陵船橋荘의 의미 차이; 고유명 번역 오류 여부 확인 필요"),
    "2675091": ("원당샘공원", "AI Hub 주소에 원당경로당이 포함됨; 공원과 경로당의 동일 위치 및 대상 범위 확인 필요"),
    "250398": ("정취암", "AI Hub 일본어 情趣岩과 TourAPI 浄趣庵의 명칭 의미 차이; AI Hub 한 원문의 주차·비용·체험 값도 언어별로 불일치"),
    "129000": ("추암 촛대바위", "TourAPI 영·일 주소에 한국어·AI Hub 지번 산69가 누락되어 동일 세부 위치 확인 필요"),
    "128637": ("경주 감은사지", "TourAPI 일본어 양북면과 한국어·AI Hub 문무대왕면 표기 차이 및 영어 지번 17 누락; 동일 주소 여부 확인 필요"),
    "128961": ("강변스파랜드", "TourAPI 영·일과 AI Hub 한국어·일본어는 지하2층, AI Hub 영어는 2 G/F로 층 표기 불확실; 번역 오류 여부 확인 필요"),
    "126429": ("광주 예술의 거리", "TourAPI 영어 예술길 24와 한국어·일본어·AI Hub 궁동 78의 도로명·지번 대응 확인 필요"),
    "2661441": ("등기산스카이워크", "TourAPI 영어는 지번 141-21, 한국어·일본어·AI Hub는 산141-21로 산 지번 표기 누락 여부 확인 필요"),
    "128670": ("강주연못", "TourAPI 영·일 주소가 예하리까지만 기재됨; 한국어·AI Hub 지번 911-11 및 대표 위치 확인 필요"),
    "129550": ("장자호수공원", "한국어·AI Hub 교문동 537과 TourAPI 영·일 토평동 878 불일치; 공원 내 동일 대상 여부 확인 필요"),
    "129637": ("대천항", "한국어·AI Hub 신흑동 1250과 TourAPI 영·일 신흑동 2245의 지번 불일치; 동일 항구 내 위치인지 확인 필요"),
    "988449": ("월영교", "TourAPI 영·일 주소가 상아동까지만 기재됨; 한국어·AI Hub 지번 569 및 대표 위치 확인 필요"),
    "2569511": ("감악산 출렁다리", "TourAPI 일본어 설마천로 273-9와 한국어·영어·AI Hub 설마천로 238 불일치; 동일 시설인지 확인 필요"),
    "2769599": ("대청댐 전망대", "AI Hub 주소에 현암정휴게소가 포함되고 좌표 차이가 큼; 전망대와 휴게소의 동일 관광지 범위 확인 필요"),
    "2751854": ("도치돌 알파카목장", "TourAPI 일본어 도치돌길 293과 한국어·영어·AI Hub 도치돌길 303 불일치; 동일 시설인지 확인 필요"),
}


def normalize_name(value):
    return "".join(unicodedata.normalize("NFKC", value).split())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def main():
    fields, rows = read_csv(INPUT)
    assert len(rows) == 107 and fields[33:35] == ["team_final_decision", "team_review_note"]
    ids = {r["tourapi_ko_id"]: r for r in rows}
    assert len(ids) == len(rows)
    for key, (title, _) in REVIEW_REASONS.items():
        assert ids[key]["tourapi_title_ko"] == title, (key, title)

    for row in rows:
        if row["team_final_decision"].strip():
            assert row["team_review_note"].strip()
            continue

        key = row["tourapi_ko_id"]
        distance = float(row["max_distance_m"])
        if key in REVIEW_REASONS:
            row["team_final_decision"] = "REVIEW"
            row["team_review_note"] = (
                f"[AI 보조검토] {REVIEW_REASONS[key][1]}. 좌표 최대 차이 {distance:g}m."
            )
            continue

        names = [row[f] for f in ("tourapi_title_ko", "aihub_title_ko_en", "aihub_title_ko_ja")]
        assert all(names) and len({normalize_name(n) for n in names}) == 1
        assert row["name_evidence"] == "EXACT"
        assert row["address_evidence"] == "BOTH_NUMERIC_CORES"
        assert row["tourapi_trilingual_status"] == "NEW_ALIAS_COORD_REVIEW"
        assert row["aihub_two_translated_names"] == row["aihub_two_translated_attributes"] == "YES"
        assert 0 <= distance <= 100
        name_note = "이름 완전 일치" if len(set(names)) == 1 else "이름 띄어쓰기 정규화 후 일치"
        row["team_final_decision"] = "MATCH"
        row["team_review_note"] = (
            f"[AI 보조검토] {name_note}, 양 출처 상세 주소 정규화 일치, "
            f"TourAPI 영·일 한국어 별칭 일치 및 주소 상충 없음. 좌표 최대 차이 {distance:g}m."
        )
        if key == "129683":
            row["team_review_note"] += " 정보 보완 시 AI Hub 입장료 원문(전화문의/없음) 불일치 별도 확인 필요."

    # Never overwrite the input or an already reviewed output copy.
    with OUTPUT.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("판정 결과:", Counter(r["team_final_decision"] for r in rows))
    print("AI 보조검토:", Counter(r["team_final_decision"] for r in rows if r["team_review_note"].startswith("[AI 보조검토]")))
    print("추가 검토 행:", [i for i, r in enumerate(rows, 2) if r["team_final_decision"] == "REVIEW"])


if __name__ == "__main__":
    main()
