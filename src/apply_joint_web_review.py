"""Attach external web evidence to the 24 unresolved trilingual candidates.

The evidence was reviewed with AI assistance on 2026-11-24 (KST).
Entity identity and the correctness of every translated address or attribute
are deliberately treated as distinct questions. No source field is edited.
"""

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "joint_poi_scan"
INPUT = FOLDER / "new_trilingual_review_ai_completed.csv"
OUTPUT = FOLDER / "new_trilingual_review_web_verified.csv"


def vk(lang, no):
    return f"https://{lang}.visitkorea.or.kr/svc/contents/contentsView.do?vcontsId={no}"


# These are individually examined external sources, not a fabricated manual check.
# Each key is the Korean TourAPI content ID, rather than a mutable row number.
EVIDENCE = {
    "2024432": (
        "가우도", "MATCH",
        "관광공사 영어 가우도 안내에 월곶로 473과 신기리 산31-2를 같은 관광지의 주소로 병기함. 정책브리핑도 가우도 방문 주소로 월곶로 473을 안내함.",
        ["https://english.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId=66464", "https://www.korea.kr/news/policyNewsView.do?newsId=148926714"],
    ),
    "2754982": (
        "능포항", "MATCH",
        "관광공사 영어 페이지가 능포항에 대해 능포동과 한국어 상세 지번 480-42를 같이 안내함. 영어 상세주소 생략이며 대상 일치.",
        [vk("english", 220301)],
    ),
    "2563697": (
        "지산유원지", "MATCH",
        "관광공사 지산유원지 영어 페이지에 지호로164번길 35-1과 한국어 23이 함께 기재되고, 광주 공식 관광 안내도 해당 유원지의 35-1 주소를 확인함. 관광지 단위 일치, 건물번호 자체의 차이는 유지.",
        [vk("english", 14831), "https://tour.gwangju.go.kr/home/tour/info/sports.cs?act=view&infoId=29"],
    ),
    "2617724": (
        "마린시티", "MATCH",
        "관광공사 영어 마린시티 안내는 우동으로, 병기된 한국어 주소는 우동 1411로 표시함. 열린관광 공식 정보도 동일 지번을 확인함.",
        ["https://english.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId=60310", "https://access.visitkorea.or.kr/ms/detail.do?cotId=2b0f1cfb-f282-404c-8fa3-02eadf747162"],
    ),
    "2661528": (
        "조천연꽃공원", "MATCH",
        "관광공사 영·일 페이지와 AI Hub 한국어 원문이 번암리 226 및 같은 공원명을 확인함. 세종시 제공 2023년 길관광 표준데이터 재게시본은 조천연꽃공원 주차장에 비로봉길 94와 번암리 226을 함께 기록함. 주소 검색상 94의 지번은 226-4로 나오는 별도 접근 지점이므로 정확히 같은 필지·좌표라고 간주하지 않음. 동일 공원 단위 일치, 영문 주소 품질은 별도 검토.",
        [vk("english", 191175), "https://flt.purpleo.co.kr/view/870", "https://www.data.go.kr/data/15017321/standard.do", "https://findby.co.kr/details/30036-361104574222-st-652b06a38c641f9e3886ab3b"],
    ),
    "2715684": (
        "내장산 단풍생태공원", "MATCH",
        "관광공사 영문 페이지는 내장호반로 273-17 및 내장동 364-2를, 일본어 페이지는 내장동 560을 동일 공원 명칭과 설명으로 안내함. 열린관광에도 560이 등록되어 있어 공원 단위 일치로 판정하고 지번 차이는 유지함.",
        [vk("english", 220426), vk("japanese", 229986), "https://access.visitkorea.or.kr/ms/detail.do?cotId=8e9c5b4b-b933-4642-b391-03d5f31c1133"],
    ),
    "2661736": (
        "행주산성 역사공원", "MATCH",
        "관광공사 영문 역사공원 안내에 행주외동과 한국어 지번 140-8이 함께 기재됨. 영어 주소의 지번 생략이며 대상 일치.",
        ["https://english.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId=188570"],
    ),
    "126516": (
        "보신각터", "MATCH",
        "관광공사 공식 표제에서 Bosingak Belfry와 보신각 터를 같은 관광지로 병기하고 종로 54를 안내함. 역사적 터 및 현재 종각을 동일 관광지로 안내하는 관계를 확인함.",
        ["https://english.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId=111240"],
    ),
    "585605": (
        "금진해변", "MATCH",
        "관광공사 영·일 금진해변 페이지가 금진리 176-2와 한국어 헌화로 271을 같은 해변 주소로 병기함.",
        [vk("english", 188432), vk("japanese", 189747)],
    ),
    "125800": (
        "강릉 선교장", "MATCH",
        "관광공사 일본어 공식 표제 江陵船橋荘 및 선교장 공식 홈페이지의 운정길 63 주소, AI Hub 한국어 원문 및 전화번호(033-648-5303)가 일치함. AI Hub 일본어 宣教場은 고유명 번역 오류로 판단하며 번역 자체는 수정하지 않음.",
        ["https://japanese.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId=107749", "https://www.knsgj.net/", "https://www.kh.or.kr/visit/kor/road/2/view.do?key=2408220002"],
    ),
    "2675091": (
        "원당샘공원", "MATCH",
        "도봉구 문화관광 공식 원당샘공원 안내의 전화번호(02-2091-3765)가 AI Hub와 일치. 도봉구 비상급수시설 명단에서도 원당샘터공원을 해등로32가길 16(방학동, 원당경로당)으로 기재하여 주소 내 시설명 차이를 설명함.",
        ["https://tour.dobong.go.kr/Contents.asp?code=10003486", "https://www.dobong.go.kr/Contents.asp?code=10009539", "https://english.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId=190240"],
    ),
    "250398": (
        "정취암", "MATCH",
        "관광공사 영문 정취암 안내의 둔철산로 675-87 및 전화번호(055-972-3339)가 AI Hub와 일치하고 사찰임을 확인함. 일본어 情趣岩 표기는 명칭 번역 문제이며 AI Hub 주차·비용·체험 원문 불일치는 정보 보완 단계에서 별도 검증 필요.",
        ["https://english.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId=188775"],
    ),
    "129000": (
        "추암 촛대바위", "MATCH",
        "관광공사 영·일 촛대바위 페이지가 상세 한국어 주소 추암동 산69와 축약 외국어 주소를 같은 바위 명칭으로 안내함.",
        [vk("english", 94231), vk("japanese", 94013)],
    ),
    "128637": (
        "경주 감은사지", "MATCH",
        "관광공사 일본어 감은사지 페이지가 양북면 용당리 17과 한국어 문무대왕면 용당리 17을 병기함. 경주시 공식 시정자료에서 양북면이 2021년 4월 1일 문무대왕면으로 변경된 사실을 확인함.",
        ["https://japanese.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId=109158", "https://www.gyeongju.go.kr/news/page.do?mnu_uid=1336&parm_bod_uid=217447&step=258"],
    ),
    "128961": (
        "강변스파랜드", "MATCH",
        "서울 공식 관광정보 및 관광공사 영문 강변스파랜드 안내 모두 구의강변로 45 지하2층을 확인함. AI Hub 영어 2 G/F는 층 표기 번역 오류로 판단하고 시설명·건물주소가 일치해 동일 시설로 판정함.",
        ["https://korean.visitseoul.net/entertainment/%EA%B0%95%EB%B3%80-%EC%8A%A4%ED%8C%8C%EB%9E%9C%EB%93%9C-KR/KOP009576", vk("english", 96745)],
    ),
    "126429": (
        "광주 예술의 거리", "MATCH",
        "관광공사 영문 광주 예술의 거리 페이지가 예술길 24와 한국어 궁동 78을 같은 관광거리 주소로 병기함.",
        [vk("english", 111557)],
    ),
    "2661441": (
        "등기산스카이워크", "MATCH",
        "관광공사 영문 페이지가 후포리 141-21 및 한국어 산141-21을 한 관광지에 병기하고 일본어 공식 안내도 산141-21을 확인함. 영어 산 표기 생략.",
        [vk("english", 189341), vk("japanese", 59199)],
    ),
    "128670": (
        "강주연못", "MATCH",
        "관광공사 강주연못 영어 안내는 예하리 및 한국어 상세주소 예하리 911-11을 같은 연못에 병기함. 외국어 지번 생략이며 대상 일치.",
        [vk("english", 201425)],
    ),
    "129550": (
        "장자호수공원", "MATCH",
        "관광공사 영·일 장자호수공원 안내가 토평동 878과 한국어 교문동 537을 동일 호수공원 주소로 병기함. 지번이 같은 필지라는 의미는 아니며 관광지 범위 일치.",
        [vk("english", 93627), vk("japanese", 67057)],
    ),
    "129637": (
        "대천항", "MATCH",
        "관광공사 영·일 대천항 안내에 신흑동 2245와 한국어 신흑동 1250이 같은 항구 주소로 병기되어 있음. 항구 단위 일치, 지번 차이는 유지함.",
        [vk("english", 191403), vk("japanese", 191501)],
    ),
    "988449": (
        "월영교", "MATCH",
        "관광공사 영·일 월영교 안내 모두 상아동 및 한국어 지번 상아동 569를 함께 안내함. 외국어 지번 생략이며 같은 교량임.",
        [vk("english", 80957), vk("japanese", 79021)],
    ),
    "2569511": (
        "감악산 출렁다리", "MATCH",
        "관광공사 일본어 감악산 출렁다리 안내에서 설마천로 273-9와 한국어 설마천로 238을 동시 확인함. 같은 출렁다리 관광지이며 건물번호 차이는 유지함.",
        ["https://japanese.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId=174608"],
    ),
    "2769599": (
        "대청댐 전망대", "MATCH",
        "충청북도 공식 관광 안내는 대청댐 전망대를 현암정으로 병칭하고 대청호반로 206을 안내함. 청주시도 같은 주소와 전망대의 현암정을 확인함. 현암정휴게소 주소 표기에도 관광지 대상 일치.",
        ["https://tour.chungbuk.go.kr/www/selectTourCntnts.do?key=60&searchMainSe=ATTR&tourNo=2604", "https://cjyu.cheongju.go.kr/home/sub.do?code1=apply1&menukey=7104&mode=view&sn=220", vk("english", 140739)],
    ),
    "2751854": (
        "도치돌 알파카목장", "MATCH",
        "관광공사 일본어 도치돌 알파카목장 안내에 도치돌길 293과 한국어 도치돌길 303이 같은 목장 주소로 병기되어 있음. 동일 관광지로 판정하고 건물번호 차이는 유지함.",
        [vk("japanese", 199792)],
    ),
}


def main():
    with INPUT.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    assert len(rows) == 107
    assert fields[33:35] == ["team_final_decision", "team_review_note"]
    pending = {r["tourapi_ko_id"] for r in rows if r["team_final_decision"] == "REVIEW"}
    assert pending == set(EVIDENCE), "Unexpected changes to REVIEW candidates"

    for row in rows:
        if row["team_final_decision"] != "REVIEW":
            continue
        title, decision, note, urls = EVIDENCE[row["tourapi_ko_id"]]
        assert title == row["tourapi_title_ko"]
        row["team_final_decision"] = decision
        row["team_review_note"] = (
            f"[AI 보조검토][외부 웹 근거 확인 2026-11-24] {note} "
            f"TourAPI 언어별 좌표 최대 차이 {float(row['max_distance_m']):g}m. "
            f"근거 URL: {' | '.join(urls)}"
        )

    with OUTPUT.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("외부 웹 근거 확인:", len(EVIDENCE))
    print("최종 판정:", Counter(r["team_final_decision"] for r in rows))


if __name__ == "__main__":
    main()
