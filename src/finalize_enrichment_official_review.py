"""Resolve the 543 source-consistent candidates against captured official data.

Review decisions are AI-assisted comparisons of the downloaded public fields,
not human annotations. A matching number alone is insufficient: a seasonal
range, a different ticket type, and a different passenger category are treated
explicitly. The AI Hub value is retained rather than replaced with web data.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"


def verified(note, scope):
    return "VERIFIED", note, scope


def hold(note):
    return "HOLD", note, ""


def exclude(note):
    return "EXCLUDE", note, ""


OPEN = {
    "2718685": verified("09:00~18:00은 치유센터 운영시간으로 공식 안내에서 일치함. 숲길은 별도 상시 개방.", "치유센터 일반 운영시간(숲길 전체 시간이 아님)"),
    "741109": hold("08:30~18:30은 하절기/간절기에만 적용. 동절기 18:00 종료로 계절 조건 누락."),
    "127539": exclude("공식 안내 10:00~18:00과 AI Hub 개장시각 09:30 불일치."),
    "129443": verified("공식 09:00~18:00 일치.", "일반 운영시간"),
    "2447115": exclude("공식 09:00~18:00, AI Hub 09:00~21:00으로 불일치."),
    "127860": exclude("공식 08:30~17:30, AI Hub 08:00~17:30으로 불일치."),
    "2500687": verified("공식 상시 개방은 AI Hub 00:00~24:00과 의미상 일치.", "공원 상시 개방"),
    "2711379": exclude("공식 09:00~18:00으로 AI Hub 10:00~16:00과 불일치."),
    "126498": hold("10:00~22:00은 금·토요일 운영시간. 월~목·일요일 21:00 종료로 요일 조건 누락."),
    "1329201": verified("공식에 09:00~18:00으로 안내. 계절별 시간 변동 가능.", "기본 공지 운영시간; 계절별 변동 가능"),
    "1087114": hold("공식 안내는 숙박·야영장 입실/퇴실 시간만 있어 24시간 일반 운영 확인 불가."),
    "2436233": verified("공식 09:00~18:00 일치. 계절·기상에 따라 변경될 수 있음.", "수목원 일반 운영시간; 계절·기상 변동 가능"),
    "125583": exclude("공식 하절기 09:30~17:30, 동절기 10:00~17:00으로 AI Hub 09:00~17:00과 불일치."),
    "2555756": verified("공식 09:00~18:00 일치.", "일반 운영시간"),
    "2656482": verified("공식 09:00~22:00 일치.", "일반 운영시간"),
    "894145": hold("공식에는 한옥스테이 입퇴실 시간만 표시되어 10:00~18:00 적용 시설 확인 불가."),
    "1059479": verified("공식 상시 개방은 00:00~24:00과 의미상 일치.", "공원 상시 개방"),
    "613310": exclude("공식 05:00~22:00과 AI Hub 05:00~21:00 불일치."),
    "2563697": hold("공식 관광 안내 10:00~18:00은 평일만 해당; 주말 시간 다르고 운영사 안내도 일몰 기준."),
    "2774063": verified("공식 상시 개방과 00:00~24:00 일치.", "거리 상시 개방"),
    "298095": exclude("공식 상시 개방과 AI Hub 일출~일몰 불일치. 원문을 운영시간으로 추가하지 않음."),
    "125957": hold("09:00~18:00은 하절기만 해당; 동절기 17:00 종료."),
    "2633896": verified("공식 기본 운영 10:00~18:00 일치. 세부 시설/프로그램별 시간 상이.", "가산수피아 일반 운영시간; 개별 시설 시간 아님"),
    "127641": hold("공식 안내에 시설별 상이로만 표시되어 09:30~20:00의 대상 확인 불가."),
    "2603500": exclude("공식 09:00~18:00과 AI Hub 10:00~18:00 불일치."),
    "125909": hold("전체 시간 범위는 09:00~17:00과 일치하나 공식 점심 휴게 12:00~13:00 조건이 원문에 없음."),
    "1928731": exclude("공식 11:30~20:00과 AI Hub 11:30~20:30 불일치."),
    "126484": verified("공식 09:00~18:00 일치.", "일반 운영시간; 입장마감은 별도"),
    "126487": verified("공식 상시 개방과 00:00~24:00 일치.", "공원 상시 개방; 공연장·주변 시설 시간 아님"),
    "2733962": verified("공식 상시 개방과 00:00~24:00 일치.", "계곡 상시 개방"),
    "126668": exclude("공식 08:30~19:00과 AI Hub 08:00~19:00 불일치."),
    "407051": exclude("공식 일반 운영 09:00~18:00과 AI Hub 08:00~20:00 불일치; 축제시간 별도."),
    "126666": exclude("공식 비수기/성수기 모두 08:30 개장으로 AI Hub 09:00~18:00과 불일치."),
    "125790": exclude("공식 상시 개방과 AI Hub 09:00~18:00 불일치."),
    "125773": exclude("공식 하절기 06:00~18:00, 동절기 06:00~17:00으로 AI Hub 06:00~18:30 불일치."),
    "2481866": exclude("공식 09:00~18:00과 AI Hub 10:00~18:00 불일치."),
    "970460": verified("공식 상시 개방과 00:00~24:00 일치.", "공원 상시 개방"),
    "128961": verified("공식 사이트에서 365일 24시간 운영 확인. 주간·야간 연속 이용.", "일반 영업시간; 시설별 이용 제한·요금 별도"),
    "127075": verified("공식 상시 개방과 00:00~24:00 일치.", "공원 상시 개방"),
    "125891": exclude("공식 05:30~20:00과 AI Hub 00:00~24:00 불일치."),
    "584698": verified("공식 상시 운영과 00:00~24:00 일치.", "일반 영업시간"),
    "126528": verified("공식 09:30~17:00 일치.", "순교성지 일반 관람시간"),
    "126825": verified("공식 상시 개방과 00:00~24:00 일치.", "공원 상시 개방"),
    "2831210": verified("공식 10:00~18:00 일치.", "일반 운영시간; 매표마감 별도"),
    "945824": exclude("공식 09:00~18:00과 AI Hub 10:00~18:00 불일치."),
    "2027193": verified("공식 상시 개방과 00:00~24:00 일치.", "사찰 상시 개방; 프로그램 시간 아님"),
    "2569511": verified("공식 09:00~18:00 일치.", "일반 운영시간"),
    "126747": hold("09:00~21:00은 하절기 운영시간; 동절기 20:00 종료."),
    "2751854": hold("10:00~18:00은 하절기 운영시간; 동절기 09:30~17:30으로 계절 조건 누락."),
}

HOLIDAY = {
    "2718685": verified("공식 무장애 관광 안내에 치유센터 매주 토·일요일 및 법정공휴일 휴무 명시.", "치유센터 휴무일; 치유숲길은 별도 상시 개방"),
    "2613450": exclude("공식 연중무휴와 AI Hub 매주 월요일 휴무 불일치."),
    "2638475": verified("공식 매주 월요일 휴무 일치. 1월 1일·명절도 별도 휴무.", "정기휴무 요일; 전체 휴무일 목록 아님"),
    "2713440": exclude("공식은 토·일요일과 명절 등 휴무; AI Hub 매주 월요일·공휴일과 불일치."),
    "1845544": exclude("공식 매주 월요일만 명시; AI Hub는 화요일 휴무도 포함함."),
    "125583": verified("공식 매주 월요일 휴무 일치.", "정기휴무 요일"),
    "2555756": exclude("공식 매월 첫째 월요일 및 법정공휴일; AI Hub 토·일요일과 불일치."),
    "894145": exclude("공식 연중무휴와 AI Hub 월요일 휴무 불일치."),
    "2594360": verified("공식 매주 월요일 휴무 일치. 촬영일 별도 휴무.", "정기휴무 요일; 촬영일 등 별도 휴무 가능"),
    "125957": verified("공식 매주 월요일 휴무 일치.", "정기휴무 요일"),
    "2603500": verified("공식 매주 월요일 휴무 일치.", "정기휴무 요일"),
    "126484": verified("공식 매주 월요일 휴무 일치; 1월 1일 별도.", "정기휴무 요일; 전체 휴무일 목록 아님"),
    "971122": hold("공식은 1월 1일과 설·추석 연휴만 명시; 모든 공휴일 휴무로 해석할 수 없음."),
    "127817": verified("공식 매주 월요일 휴무 일치. 공휴일과 겹치면 다음날 휴원.", "정기휴무 요일; 공휴일 예외가 있음"),
    "2481866": verified("공식 매주 월요일 휴무 일치; 공휴일인 월요일은 개관.", "정기휴무 요일; 공휴일 예외가 있음"),
    "126528": verified("공식 매주 월요일 휴무 일치.", "정기휴무 요일"),
    "127740": verified("공식 매주 월요일 및 법정공휴일 휴무 일치.", "휴무일; 기관 사정에 따라 변동 가능"),
    "2831210": verified("공식 매주 월요일 휴무 일치.", "정기휴무 요일"),
    "945824": verified("공식 매주 월요일 휴무 일치. 1월 1일·공휴일 별도.", "정기휴무 요일; 전체 휴무일 목록 아님"),
    "126747": verified("공식 매주 월요일 휴무 일치; 공휴일이면 다음날 휴관.", "정기휴무 요일; 공휴일 예외가 있음"),
}

FEE = {
    "2718685": hold("공식 프로그램 요금표는 시간·개인/단체별 차등 요금. AI Hub 숲하모니 5,000원 및 2시간 10,000원의 프로그램·대상 조건 미확인."),
    "741109": exclude("공식 개인 성인 12,000원·어린이 9,000원, AI Hub 성인 9,000원·어린이 6,000원과 불일치."),
    "2719004": exclude("공식 축제 시즌 입장료 5,000원·비시즌 무료로 3,000원과 불일치."),
    "127539": verified("공식 자유이용권 성인 32,000원·어린이 22,000원과 일치; 공원입장권은 다른 가격.", "자유이용권 성인·어린이 요금(공원입장권 아님)"),
    "127860": exclude("공식 성인 6,000원·청소년 5,000원, AI Hub 4,000원·3,000원으로 불일치."),
    "126498": exclude("공식 종합·파크 종일권 요금 모두 AI Hub의 성인·청소년·어린이 요금과 불일치."),
    "1329201": exclude("공식 성인 16,000원·어린이 13,000원과 AI Hub 11,000원·8,000원 불일치."),
    "1845544": exclude("공식 종일권 68,000원·오후권 50,000원. AI Hub 60,000원의 현행 권종 불일치."),
    "128627": exclude("공식 자유이용권 성인 49,000원·청소년 44,000원·어린이 39,000원과 불일치."),
    "2436233": exclude("공식 성인 11,000원·어린이 7,000원, AI Hub 10,000원·6,000원으로 불일치."),
    "125583": exclude("공식 입장료 무료로 안내되어 AI Hub 3,000원·1,600원과 불일치."),
    "613310": exclude("공식 온천 사우나의 평일·주말 요금 모두 AI Hub 9,000원·7,000원과 불일치."),
    "126581": verified("공식 일반·중고생·군경·어린이 및 청소년/어린이 단체 요금 모두 대응 금액 일치.", "일반 입장료와 해당 단체 요금; 군경은 제복 사병, 단체는 인원 조건 적용"),
    "2563697": verified("운영사 공식 페이지 리프트+모노레일 왕복 일반 대인 17,000원·소인 14,000원 일치.", "리프트+모노레일 왕복 일반 패키지 요금"),
    "2633896": exclude("한국관광공사 국문 상세에는 무료, 영문 상세에는 성인 8,000원·어린이 6,000원으로 상이함; 모두 AI Hub 성인 9,000원과 불일치."),
    "1626349": exclude("공식 대인 편도 11,000원은 일치하나 소인 편도 5,500원·왕복 8,000원으로 AI Hub 6,000원과 일치하는 같은 권종 없음."),
    "127889": hold("공식 예약은 시즌·권종별 가격을 안내. AI Hub 34,000원·26,000원의 적용 시즌 및 권종 확인 불가."),
    "662637": exclude("공식 개인 성인 11,000원·소인 9,000원으로 AI Hub 7,000원·5,000원과 불일치."),
    "125761": exclude("공식 입장료 무료와 AI Hub 유료 입장료 불일치."),
    "126668": verified("공식 일반 어른 11,000원·청소년 8,500원·어린이 7,500원 모두 일치.", "일반 개인 입장료; 할인·특별요금 별도"),
    "407051": exclude("공식 일반 성인 7,000원·경로 4,000원·어린이/청소년 4,000원으로 AI Hub 값과 불일치."),
    "127817": exclude("공식 입장료 무료와 AI Hub 유료 요금 불일치."),
    "126666": exclude("AI Hub 성인 4,000원은 단체, 어린이 2,000원은 개인 요금과 각각 일치해 동일 권종 조합 불일치."),
    "125773": exclude("공식 입장료 무료와 AI Hub 유료 입장료 불일치."),
    "128961": exclude("운영사 공식 주간 사우나 성인 13,000원 등 안내. AI Hub 10,000원의 적용 요금 불일치."),
    "584698": exclude("공식 입장료 22,000원과 AI Hub 16,000원 불일치."),
    "126397": exclude("공식 개인 성인 4,000원으로 AI Hub 3,000원과 다름. 나머지 일부 금액 일치하나 전체 원문을 제외."),
    "127740": verified("공식 개인 성인 7,000원·어린이 5,000원 모두 일치.", "개인 성인·어린이 요금; 청소년 등 추가 대상 별도"),
    "2751854": exclude("공식 개인 일반 15,000원으로 AI Hub 10,000원과 불일치."),
}

# Additional named official sources for scope ambiguity, missing KTO fields,
# and known contradictions; only the listed field is affected.
EXTRA = {
    ("2718685", "opening_hours"): ("https://access.visitkorea.or.kr/ms/detail.do?cotId=1ea886f6-0f61-46ea-afd3-977103797563", "치유센터 09:00~18:00 / 치유숲길 상시 개방"),
    ("2718685", "holidays"): ("https://access.visitkorea.or.kr/ms/detail.do?cotId=1ea886f6-0f61-46ea-afd3-977103797563", "치유센터 매주 토요일, 일요일, 법정공휴일 / 치유숲길 연중무휴"),
    ("2718685", "admission_fee"): ("https://www.sooperang.go.kr/pot/cn/selectCntnsView.do?cntnsSeq=91&hmpgId=FT00006", "프로그램 개인 1시간 6,000원·2시간 12,000원 / 단체 5,000원·10,000원, 20인 이상"),
    ("2718685", "parking"): ("https://www.sooperang.go.kr/pot/cn/selectCntnsView.do?cntnsSeq=91&hmpgId=FT00006", "치유의 숲 내 주차장은 프로그램 예약고객만 주차 가능"),
    ("2563697", "opening_hours"): ("https://www.jisanpark.co.kr/15", "운영사 주중 10시~일몰, 주말 09시~일몰"),
    ("2563697", "admission_fee"): ("https://www.jisanpark.co.kr/15", "리프트+모노레일 왕복 일반 대인 17,000원; 소인 14,000원"),
    ("127889", "admission_fee"): ("https://reservation.everland.com/web/cb.do?method=productMain", "시즌·권종·우대조건별 예약 상품 표시; AI Hub 금액 조건 미확인"),
    ("2633896", "admission_fee"): ("https://english.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId=216204", "영문 관광공사 안내 성인 8,000원·어린이 6,000원"),
    ("128961", "admission_fee"): ("https://ispaland.co.kr/guide/", "주간 사우나 성인 13,000원·소인 9,000원 등; 시간/시설별 요금"),
    ("128961", "opening_hours"): ("https://ispaland.co.kr/guide/", "365일 24시 연중무휴; 주간 05~20시, 야간 20~05시"),
}


def read(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (ROOT / name).open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert read(name) == rows


def decision(pid, field, evidence):
    if evidence["fetch_status"] != "OK" or evidence["title_ko"] != evidence["official_title"]:
        return hold("공식 상세자료 수집 실패 또는 관광지 식별자 확인 필요.")
    if field == "opening_hours":
        return OPEN[pid]
    if field == "holidays":
        return HOLIDAY[pid]
    if field == "admission_fee":
        return FEE[pid]
    if field == "parking":
        if pid == "2718685":
            return hold("관광공사에는 가능으로 표시되지만 운영사 안내는 프로그램 예약고객만 주차 가능. 일반 방문객 주차 가능으로 보완하지 않음.")
        if pid == "1899635":
            return verified("공식 개방기간 현장 주차 안내에 따라 이용 가능. 상시 주차 여부 보장 아님.", "은행나무숲 개방기간에 현장 안내를 따르는 주차")
        value = evidence["parking"].strip()
        if value.startswith("불가능"):
            return exclude("공식 주차 항목은 불가능으로 AI Hub 주차 가능 정보와 불일치.")
        if value.startswith("가능"):
            return verified("공식 상세 주차 항목에 가능으로 표시되어 AI Hub 주차 가능과 일치.", "주차 가능 여부만; 요금·면수·주차장 위치 추정 금지")
        return hold("공식 주차 가능 여부를 구체적으로 확인할 수 없음.")
    raise ValueError(field)


def main():
    rows = read("common_enrichment_screening.csv")
    evidence = {r["place_id"]: r for r in read("common_enrichment_official_evidence.csv")}
    candidates = [r for r in rows if r["screening_status"] == "SOURCE_CONSISTENT"]
    keys = {(r["place_id"], r["field"]) for r in candidates}
    assert len(rows) == 736 and len(candidates) == 543 and len(keys) == 181 and len(evidence) == 96
    assert {p for p, f in keys if f == "opening_hours"} == set(OPEN)
    assert {p for p, f in keys if f == "holidays"} == set(HOLIDAY)
    assert {p for p, f in keys if f == "admission_fee"} == set(FEE)
    assert set(EXTRA) <= keys
    assert all(evidence[p]["fetch_status"] == "OK" and evidence[p]["title_ko"] == evidence[p]["official_title"] for p in evidence)

    group_decisions = {key: decision(*key, evidence[key[0]]) for key in keys}
    out = []
    for r in rows:
        before = r["screening_status"]
        new = {**r, "screening_before_official_check": before, "official_detail_source_url": "",
               "official_detail_evidence": "", "official_detail_title": "", "official_detail_address": "",
               "official_detail_collected_at_kst": "", "additional_official_source_url": "", "additional_official_evidence": ""}
        if before == "SOURCE_CONSISTENT":
            key = r["place_id"], r["field"]
            e = evidence[key[0]]
            status, note, scope = group_decisions[key]
            extra_url, extra_text = EXTRA.get(key, ("", ""))
            new.update({
                "screening_status": status,
                "candidate_value": r["candidate_value"] if status == "VERIFIED" else "",
                "source_check_method": "AI_KTO_AND_ADDITIONAL_OFFICIAL_CHECK" if extra_url else "AI_OFFICIAL_KTO_DETAIL_CHECK",
                "verification_source": extra_url or e["source_url"],
                "scope_note": scope,
                "screening_note": note,
                "eligible_for_enrichment": "YES" if status == "VERIFIED" else "NO",
                "official_detail_source_url": e["source_url"],
                "official_detail_evidence": e[key[1]],
                "official_detail_title": e["official_title"],
                "official_detail_address": e["official_address"],
                "official_detail_collected_at_kst": e["collected_at_kst"],
                "additional_official_source_url": extra_url,
                "additional_official_evidence": extra_text,
            })
        out.append(new)

    assert len(out) == 736 and len({(r["place_id"], r["language"], r["field"]) for r in out}) == 736
    assert sum(r["screening_before_official_check"] == "SOURCE_CONSISTENT" for r in out) == 543
    assert not any(r["screening_status"] == "SOURCE_CONSISTENT" for r in out)
    assert all(r["candidate_value"] and r["verification_source"] and r["scope_note"] for r in out if r["eligible_for_enrichment"] == "YES")
    assert all(not r["candidate_value"] and r["eligible_for_enrichment"] == "NO" for r in out if r["screening_status"] != "VERIFIED")
    verified_rows = [r for r in out if r["screening_status"] == "VERIFIED"]
    names = ("common_enrichment_screening_final.csv", "common_verified_enrichment_final.csv")
    assert all(not (ROOT / n).exists() for n in names)
    write(names[0], out)
    write(names[1], verified_rows)
    print("New group decisions:", Counter(s[0] for s in group_decisions.values()))
    print("New row decisions:", Counter(r["screening_status"] for r in out if r["screening_before_official_check"] == "SOURCE_CONSISTENT"))
    print("All rows:", Counter(r["screening_status"] for r in out))
    print("Verified rows:", len(verified_rows), "places:", len({r["place_id"] for r in verified_rows}), "place-field groups:", len({(r["place_id"], r["field"]) for r in verified_rows}))
    print("Verified by field:", dict(Counter(r["field"] for r in verified_rows)))
    print("Verified by language:", dict(Counter(r["language"] for r in verified_rows)))


if __name__ == "__main__":
    main()
