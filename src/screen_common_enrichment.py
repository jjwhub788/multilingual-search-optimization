"""Screen multilingual AI Hub values for potential TourAPI description enrichment.

The unit of source comparison is a (place_id, field) group; the unit of output
is a (place_id, language, field) gap. Agreement between translations is not
proof of correctness/currentness. Only documented checks get VERIFIED.
"""

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"
LANGS = ("ko", "en", "ja")
CHECK_DATE = "2026-11-24"


def vk(content_id):
    return f"https://english.visitkorea.or.kr/svc/whereToGo/locIntrdn/rgnContentsView.do?vcontsId={content_id}"


CHEORWON = vk(216236)
GYEONGBOK = "https://english.visitkorea.or.kr/svc/contents/contentsView.do?vcontsId=87740"
CHEONGTAE = vk(94825)
GAEAM = vk(105357)
GAPSA = vk(110903)
JADE = vk(82222)
KKANG = vk(199645)
BUSAN_SKY = "https://www.busanxthesky.com/"
LOTTE = "https://app.lottehotel.com/resort-gimhae/ko/themepark/lotte-water-park"
HAENGJU = "https://access.visitkorea.or.kr/cos/detail.do?cotId=ac0039ee-8a79-4185-b5a1-72c97053c25b"

# Official evidence checked during the work. Do not apply these results to
# other places or attributes. Scope notes are verification metadata, not new
# AI Hub facts to silently insert into the baseline.
WEB_CHECKS = {
    ("126346", "experience"): ("VERIFIED", "https://gaeamsa.templestay.com/", "개암사 공식 템플스테이 프로그램 확인.", "템플스테이 프로그램 존재"),
    ("125891", "experience"): ("VERIFIED", "https://www.templestay.com/fe/MI000000000000000062/templestay/prgList.do?areaCd=CD00000303&areaSelect=CD00000303&pageIndex=1&searchCnt=&searchEndDate=&searchKeyword=%EA%B0%91%EC%82%AC&searchStaDate=&templeBookMarkId=&templeId=&templeIdTmp=&templePrgType=&templestaySeq=", "공식 템플스테이 프로그램 목록 및 TourAPI 한국어·일본어 설명과 일치.", "템플스테이 프로그램 존재"),
    ("2554070", "experience"): ("VERIFIED", KKANG, "한국관광공사에서 해상·마을·통합투어 안내 확인.", "투어 종류 안내; 회차·운영시각 보장 아님"),
    ("125266", "experience"): ("HOLD", "https://www.foresttrip.go.kr/indvz/main.do?hmpgId=0106", "건강숲길 내 일부 황토길 운영 중단 공지. 현재 이용 가능한 세부 구간 불명확.", ""),
    ("2719004", "experience"): ("HOLD", "https://www.gochang.go.kr/tour/board/view.gochang?boardId=BBS_0000398&dataSid=806810&menuCd=DOM_000000401005000000", "고창군 자료와 영문 설명의 Geochang 지명 불일치. 시설명/활동의 새 정보 범위도 재확인 필요.", ""),
    ("2668973", "opening_hours"): ("VERIFIED", BUSAN_SKY, "공식 전망대 운영시간 10:00~21:00과 일치.", "전망대 일반 운영시간"),
    ("126346", "opening_hours"): ("VERIFIED", GAEAM, "공식 관광 안내 09:00~18:00 일치.", "사찰 일반 이용시간"),
    ("126508", "opening_hours"): ("HOLD", GYEONGBOK, "09:00~18:30은 6~8월에만 해당하며 나머지 계절 폐장시간 다름. 원문의 계절 조건 누락.", ""),
    ("3072021", "opening_hours"): ("HOLD", CHEORWON, "09:00~18:00은 3~11월 공원 운영시간이며 동절기·모노레일 시간 상이.", ""),
    ("125266", "opening_hours"): ("VERIFIED", CHEONGTAE, "공식 안내의 당일 방문객 운영시간 09:00~18:00과 일치.", "당일 방문객 이용시간; 숙박 체크인/아웃 시간 아님"),
    ("126508", "holidays"): ("VERIFIED", GYEONGBOK, "화요일 정기휴무 일치. 공휴일이면 다음 영업일 휴무 예외 있음.", "정기휴무 요일만; 공휴일 예외가 있음"),
    ("2554070", "holidays"): ("VERIFIED", KKANG, "관광공사 월요일 휴무 안내와 일치.", "투어 등 운영시설의 정기휴무; 마을 전체 출입 금지 아님"),
    ("3072021", "holidays"): ("VERIFIED", CHEORWON, "화요일 휴무 일치. 명절 휴무도 별도 존재.", "정기휴무 요일만; 전체 휴무일 목록 아님"),
    ("125266", "holidays"): ("HOLD", CHEONGTAE, "화요일 휴무에 7~8월 예외가 있어 원문에 적용 조건 누락.", ""),
    ("126508", "admission_fee"): ("VERIFIED", GYEONGBOK, "공식 성인 입장료 3,000원 확인.", "일반 성인 입장료; 면제·할인 대상 별도"),
    ("125266", "admission_fee"): ("VERIFIED", CHEONGTAE, "개인 입장료 성인 1,000원, 어린이 300원 일치.", "개인 입장료의 두 대상만; 전체 요금표 아님"),
    ("1607565", "admission_fee"): ("EXCLUDE", JADE, "공식 성인 요금 11,000원은 AI Hub 10,000원과 다름. 원문 전체 요금 항목을 제외.", ""),
    ("125891", "admission_fee"): ("EXCLUDE", GAPSA, "공식 관광 안내의 개인 요금표와 AI Hub 금액 불일치. 현재 요금 확정 대신 AI Hub 원문 제외.", ""),
    ("2554070", "admission_fee"): ("HOLD", KKANG, "마을 자체는 무료 안내, AI Hub 6,000원은 투어 등 개별 프로그램 범위 확인 필요.", ""),
    ("1921423", "admission_fee"): ("HOLD", LOTTE, "시설·시즌별로 운영 조건 상이; 75,000/61,000원의 적용 시즌·권종 미확인.", ""),
    ("125636", "parking"): ("HOLD", "https://korean.visitkorea.or.kr/kor/bz15/where/where_tour.jsp?cid=125636", "공식 설명에는 캠프장 주차시설이 있지만 상세 주차 항목은 불가능으로 상충. 범위 확인 전 보류.", ""),
}

for pid, url, scope in [
    ("1921423", LOTTE, "워터파크 방문객 주차 가능"),
    ("126346", GAEAM, "개암사 주차 가능"),
    ("126508", GYEONGBOK, "경복궁 주차 가능"),
    ("2668973", BUSAN_SKY, "전망대 고객 주차 가능"),
    ("2743288", "https://seonyudocruise.com/useinfo/parking/", "선유도유람선 선착장 고객 주차 가능"),
    ("2764342", vk(188837), "온천랜드 주차 가능"),
    ("2661736", HAENGJU, "행주산성 역사공원 주차 가능"),
    ("125266", CHEONGTAE, "자연휴양림 방문객 주차 가능"),
    ("1607565", JADE, "수목원 주차 가능"),
    ("3072021", CHEORWON, "역사문화공원 주차 가능"),
    ("125891", GAPSA, "갑사 주차 가능"),
    ("126341", vk(104966), "백양사 방문객 주차 가능"),
]:
    WEB_CHECKS[pid, "parking"] = ("VERIFIED", url, "공식 안내에 방문객 주차시설이 명시됨.", scope + "; 주차료·면수는 추가하지 않음")


def norm(value):
    return unicodedata.normalize("NFKC", value).strip().casefold()


def compact(value):
    return re.sub(r"\s+", "", norm(value)).strip(".")


def hour_key(value):
    v = re.sub(r"\s+", "", norm(value)).replace("to", "~").replace("-", "~").replace("–", "~").replace("—", "~")
    if v in {"일출~일몰", "sunrise~sunset", "日の出~日没"}:
        return "SUNRISE_SUNSET"
    m = re.fullmatch(r"(\d{1,2}):(\d{2})~(\d{1,2}):(\d{2})", v)
    if m:
        h1, m1, h2, m2 = map(int, m.groups())
        if 0 <= h1 < 24 and 0 <= m1 < 60 and (0 <= h2 < 24 and 0 <= m2 < 60 or h2 == 24 and m2 == 0):
            if h1 * 60 + m1 != h2 * 60 + m2:
                return f"{h1:02}:{m1:02}~{h2:02}:{m2:02}"
    return ""


DAYS = [
    ("MON", "월요일", "monday", "月曜日"), ("TUE", "화요일", "tuesday", "火曜日"),
    ("WED", "수요일", "wednesday", "水曜日"), ("THU", "목요일", "thursday", "木曜日"),
    ("FRI", "금요일", "friday", "金曜日"), ("SAT", "토요일", "saturday", "土曜日"),
    ("SUN", "일요일", "sunday", "日曜日"), ("PUBLIC_HOLIDAY", "공휴일", "publicholiday", "祝日"),
]


def day_key(value, lang):
    v = compact(value)
    i = LANGS.index(lang) + 1
    found = [row[0] for row in DAYS if row[i] in v]
    # Remove only observed exact day names and permissible connector words.
    for row in DAYS:
        v = v.replace(row[i], "")
    for token in ("매주", "毎週", "every", "and", "、", ",", "s"):
        v = v.replace(token, "")
    return "+".join(found) if found and not v else ""


def fees(value):
    # Preserve amounts (including age/role order) without treating a program's
    # 2-hour duration as a fee. Commas inside numbers are thousands separators.
    v = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", norm(value))
    explicit = re.findall(r"(\d+)\s*(?:원|ウォン|krw|won)\b", v)
    if explicit:
        return tuple(map(int, explicit))
    return tuple(map(int, re.findall(r"\d+", v)))


NO_VALUE = {"없음", "none", "no", "notavailable", "なし", "無し"}
VAGUE = {"전화문의", "atelephoneenquiry", "atelephoneinquiry", "電話でのお問い合わせ", "pleaseall", "pleasecall", "상황에따라다름", "dependingonthesituation", "状況によって異なる"}
POS_PARK = ({"있음", "가능"}, {"yes", "available"}, {"有り", "あり", "有"})


def screen_group(key, rows):
    pid, field = key
    vals = [rows[l]["aihub_raw_value"] for l in LANGS]
    cv = [compact(v) for v in vals]
    # In these data the English scan sometimes marked No/telephone enquiry
    # VALUE_CANDIDATE. They are not usable information of the target field.
    if key in WEB_CHECKS:
        status, url, note, scope = WEB_CHECKS[key]
        return status, "AI_OFFICIAL_WEB_CHECK", url, note, scope, ""
    if any(x in VAGUE or not x for x in cv):
        return "EXCLUDE", "VALUE_SEMANTIC_SCREEN", "", "전화문의·상황별 변동·빈 값 등 실제 보완할 구체 정보 없음.", "", ""
    if field == "experience":
        if pid in {"127860", "2436233"}:
            return "EXCLUDE", "AI_SEMANTIC_SCREEN", "", "식물원/수목원은 시설 유형에 해당하며 구체 체험 내용을 제공하지 않음.", "", ""
        if any(x in NO_VALUE for x in cv):
            return "EXCLUDE", "VALUE_SEMANTIC_SCREEN", "", "체험 없음 또는 여부 불명확이라는 값으로 구체 체험 정보 없음.", "", ""
        return "HOLD", "AI_SEMANTIC_SCREEN", "", "체험 사실의 세부 범위와 새 정보 여부를 확인해야 함.", "", ""
    if field == "parking":
        if all(v in p for v, p in zip(cv, POS_PARK)):
            return "SOURCE_CONSISTENT", "MULTILINGUAL_VALUE_MATCH", "", "세 언어의 주차 가능 값 일치. 주차장 존재나 접근 조건의 현재 상태는 공식 확인 전 미확정.", "주차 가능 여부만; 요금·면수 등은 포함하지 않음", "PARKING_AVAILABLE"
        return "HOLD", "MULTILINGUAL_VALUE_CHECK", "", "부정형이나 불명확 값은 주차장 없음과 정보 없음의 구분이 어려워 확인 전 미사용.", "", ""
    if field == "opening_hours":
        if any(x in NO_VALUE for x in cv):
            return "EXCLUDE", "VALUE_SEMANTIC_SCREEN", "", "운영시간을 특정하는 값이 아님.", "", ""
        keys = [hour_key(v) for v in vals]
        if all(keys) and len(set(keys)) == 1:
            return "SOURCE_CONSISTENT", "MULTILINGUAL_VALUE_MATCH", "", "세 언어에서 동일한 단일 운영시간 범위 확인. 계절·시설별 적용 및 최신성은 공식 확인 전 미확정.", "기록된 운영시간; 일반/시즌/시설 범위 확인 필요", keys[0]
        return "HOLD", "MULTILINGUAL_VALUE_CHECK", "", "복수 운영시간 또는 해석 불일치. 적용 요일·계절·시설을 알 수 없어 보류.", "", ""
    if field == "holidays":
        if any(x in NO_VALUE for x in cv):
            return "HOLD", "MULTILINGUAL_VALUE_CHECK", "", "없음/No/なし를 연중무휴로 단정하지 않음. 영어가 무휴로 번역되어도 원문 확인 필요.", "", ""
        keys = [day_key(v, l) for v, l in zip(vals, LANGS)]
        if all(keys) and len(set(keys)) == 1:
            return "SOURCE_CONSISTENT", "MULTILINGUAL_VALUE_MATCH", "", "세 언어의 휴무 요일·공휴일 표현 일치. 예외 일정과 최신성은 확인 전 미확정.", "기록된 정기휴무 요일/공휴일; 전체 휴무 일정 아님", keys[0]
        return "HOLD", "MULTILINGUAL_VALUE_CHECK", "", "휴무 조건 또는 공휴일 지칭이 명확히 일치하지 않아 보류.", "", ""
    if field == "admission_fee":
        if any(x in NO_VALUE for x in cv):
            return "EXCLUDE", "VALUE_SEMANTIC_SCREEN", "", "없음/No 등의 값은 무료 요금으로 확정할 수 없음.", "", ""
        if pid == "2831210":
            return "HOLD", "AI_SEMANTIC_SCREEN", "", "영어 Youth Rate와 한국어·일본어 어린이 요금 대상 범위 상이.", "", ""
        if pid == "1928731":
            return "HOLD", "AI_SEMANTIC_SCREEN", "", "정/성 코스 비용은 식사 코스일 수 있어 입장·체험 이용료인지 불명확.", "", ""
        if pid == "2555756":
            return "HOLD", "AI_SEMANTIC_SCREEN", "", "개인·단체 금액은 있으나 적용 프로그램 또는 시설 범위 불명확.", "", ""
        # Semantic review of the candidate fee groups: category ordering and
        # intended targets were compared in addition to numeric equality.
        approved_fee_mapping = {
            "2718685", "741109", "2719004", "127539", "127860", "126498", "1329201", "1845544", "128627", "2436233",
            "125583", "613310", "126581", "2563697", "2633896", "1626349", "127889", "662637", "125761",
            "126668", "407051", "127817", "126666", "125773", "128961", "584698", "126397", "127740", "2751854",
        }
        amounts = [fees(v) for v in vals]
        if pid in approved_fee_mapping and all(amounts) and len(set(amounts)) == 1:
            return "SOURCE_CONSISTENT", "MULTILINGUAL_VALUE_AND_AI_SEMANTIC_CHECK", "", "금액 및 해당 대상 표현의 대응 확인. 실제 입장료/권종·시즌·최신성 공식 확인 필요.", "원문 요금 및 대상 범위만; 무료·할인·추가 요금 추정 금지", "+".join(map(str, amounts[0]))
        return "HOLD", "MULTILINGUAL_VALUE_CHECK", "", "요금 금액 또는 적용 대상/단위의 일치가 불명확하여 보류.", "", ""
    raise ValueError(field)


def read_csv(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_new(name, rows):
    assert rows
    with (ROOT / name).open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert read_csv(name) == rows


def main():
    all_rows = read_csv("common_gap_review_final.csv")
    groups = defaultdict(dict)
    for r in all_rows:
        groups[r["place_id"], r["field"]][r["language"]] = r
    assert len(all_rows) == 2070 and all(set(g) == set(LANGS) for g in groups.values())
    assert sum(r["supplement_status"] == "CANDIDATE_UNVERIFIED" for r in all_rows) == 725
    assert sum(r["supplement_status"] == "APPROVE" for r in all_rows) == 11
    decisions = {
        key: screen_group(key, g)
        for key, g in groups.items()
        if any(r["supplement_status"] == "CANDIDATE_UNVERIFIED" for r in g.values())
    }
    assert len(decisions) == 261
    out = []
    for r in all_rows:
        state = r["supplement_status"]
        if state not in {"CANDIDATE_UNVERIFIED", "APPROVE"}:
            continue
        key = r["place_id"], r["field"]
        g = groups[key]
        if state == "APPROVE":
            status, method, url, note, scope, nk = (
                "VERIFIED", r["supplement_review_provenance"], r["verification_source"], r["supplement_review_note"], r["supplement_scope"], ""
            )
            value = r["approved_value"]
        else:
            status, method, url, note, scope, nk = decisions[key]
            value = r["aihub_raw_value"] if status in {"VERIFIED", "SOURCE_CONSISTENT"} else ""
        assert r["tourapi_final_status"] in {"MISSING", "PARTIAL"}
        assert status != "VERIFIED" or (url and value and scope)
        assert status != "SOURCE_CONSISTENT" or (not url and value)
        out.append({
            "place_id": r["place_id"], "language": r["language"], "title": r["tourapi_title"], "title_ko": g["ko"]["tourapi_title"],
            "information_type": r["information_type"], "field": r["field"], "tourapi_status": r["tourapi_final_status"],
            "tourapi_review_note": r["review_note"], "aihub_raw_value": r["aihub_raw_value"],
            **{f"aihub_value_{l}": g[l]["aihub_raw_value"] for l in LANGS},
            "ko_value_from_en": r["ko_value_from_en"] if r["language"] == "ko" else g["ko"]["ko_value_from_en"],
            "ko_value_from_ja": r["ko_value_from_ja"] if r["language"] == "ko" else g["ko"]["ko_value_from_ja"],
            "prior_status": state, "screening_status": status, "candidate_value": value,
            "normalized_comparison_key": nk, "source_check_method": method, "verification_source": url,
            "screening_date": CHECK_DATE, "scope_note": scope, "screening_note": note,
            "eligible_for_enrichment": "YES" if status == "VERIFIED" else "NO",
        })
    assert len(out) == 736 and len({(r["place_id"], r["language"], r["field"]) for r in out}) == 736
    verified = [r for r in out if r["eligible_for_enrichment"] == "YES"]
    names = ("common_enrichment_screening.csv", "common_verified_enrichment.csv")
    assert all(not (ROOT / n).exists() for n in names)
    write_new(names[0], out)
    write_new(names[1], verified)
    print("All records:", Counter(r["screening_status"] for r in out))
    print("New screening:", Counter(r["screening_status"] for r in out if r["prior_status"] == "CANDIDATE_UNVERIFIED"))
    print("Group screening:", Counter(status[0] for status in decisions.values()))
    print("Verified:", len(verified), "groups:", len({(r["place_id"], r["field"]) for r in verified}), "places:", len({r["place_id"] for r in verified}))
    for field in ("experience", "opening_hours", "holidays", "admission_fee", "parking"):
        print(field, dict(Counter(r["screening_status"] for r in out if r["field"] == field)))
    print("Source-consistent holds: verify with official sources before use.")


if __name__ == "__main__":
    main()
