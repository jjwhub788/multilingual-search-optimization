"""Combine the user's recorded source checks with a separately labeled AI recheck.

Run from the project root. The original AI review and the user's exported CSV
are never modified. Unreviewed AI Hub candidates remain unverified.
"""

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"
LANGS = ("ko", "en", "ja")

# Second AI reading of the collected TourAPI descriptions, not human labels.
AI_RECHECK = {
    ("584698", "experience"): (
        ("PRESENT", "PRESENT", "PRESENT"),
        "한국어·일본어는 노천탕·족욕·찜질방, 영어는 여성전용 찜질방·사우나를 명시함. 세 언어 모두 구체 체험 시설 존재.",
    ),
    ("127859", "opening_hours"): (
        ("PARTIAL", "MISSING", "MISSING"),
        "한국어는 강변물놀이장 여름철 이용 조건만 명시하고 운영 시각은 없음. 영어·일본어는 운영시간 안내 없음.",
    ),
    ("126487", "holidays"): (
        ("PARTIAL", "MISSING", "MISSING"),
        "한국어의 연중무휴는 일부 공연·행사에 한정되므로 공원 전체의 휴무일 판정을 PARTIAL로 조정. 영어·일본어는 휴무 안내 없음.",
    ),
    ("126487", "admission_fee"): (
        ("PRESENT", "MISSING", "MISSING"),
        "한국어의 일부 공연·행사 무료 안내는 해당 프로그램에 대한 요금 정보임. 영어·일본어는 무료 여부를 명시하지 않음.",
    ),
    ("3072021", "parking"): (
        ("PRESENT", "MISSING", "MISSING"),
        "한국어는 주차장 요금 무료를 명시해 주차시설 존재 확인. 영어·일본어에는 주차시설 안내 없음.",
    ),
}

# Accepted values are restricted to the actual AI Hub value/variant supported
# by the recorded check. Equal times in the other languages are normalized from
# the original source, not new facts obtained from a third source.
APPROVED = {
    ("250398", "parking"): (
        ("있음", "Yes", ""), "주차 가능 여부만 보완. 일본어 AI Hub 원문 なし는 사용하지 않음.",
    ),
    ("2764342", "opening_hours"): (
        ("05:00~22:50", "05:00~22:50", "05:00~22:50"), "온천랜드 일반 운영시간; 야외 수영장 운영시간 아님.",
    ),
    ("1607565", "opening_hours"): (
        ("09:00~18:00", "09:00~18:00", "09:00~18:00"), "수목원 일반 운영시간; 야간 특별 개장 시간 아님.",
    ),
    ("769761", "holidays"): (
        ("매주 월요일", "Every Monday", "毎週月曜日"), "분수 운영기간에 적용되는 월요일 휴무; 연중 운영을 뜻하지 않음.",
    ),
}


def read_csv(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_new(name, columns, rows):
    path = ROOT / name
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    actual_columns, actual_rows = read_csv(name)
    assert actual_columns == columns and actual_rows == rows


def group_key(row):
    return row["place_id"], row["field"]


def main():
    cols, original = read_csv("common_gap_manual_review.csv")
    user_cols, checked = read_csv("common_gap_manual_review_checked.csv")
    gap_cols, gap = read_csv("common_gap_review_ai_completed.csv")
    assert cols == user_cols and len(original) == len(checked) == 20
    assert len(gap) == 2070
    by_group = {group_key(r): r for r in checked}
    assert len(by_group) == 20 and set(by_group) == {group_key(r) for r in original}
    editable = set(cols[22:])
    manual_extra = [
        "review_provenance", "effective_verification_source", "approved_scope",
        *(f"effective_approved_value_{l}" for l in LANGS),
        *(f"ai_recheck_status_{l}" for l in LANGS), "ai_recheck_note",
    ]
    review_rows = []
    for r in original:
        key = group_key(r)
        u = by_group[key]
        # Numbers may change display formatting. Preserve the original source data.
        assert all(r[c] == u[c] for c in cols if c not in editable), key
        out = {**r, **{c: u[c] for c in editable}, **dict.fromkeys(manual_extra, "")}
        if r["review_priority"] == "HIGH":
            assert u["supplement_review_status"] in {"APPROVE", "HOLD", "EXCLUDE"}, key
            out["review_provenance"] = "USER_RECORDED_WITH_AI_ASSISTANCE"
            out["effective_verification_source"] = u["verification_source"]
            if key == ("2668973", "admission_fee") and not out["effective_verification_source"]:
                # URL supplied during the conversation; user independently confirmed the listed prices.
                out["effective_verification_source"] = "https://www.busanxthesky.com/"
            if u["supplement_review_status"] == "APPROVE":
                values, scope = APPROVED[key]
                assert u["verification_source"] and u["confirmed_value_ko"]
                for l, v in zip(LANGS, values):
                    if u[f"confirmed_value_{l}"]:
                        assert u[f"confirmed_value_{l}"] == v, (key, l)
                    out[f"effective_approved_value_{l}"] = v
                out["approved_scope"] = scope
        else:
            assert r["review_priority"] == "QA_SAMPLE" and key in AI_RECHECK
            assert all(not u[c] for c in editable), key
            labels, note = AI_RECHECK[key]
            out["review_provenance"] = "AI_RECHECK_NOT_HUMAN"
            for l, s in zip(LANGS, labels):
                out[f"ai_recheck_status_{l}"] = s
            out["ai_recheck_note"] = "[AI 재검토] " + note
        review_rows.append(out)

    reviewed = {group_key(r): r for r in review_rows}
    assert Counter(r["supplement_review_status"] for r in review_rows if r["review_priority"] == "HIGH") == {"APPROVE": 4, "HOLD": 5, "EXCLUDE": 6}
    gap_extra = [
        "presence_review_provenance", "supplement_status", "approved_value",
        "supplement_scope", "supplement_review_provenance", "verification_source", "supplement_review_note",
    ]
    final = []
    changed = 0
    for row in gap:
        key, l = group_key(row), row["language"]
        r = {**row, **dict.fromkeys(gap_extra, "")}
        r["presence_review_provenance"] = "AI_INITIAL_REVIEW"
        g = reviewed.get(key)
        if key in AI_RECHECK:
            revised = g[f"ai_recheck_status_{l}"]
            changed += revised != r["tourapi_final_status"]
            r["tourapi_final_status"] = revised
            r["review_note"] += " | " + g["ai_recheck_note"]
            r["presence_review_provenance"] = "AI_INITIAL_AND_RECHECK"
        if g and g["review_priority"] == "HIGH":
            status = g["supplement_review_status"]
            value = g[f"effective_approved_value_{l}"]
            if status == "APPROVE" and not value:
                # The group is approved only for the supported language variants.
                status = "EXCLUDE"
            r["supplement_status"] = status
            r["approved_value"] = value
            r["supplement_scope"] = g["approved_scope"] if value else ""
            r["supplement_review_provenance"] = "USER_RECORDED_WITH_AI_ASSISTANCE"
            if value and not g[f"confirmed_value_{l}"]:
                r["supplement_review_provenance"] += "; AI_EQUIVALENT_SOURCE_VALUE"
            if g["supplement_review_status"] == "APPROVE" and not value:
                r["supplement_review_note"] = "해당 언어의 AI Hub 원문은 공식 주차 가능 정보와 불일치하여 제외."
            else:
                r["supplement_review_note"] = g["manual_note"]
            r["verification_source"] = g["effective_verification_source"]
        elif r["tourapi_final_status"] == "PRESENT":
            r["supplement_status"] = "NO_GAP"
        elif r["aihub_value_status"] == "VALUE_CANDIDATE":
            r["supplement_status"] = "CANDIDATE_UNVERIFIED"
        else:
            r["supplement_status"] = "NO_CONCRETE_VALUE"
        assert r["supplement_status"] != "APPROVE" or (r["approved_value"] and r["verification_source"] and r["tourapi_final_status"] in {"MISSING", "PARTIAL"})
        final.append(r)

    assert changed == 1
    assert sum(r["supplement_status"] == "APPROVE" for r in final) == 11
    assert len({(r["place_id"], r["language"], r["field"]) for r in final}) == 2070
    names = ("common_gap_manual_review_final.csv", "common_gap_review_final.csv")
    assert all(not (ROOT / name).exists() for name in names), "Output already exists; no overwrite."
    write_new(names[0], cols + manual_extra, review_rows)
    write_new(names[1], gap_cols + gap_extra, final)
    print("Human-recorded groups:", Counter(r["supplement_review_status"] for r in review_rows if r["review_priority"] == "HIGH"))
    print("AI rechecked labels: 15; changed:", changed)
    print("Presence labels:", Counter(r["tourapi_final_status"] for r in final))
    print("Supplement statuses:", Counter(r["supplement_status"] for r in final))
    print("Approved:", [(r["tourapi_title"], r["language"], r["field"], r["approved_value"]) for r in final if r["supplement_status"] == "APPROVE"])


if __name__ == "__main__":
    main()
