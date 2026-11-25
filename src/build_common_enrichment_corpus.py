"""Render verified AI Hub enrichment items in three equivalent forms.

The original TourAPI title/description and corpus membership never change.
Only verified AI Hub facts are used. Labels, presentation sentences and scope
qualifiers are authored renderings, not original TourAPI or AI Hub documents.
The qualification text restricts the scope identified during official checks;
it does not introduce new fees, hours, tourism activities or parking capacity.
The full corpus is an upper-bound reference; a fair fixed-budget comparison
must render exactly the same selected fact IDs in all three conditions.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"
LANGS = ("ko", "en", "ja")
FIELDS = ("experience", "opening_hours", "holidays", "admission_fee", "parking")
MODEL = "BAAI/bge-m3"


LABELS = {
    "experience": ("체험 프로그램", "Activities", "体験プログラム"),
    "opening_hours": ("운영시간", "Operating hours", "営業時間"),
    "holidays": ("정기휴무일", "Regular closing days", "定休日"),
    "admission_fee": ("일반 개인 입장료", "Individual admission fees", "一般個人入場料"),
    "parking": ("주차", "Parking", "駐車"),
}

# The most important qualifiers are encoded in the field label itself. Do not
# convert a facility-specific operating time into a whole-area operating time.
SCOPE_LABELS = {
    ("2718685", "opening_hours"): ("치유센터 운영시간", "Healing center operating hours", "癒やしセンター営業時間"),
    ("126346", "opening_hours"): ("사찰 일반 이용시간", "Temple visiting hours", "寺院の一般参拝時間"),
    ("2668973", "opening_hours"): ("전망대 운영시간", "Observatory operating hours", "展望台営業時間"),
    ("2436233", "opening_hours"): ("수목원 일반 운영시간", "Arboretum operating hours", "樹木園営業時間"),
    ("2633896", "opening_hours"): ("일반 운영시간", "General operating hours", "通常営業時間"),
    ("2764342", "opening_hours"): ("온천랜드 일반 운영시간", "Hot spring facility operating hours", "温泉施設営業時間"),
    ("125266", "opening_hours"): ("당일 방문객 이용시간", "Day visitor hours", "日帰り利用時間"),
    ("1607565", "opening_hours"): ("수목원 일반 운영시간", "Arboretum operating hours", "樹木園通常営業時間"),
    ("126528", "opening_hours"): ("성지 일반 관람시간", "Shrine visiting hours", "聖地の一般見学時間"),
    ("2718685", "holidays"): ("치유센터 정기휴무일", "Healing center closing days", "癒やしセンター定休日"),
    ("2554070", "holidays"): ("투어 운영시설 정기휴무일", "Tour facility closing days", "ツアー関連施設の定休日"),
    ("769761", "holidays"): ("분수 운영기간 정기휴무일", "Closing day during fountain operating season", "噴水の運営期間中の定休日"),
    ("126508", "admission_fee"): ("일반 성인 입장료", "Standard adult admission fee", "一般大人入場料"),
    ("127539", "admission_fee"): ("자유이용권 요금", "All-ride pass fees", "フリーパス料金"),
    ("126581", "admission_fee"): ("일반·단체 입장료", "Individual and group admission fees", "一般・団体入場料"),
    ("2563697", "admission_fee"): ("리프트·모노레일 왕복 패키지 요금", "Round-trip lift and monorail package fees", "リフト・モノレール往復セット料金"),
    ("2668973", "parking"): ("전망대 방문객 주차", "Observatory visitor parking", "展望台利用者の駐車"),
    ("2743288", "parking"): ("선착장 주차", "Pier parking", "船着場の駐車"),
    ("1899635", "parking"): ("개방기간 주차", "Parking during opening period", "公開期間中の駐車"),
}

for pid in ("2500687", "1059479", "970460", "127075", "126825", "126487"):
    SCOPE_LABELS[pid, "opening_hours"] = ("공원 개방시간", "Park opening hours", "公園開放時間")
SCOPE_LABELS["2774063", "opening_hours"] = ("거리 개방시간", "Street access hours", "通りの開放時間")
SCOPE_LABELS["2733962", "opening_hours"] = ("계곡 개방시간", "Valley access hours", "渓谷開放時間")
SCOPE_LABELS["2027193", "opening_hours"] = ("사찰 개방시간", "Temple access hours", "寺院開放時間")

# Qualifiers describe applicability, not additional numeric/operational facts.
# The identical phrase appears in both natural and key-value renderings.
QUALIFIERS = {
    ("1329201", "opening_hours"): ("계절에 따라 달라질 수 있습니다.", "Hours may vary by season.", "季節によって変わる場合があります。"),
    ("2436233", "opening_hours"): ("계절과 기상에 따라 달라질 수 있습니다.", "Hours may vary by season and weather.", "季節や天候により変わる場合があります。"),
    ("2633896", "opening_hours"): ("개별 시설 운영시간은 별도입니다.", "Individual facilities have separate hours.", "個別施設の営業時間は別です。"),
    ("2764342", "opening_hours"): ("야외 수영장 운영시간은 별도입니다.", "The outdoor pool has separate hours.", "屋外プールの営業時間は別です。"),
    ("126484", "opening_hours"): ("입장 마감은 별도입니다.", "Last admission has a separate cutoff.", "最終入場時刻は別です。"),
    ("2831210", "opening_hours"): ("매표 마감은 별도입니다.", "Ticket sales have a separate cutoff.", "チケット販売終了時刻は別です。"),
    ("1607565", "opening_hours"): ("야간 특별 개장시간은 별도입니다.", "Special evening opening hours are separate.", "特別な夜間開園時間は別です。"),
    ("128961", "opening_hours"): ("개별 시설의 이용 제한은 별도입니다.", "Individual facility restrictions are separate.", "個別施設の利用制限は別です。"),
    ("126508", "holidays"): ("공휴일에는 예외가 있습니다.", "Public holiday exceptions apply.", "祝日には例外があります。"),
    ("127817", "holidays"): ("공휴일에는 예외가 있습니다.", "Public holiday exceptions apply.", "祝日には例外があります。"),
    ("2481866", "holidays"): ("공휴일에는 예외가 있습니다.", "Public holiday exceptions apply.", "祝日には例外があります。"),
    ("126747", "holidays"): ("공휴일에는 예외가 있습니다.", "Public holiday exceptions apply.", "祝日には例外があります。"),
    ("127740", "holidays"): ("운영 상황에 따라 변경될 수 있습니다.", "Closures may change according to operating conditions.", "運営状況により変更される場合があります。"),
    ("2594360", "holidays"): ("추가 휴무일이 있을 수 있습니다.", "Additional closing days may apply.", "別途休業日がある場合があります。"),
    ("126581", "admission_fee"): ("군경·단체 요금은 적용 조건이 있습니다.", "Military, police and group rates have eligibility conditions.", "軍人・警察官および団体料金には適用条件があります。"),
    ("1899635", "parking"): ("현장 안내에 따라 이용합니다.", "Follow on-site parking guidance.", "現地の駐車案内に従って利用します。"),
}

# Explicit audience/price formatting of the seven verified fee groups. These
# are the supplied AI Hub amounts, not newly sourced replacement prices.
FEE_VALUES = {
    "126508": ("3,000원", "KRW 3,000", "3,000ウォン"),
    "127539": (
        "성인 32,000원, 어린이 22,000원",
        "KRW 32,000 for adults and KRW 22,000 for children",
        "大人32,000ウォン、子ども22,000ウォン",
    ),
    "126581": (
        "일반 11,000원, 청소년·군경 8,000원, 어린이 5,000원, 청소년 단체 6,000원, 어린이 단체 4,000원",
        "KRW 11,000 for adults, KRW 8,000 for teenagers, military and police, KRW 5,000 for children, KRW 6,000 for teenage groups and KRW 4,000 for children's groups",
        "一般11,000ウォン、青少年・軍人・警察8,000ウォン、子ども5,000ウォン、青少年団体6,000ウォン、子ども団体4,000ウォン",
    ),
    "2563697": (
        "대인 17,000원, 소인 14,000원",
        "KRW 17,000 for adults and KRW 14,000 for children",
        "大人17,000ウォン、子ども14,000ウォン",
    ),
    "126668": (
        "성인 11,000원, 청소년 8,500원, 어린이 7,500원",
        "KRW 11,000 for adults, KRW 8,500 for teenagers and KRW 7,500 for children",
        "大人11,000ウォン、青少年8,500ウォン、子ども7,500ウォン",
    ),
    "125266": (
        "성인 1,000원, 어린이 300원",
        "KRW 1,000 for adults and KRW 300 for children",
        "大人1,000ウォン、子ども300ウォン",
    ),
    "127740": (
        "성인 7,000원, 어린이 5,000원",
        "KRW 7,000 for adults and KRW 5,000 for children",
        "大人7,000ウォン、子ども5,000ウォン",
    ),
}


def read_csv(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    assert rows and not path.exists(), f"Refusing to overwrite {path}"
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8-sig", newline="") as f:
        assert list(csv.DictReader(f)) == rows


def ko_topic(label):
    last = ord(label[-1])
    return label + ("은" if 0xAC00 <= last <= 0xD7A3 and (last - 0xAC00) % 28 else "는")


def number_sequence(value):
    return tuple(int(x) for x in re.findall(r"\d+", re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", value)))


def canonical_item(row):
    pid, lang, field = row["place_id"], row["language"], row["field"]
    i = LANGS.index(lang)
    key = pid, field
    raw = row["candidate_value"]
    label = SCOPE_LABELS.get(key, LABELS[field])[i]
    qualifier = QUALIFIERS.get(key, ("", "", ""))[i]
    if field == "parking":
        assert raw.strip().casefold() in {"있음", "yes", "有り", "available"}, (key, raw)
        value = ("가능", "Available", "利用可")[i]
        if lang == "ko":
            main = f"{label}가 가능합니다."
        elif lang == "en":
            main = f"{label} is available."
        else:
            main = f"{label}が可能です。"
    elif field == "opening_hours":
        h = re.fullmatch(r"\s*(\d{2}:\d{2})\s*(?:~|-|to|～|–)\s*(\d{2}:\d{2})\s*", raw, re.I)
        assert h, (key, raw)
        if h.groups() == ("00:00", "24:00"):
            value = ("24시간", "24 hours", "24時間")[i]
        else:
            value = ("~", "–", "～")[i].join(h.groups())
        if lang == "ko":
            main = f"{ko_topic(label)} {value}입니다."
        elif lang == "en":
            main = f"{label} are {value}."
        else:
            main = f"{label}は{value}です。"
    elif field == "holidays":
        if pid == "2718685":
            value = ("매주 토요일·일요일 및 공휴일", "every Saturday and Sunday and on public holidays", "毎週土曜日・日曜日と祝日")[i]
        elif pid == "127740":
            value = ("매주 월요일 및 공휴일", "every Monday and on public holidays", "毎週月曜日と祝日")[i]
        elif pid in {"126508", "3072021"}:
            value = ("매주 화요일", "every Tuesday", "毎週火曜日")[i]
        else:
            value = ("매주 월요일", "every Monday", "毎週月曜日")[i]
        if lang == "ko":
            main = f"{ko_topic(label)} {value}입니다."
        elif lang == "en":
            subject = {"2718685": "The healing center", "2554070": "The tour facilities", "769761": "The fountain during its operating season"}.get(pid, "The site")
            main = f"{subject} {'are' if pid == '2554070' else 'is'} regularly closed {value}."
        else:
            main = f"{label}は{value}です。"
    elif field == "admission_fee":
        value = FEE_VALUES[pid][i]
        assert number_sequence(raw) == number_sequence(value), (key, raw, value)
        if lang == "ko":
            main = f"{ko_topic(label)} {value}입니다."
        elif lang == "en":
            main = f"The {label.lower()} {'is' if pid == '126508' else 'are' if label.endswith('fees') else 'is'} {value}."
        else:
            main = f"{label}は{value}です。"
    elif field == "experience":
        if pid in {"126346", "125891"}:
            value = ("템플스테이", "Temple stay program", "テンプルステイ")[i]
            main = ("템플스테이를 체험할 수 있습니다.", "Visitors can join a temple stay program.", "テンプルステイを体験できます。")[i]
        elif pid == "2554070":
            value = ("해상투어, 마을투어, 깡깡이 통합투어, 유람선", "Sea tour, village tour, Kkangkangi integrated tour and cruise ship", "海上ツアー、村ツアー、カンカンイ統合ツアー、遊覧船")[i]
            main = (f"{value}을 이용할 수 있습니다.", f"Visitors can enjoy the {value.lower()}.", f"{value}を楽しめます。")[i]
        else:
            raise ValueError(key)
    else:
        raise ValueError(field)

    natural = main + ((" " + qualifier) if qualifier else "")
    key_value = f"{label}: {value}" + (f" ({qualifier})" if qualifier else "")
    return label, value, qualifier, natural, key_value


def build_item_pool(verified_rows, tokenizer):
    out = []
    for row in verified_rows:
        pid, lang, field = row["place_id"], row["language"], row["field"]
        assert row["eligible_for_enrichment"] == "YES" and row["screening_status"] == "VERIFIED"
        assert row["tourapi_status"] in {"PARTIAL", "MISSING"}
        label, value, qualifier, natural, key_value = canonical_item(row)
        tokens = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
        fact_id = f"{pid}:{lang}:{field}"
        structured = {"fact_id": fact_id, "field": field, "label": label, "value": value, "qualifier": qualifier, "field_text": key_value}
        out.append({
            "fact_id": fact_id, "place_id": pid, "language": lang, "title_ko": row["title_ko"],
            "information_type": row["information_type"], "field": field, "tourapi_status": row["tourapi_status"],
            "aihub_raw_value": row["aihub_raw_value"], "verified_aihub_value": row["candidate_value"],
            "canonical_label": label, "canonical_value": value, "scope_qualifier": qualifier,
            "natural_text": natural, "key_value_text": key_value,
            "structured_field_json": json.dumps(structured, ensure_ascii=False, separators=(",", ":")),
            "item_cost": "1", "canonical_budget_tokens": str(tokens(key_value)),
            "natural_tokens": str(tokens(natural)), "key_value_tokens": str(tokens(key_value)),
            "verification_source": row["verification_source"], "verification_scope_note": row["scope_note"],
            "source_check_method": row["source_check_method"],
            "render_provenance": "TEMPLATE_RENDERED_FROM_VERIFIED_AIHUB_VALUE_WITH_VERIFIED_SCOPE",
        })
    return out


def render_corpus(baseline, items, tokenizer, selected=None):
    by_doc = defaultdict(list)
    for item in items:
        if selected is None or item["fact_id"] in selected:
            by_doc[item["place_id"], item["language"]].append(item)
    result = []
    for doc in baseline:
        key = doc["place_id"], doc["language"]
        facts = sorted(by_doc[key], key=lambda x: FIELDS.index(x["field"]))
        base = doc["description"]
        natural = base + ("\n\n" + " ".join(f["natural_text"] for f in facts) if facts else "")
        kv = base + ("\n\n" + "\n".join(f["key_value_text"] for f in facts) if facts else "")
        fields = {f["field"]: json.loads(f["structured_field_json"]) for f in facts}
        tokens = lambda s: len(tokenizer.encode(doc["title"] + "\n" + s, add_special_tokens=True))
        counts = (tokens(base), tokens(natural), tokens(kv))
        result.append({
            "source": "tourapi", "place_id": key[0], "language": key[1], "title": doc["title"],
            "description_baseline": base, "description_natural": natural, "description_key_value": kv,
            "structured_fields_json": json.dumps(fields, ensure_ascii=False, separators=(",", ":")),
            "selected_fact_ids": "|".join(f["fact_id"] for f in facts), "supplement_count": str(len(facts)),
            "canonical_budget_tokens": str(sum(int(f["canonical_budget_tokens"]) for f in facts)),
            "baseline_input_tokens": str(counts[0]), "natural_input_tokens": str(counts[1]), "key_value_input_tokens": str(counts[2]),
            "exceeds_1024_tokens": "YES" if max(counts) > 1024 else "NO",
            "condition": "FULL_AVAILABLE_REFERENCE" if selected is None else "SELECTED_FACTS",
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, help="Optional CSV containing fact_id; render this fixed subset in every condition")
    parser.add_argument("--strategy", choices=("random", "deficit", "intent"), help="Filter a multi-strategy selection manifest")
    parser.add_argument("--output", type=Path, help="Required for a specific selection; the original full corpus is not overwritten")
    args = parser.parse_args()
    if args.selection and not args.output:
        parser.error("--output is required with --selection")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    verified_rows = read_csv("common_verified_enrichment_final.csv")
    base = [r for r in read_csv("common_dataset.csv") if r["source"] == "tourapi"]
    assert len(verified_rows) == 440 and len(base) == 414
    assert all(r["status"] == "COMPLETE" and r["description"] for r in base)
    assert {r["language"] for r in base} == set(LANGS)
    keys = {(r["place_id"], r["language"]) for r in base}
    assert len(keys) == 414
    items = build_item_pool(verified_rows, tokenizer)
    all_ids = {r["fact_id"] for r in items}
    assert len(items) == len(all_ids) == 440
    assert all((r["place_id"], r["language"]) in keys for r in items)
    selected = None
    if args.selection:
        with args.selection.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if "strategy" in reader.fieldnames and not args.strategy:
                parser.error("Supply --strategy to select one strategy from the manifest")
            selected_rows = [r["fact_id"] for r in reader if not args.strategy or r.get("strategy") == args.strategy]
        assert selected_rows, "No items selected"
        selected = set(selected_rows)
        assert len(selected) == len(selected_rows) and selected <= all_ids
    result = render_corpus(base, items, tokenizer, selected)
    expected_ids = all_ids if selected is None else selected
    assert len(result) == len(base) == 414
    assert sum(int(r["supplement_count"]) for r in result) == len(expected_ids)
    observed_ids = {fid for r in result for fid in r["selected_fact_ids"].split("|") if fid}
    assert expected_ids == observed_ids
    assert all(r["description_baseline"] == d["description"] and r["title"] == d["title"] for r, d in zip(result, base))
    assert all(r["description_natural"].startswith(r["description_baseline"]) and r["description_key_value"].startswith(r["description_baseline"]) for r in result)
    assert all(len(json.loads(r["structured_fields_json"])) == int(r["supplement_count"]) for r in result)
    assert all(r["description_natural"] == r["description_baseline"] == r["description_key_value"] for r in result if r["supplement_count"] == "0")
    item_by_id = {r["fact_id"]: r for r in items}
    for doc in result:
        for field, fact in json.loads(doc["structured_fields_json"]).items():
            item = item_by_id[fact["fact_id"]]
            assert field == item["field"] and fact["field_text"] == item["key_value_text"]
            assert item["natural_text"] in doc["description_natural"] and item["key_value_text"] in doc["description_key_value"]
            assert fact["value"] == item["canonical_value"] and fact["qualifier"] == item["scope_qualifier"]
    if args.selection:
        write_csv(args.output, result)
    else:
        targets = (ROOT / "common_enrichment_item_pool.csv", ROOT / "common_enrichment_corpus_variants.csv")
        assert not any(t.exists() for t in targets)
        write_csv(targets[0], items)
        write_csv(targets[1], result)
    print("Items:", len(expected_ids), "documents:", len(result), "changed:", sum(r["supplement_count"] != "0" for r in result))
    print("By field:", dict(Counter(r["field"] for r in items if r["fact_id"] in expected_ids)))
    print("Max input tokens (title + description):", [max(int(r[c]) for r in result) for c in ("baseline_input_tokens", "natural_input_tokens", "key_value_input_tokens")])
    print("Documents over 1024:", sum(r["exceeds_1024_tokens"] == "YES" for r in result))


if __name__ == "__main__":
    main()
