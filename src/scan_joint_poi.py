"""Read-only scan of TourAPI Korean records against bilingual AI Hub POIs.

Writes separate review lists, never changes source data or assigns final
gold IDs. Only unique, exact-name and two-address-core agreements are
automatically shortlisted as high-confidence candidates.
"""

import argparse
import csv
import math
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "joint_poi_scan"

csv.field_size_limit(10**7)

REGIONS = {
    "서울특별시": "서울", "서울시": "서울", "서울": "서울",
    "부산광역시": "부산", "부산시": "부산", "부산": "부산",
    "대구광역시": "대구", "대구시": "대구", "대구": "대구",
    "인천광역시": "인천", "인천시": "인천", "인천": "인천",
    "광주광역시": "전남광주", "광주": "전남광주", "전남": "전남광주",
    "전라남도": "전남광주", "전남광주통합특별시": "전남광주",
    "대전광역시": "대전", "대전시": "대전", "대전": "대전",
    "울산광역시": "울산", "울산시": "울산", "울산": "울산",
    "세종특별자치시": "세종", "세종시": "세종", "세종": "세종",
    "경기도": "경기", "경기": "경기",
    "강원특별자치도": "강원", "강원도": "강원", "강원": "강원",
    "충청북도": "충북", "충북": "충북",
    "충청남도": "충남", "충남": "충남",
    "전북특별자치도": "전북", "전라북도": "전북", "전북": "전북",
    "경상북도": "경북", "경북": "경북",
    "경상남도": "경남", "경남": "경남",
    "제주특별자치도": "제주", "제주도": "제주", "제주": "제주",
}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def text(s):
    return " ".join(unicodedata.normalize("NFKC", str(s or "")).casefold().split())


def name_key(s):
    return re.sub(r"\s+", "", text(s))


def core_name(s):
    # Parenthetical qualifiers and punctuation are only an alternate-name
    # retrieval key, never sufficient for automatic acceptance.
    x = text(s)
    for _ in range(3):
        x = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", "", x)
    return re.sub(r"[^가-힣a-z0-9]", "", x)


def hangul_key(s):
    return re.sub(r"[^가-힣0-9]", "", text(s))


ADDRESS_NUMBER = re.compile(
    r"(?P<street>[가-힣a-z0-9·.\-]+(?:대로|로|길|동|리|가))\s*"
    r"(?P<number>(?:산\s*)?\d+(?:-\d+)?)(?:번지)?(?=\s|[,()]|$)"
)


def address(s):
    """Normalize province labels; retain municipality, street/lot and number.

    A parenthetical locality or trailing establishment name is ignored only
    after a recognized numeric street/lot address. Road vs lot addresses
    are not converted into each other. The combined Gwangju/Jeonnam label
    is compared to older labels only with the municipality retained.
    """
    s = text(s)
    s = re.sub(r"[,;]", " ", s)
    s = " ".join(s.split())
    tokens = s.split()
    if not tokens or tokens[0] not in REGIONS:
        return "", "", ""
    region = REGIONS[tokens[0]]
    rest = " ".join(tokens[1:])
    match = re.match(r"([^\s]+[시군구])(?:\s|$)", rest)
    locality = region + "/" + match.group(1) if match else (
        "세종/세종" if region == "세종" else ""
    )
    # Requiring the administrative context avoids matching the same street
    # and number in different cities. A number alone is not evidence.
    numeric = ADDRESS_NUMBER.search(rest) if locality else None
    key = ""
    if numeric:
        prefix = rest[:numeric.start("street")] + numeric.group("street")
        key = region + "/" + re.sub(r"\s+", "", prefix) + "/" + re.sub(
            r"\s+", "", numeric.group("number")
        )
    return region, locality, key


def tour_addresses(r):
    raw = [r["addr1"], r["addr2"], r["addr1"] + " " + r["addr2"]]
    parsed = [address(s) for s in raw if s.strip()]
    return (
        {p[0] for p in parsed if p[0]},
        {p[1] for p in parsed if p[1]},
        {p[2] for p in parsed if p[2]},
    )


def similar_name(a, b):
    if min(len(a), len(b)) < 3:
        return False
    similarity = SequenceMatcher(None, a, b, autojunk=False).ratio()
    return similarity >= .80 or (
        min(len(a), len(b)) >= 4 and (a in b or b in a) and similarity >= .65
    )


def coords(r):
    try:
        x, y = float(r["mapx"]), float(r["mapy"])
        return (x, y) if 124 <= x <= 132 and 33 <= y <= 39.5 else None
    except (ValueError, TypeError):
        return None


def distance_m(a, b):
    x1, y1, x2, y2 = map(math.radians, (*a, *b))
    h = math.sin((y2-y1)/2)**2 + math.cos(y1)*math.cos(y2)*math.sin((x2-x1)/2)**2
    return 6371000*2*math.asin(min(1, math.sqrt(h)))


def extend_trilingual(tour, existing):
    """Find new *review* candidates using explicit Korean aliases and coordinates."""
    foreign = {lang: defaultdict(list) for lang in ("en", "ja")}
    used = {lang: {r["id_"+lang]: r["id_ko"] for r in existing.values()}
            for lang in ("en", "ja")}
    for r in tour:
        lang = r["language"]
        if lang in foreign and coords(r):
            aliases = {hangul_key(s) for s in re.findall(r"\(([^()]*)\)", text(r["title"]))
                       if re.search(r"[가-힣]", s)}
            for alias in aliases-{""}:
                foreign[lang][alias].append(r)
    new = {}
    for r in tour:
        if r["language"] != "ko" or r["source_contentid"] in existing or not coords(r):
            continue
        kid = r["source_contentid"]
        alias = hangul_key(r["title"])
        if not alias or not re.search(r"[가-힣]", alias):
            continue
        matches = {}
        for lang in ("en", "ja"):
            candidates = []
            for f in foreign[lang].get(alias, []):
                if f["source_contentid"] in used[lang]:
                    continue
                d = distance_m(coords(r), coords(f))
                if d <= 100:
                    candidates.append((f["source_contentid"], d))
            if len(candidates) == 1:
                matches[lang] = candidates[0]
        if len(matches) == 2:
            new[kid] = matches
    # A foreign record proposed for multiple Korean records remains ambiguous.
    counts = {lang: Counter(m[lang][0] for m in new.values()) for lang in ("en", "ja")}
    return {kid: m for kid, m in new.items() if all(counts[l][m[l][0]] == 1 for l in counts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aihub-poi", type=Path)
    args = parser.parse_args()
    paths = list(DATA.glob("aihub_poi*.csv"))
    if args.aihub_poi:
        aihub_path = args.aihub_poi
    elif len(paths) == 1:
        aihub_path = paths[0]
    else:
        parser.error("Specify --aihub-poi with the intended file.")
    aihub = read(aihub_path)
    tour = read(DATA / "tourapi_clean.csv")
    ko = [r for r in tour if r["language"] == "ko"]
    pilot = {r["id_ko"] for r in read(DATA / "tourapi_pilot_100.csv")}
    tri = {r["id_ko"]: r for r in read(DATA / "tourapi_final_matches.csv")}
    held = {r["id_ko"] for r in read(DATA / "tourapi_sample_review_20.csv")
            if r["sample_review_status"] != "MATCH"}
    assert len(ko) == len({r["source_contentid"] for r in ko}) == 12581
    assert len(pilot) == 100 and len(tri) == 331
    assert len(aihub) == len({r["poi_id"] for r in aihub})

    paired = [r for r in aihub if r["has_en"].lower() == r["has_ja"].lower() == "true"
              and name_key(r["title_ko_en"]) and
              name_key(r["title_ko_en"]) == name_key(r["title_ko_ja"])]
    print(f"AI Hub {len(aihub):,} POIs; {len(paired):,} bilingual, consistent Korean names", flush=True)
    by_name, by_alias, by_address = defaultdict(set), defaultdict(set), defaultdict(set)
    for i, a in enumerate(paired):
        a["_name"] = name_key(a["title_ko_en"])
        a["_alias"] = core_name(a["title_ko_en"])
        a["_addr_en"] = address(a["address_ko_en"])
        a["_addr_ja"] = address(a["address_ko_ja"])
        by_name[a["_name"]].add(i)
        if len(a["_alias"]) >= 3:
            by_alias[a["_alias"]].add(i)
        for key in {a["_addr_en"][2], a["_addr_ja"][2]}-{""}:
            by_address[key].add(i)
    new_tri = extend_trilingual(tour, tri)
    tour_lookup = {(r["language"], r["source_contentid"]): r for r in tour}
    print(f"TourAPI: {len(new_tri):,} new internal multilingual review candidates beyond 331", flush=True)

    pairs = []
    for t in ko:
        kid = t["source_contentid"]
        name, alias = name_key(t["title"]), core_name(t["title"])
        regions, localities, keys = tour_addresses(t)
        possible = set(by_name[name])
        if len(alias) >= 3:
            possible.update(by_alias[alias])
        for key in keys:
            possible.update(i for i in by_address[key] if similar_name(alias, paired[i]["_alias"]))
        for i in sorted(possible):
            a = paired[i]
            name_evidence = "EXACT" if a["_name"] == name else (
                "ALIAS_CORE" if a["_alias"] == alias else "SIMILAR_NAME_SAME_ADDRESS"
            )
            en, ja = a["_addr_en"], a["_addr_ja"]
            both_numeric = bool(en[2] and ja[2] and en[2] in keys and ja[2] in keys)
            one_numeric = bool(keys & {en[2], ja[2]})
            same_locality = bool(localities & {en[1], ja[1]})
            ai_regions = {en[0], ja[0]}-{""}
            ai_localities = {en[1], ja[1]}-{""}
            conflict = (
                bool(regions and ai_regions and not regions & ai_regions)
                or bool(localities and ai_localities and not localities & ai_localities)
            )
            evidence = (
                "BOTH_NUMERIC_CORES" if both_numeric else
                "ONE_NUMERIC_CORE" if one_numeric else
                "SAME_MUNICIPALITY" if same_locality else
                "LOCATION_CONFLICT" if conflict else "NO_RELIABLE_ADDRESS_AGREEMENT"
            )
            if both_numeric and name_evidence == "EXACT":
                status = "STRONG_PENDING_UNIQUENESS"
            elif both_numeric or one_numeric:
                status = "REVIEW_NAME_OR_ADDRESS"
            elif same_locality:
                status = "REVIEW_LOCALITY"
            elif conflict:
                status = "REVIEW_LOCATION_CONFLICT"
            else:
                status = "REVIEW_NAME_ONLY"
            if kid in tri:
                link_status = "EXISTING_331_FLAGGED" if kid in held else "EXISTING_331"
                en_id, ja_id = tri[kid]["id_en"], tri[kid]["id_ja"]
                en_dist, ja_dist = "", ""
            elif kid in new_tri:
                link_status = "NEW_ALIAS_COORD_REVIEW"
                en_id, en_dist = new_tri[kid]["en"]
                ja_id, ja_dist = new_tri[kid]["ja"]
                en_dist, ja_dist = round(en_dist, 1), round(ja_dist, 1)
            else:
                link_status = "NOT_LINKED_THREE_LANGUAGES"
                en_id, ja_id, en_dist, ja_dist = "", "", "", ""
            pairs.append(dict(
                match_status=status, tourapi_ko_id=kid, aihub_poi_id=a["poi_id"],
                tourapi_title_ko=t["title"], aihub_title_ko_en=a["title_ko_en"],
                aihub_title_ko_ja=a["title_ko_ja"], tourapi_address1=t["addr1"],
                tourapi_address2=t["addr2"], aihub_address_ko_en=a["address_ko_en"],
                aihub_address_ko_ja=a["address_ko_ja"], name_evidence=name_evidence,
                address_evidence=evidence, tourapi_numeric_address_keys="|".join(sorted(keys)),
                aihub_numeric_address_en=en[2], aihub_numeric_address_ja=ja[2],
                pilot_100="YES" if kid in pilot else "NO",
                tourapi_trilingual_status=link_status, tourapi_en_id=en_id, tourapi_ja_id=ja_id,
                tourapi_title_en=tour_lookup.get(("en", en_id), {}).get("title", ""),
                tourapi_title_ja=tour_lookup.get(("ja", ja_id), {}).get("title", ""),
                tourapi_address_en=tour_lookup.get(("en", en_id), {}).get("addr1", ""),
                tourapi_address_ja=tour_lookup.get(("ja", ja_id), {}).get("addr1", ""),
                new_en_coord_distance_m=en_dist, new_ja_coord_distance_m=ja_dist,
                existing_three_descriptions="YES" if kid in pilot else "NO",
                aihub_title_en=a["title_en"], aihub_title_ja=a["title_ja"],
                aihub_two_translated_names="YES" if a["title_en"].strip() and a["title_ja"].strip() else "NO",
                aihub_two_translated_attributes="YES" if a["attributes_en"].strip() and a["attributes_ja"].strip() else "NO",
                aihub_source_split_en=a["source_split_en"], aihub_source_split_ja=a["source_split_ja"],
                aihub_cross_split=a["cross_split"],
                team_final_decision="", team_review_note="",
            ))
    strong = [r for r in pairs if r["match_status"] == "STRONG_PENDING_UNIQUENESS"]
    kt = Counter(r["tourapi_ko_id"] for r in strong)
    at = Counter(r["aihub_poi_id"] for r in strong)
    for r in strong:
        r["match_status"] = (
            "HIGH_CONFIDENCE_CANDIDATE" if kt[r["tourapi_ko_id"]] == at[r["aihub_poi_id"]] == 1
            else "REVIEW_DUPLICATE_STRONG"
        )
    high = [r for r in pairs if r["match_status"] == "HIGH_CONFIDENCE_CANDIDATE"]
    assert len(high) == len({r["tourapi_ko_id"] for r in high}) == len({r["aihub_poi_id"] for r in high})
    # Both numeric address keys must be present in TourAPI's normalized keys.
    assert all(r["name_evidence"] == "EXACT" and
               r["address_evidence"] == "BOTH_NUMERIC_CORES" for r in high)
    order = {"HIGH_CONFIDENCE_CANDIDATE": 0, "REVIEW_DUPLICATE_STRONG": 1,
             "REVIEW_NAME_OR_ADDRESS": 2, "REVIEW_LOCALITY": 3,
             "REVIEW_NAME_ONLY": 4, "REVIEW_LOCATION_CONFLICT": 5}
    pairs.sort(key=lambda r: (order[r["match_status"]],
                              r["tourapi_trilingual_status"] not in ("EXISTING_331", "NEW_ALIAS_COORD_REVIEW"),
                              r["tourapi_ko_id"], r["aihub_poi_id"]))
    high = [r for r in pairs if r["match_status"] == "HIGH_CONFIDENCE_CANDIDATE"]
    write(OUT / "joint_candidates_all.csv", pairs)
    write(OUT / "joint_candidates_high_confidence.csv", high)
    tri_high = [r for r in high if r["tourapi_trilingual_status"] in (
        "EXISTING_331", "NEW_ALIAS_COORD_REVIEW"
    )]
    tri_high.sort(key=lambda r: (r["existing_three_descriptions"] != "YES",
                                 r["tourapi_trilingual_status"] != "EXISTING_331",
                                 r["tourapi_ko_id"]))
    write(OUT / "joint_trilingual_high_candidates.csv", tri_high, list(pairs[0]))
    tri_review = [r for r in pairs if r["tourapi_trilingual_status"] in (
        "EXISTING_331", "EXISTING_331_FLAGGED", "NEW_ALIAS_COORD_REVIEW"
    ) and r["match_status"] != "REVIEW_LOCATION_CONFLICT"]
    write(OUT / "joint_trilingual_priority_review.csv", tri_review, list(pairs[0]))

    summary = []
    for scope, ids in (
        ("PILOT_100", pilot),
        ("EXISTING_TRI_331", set(tri)),
        ("EXISTING_OR_NEW_TRI", set(tri) | set(new_tri)),
        ("ALL_KOREAN_12581", {r["source_contentid"] for r in ko}),
    ):
        p = [r for r in pairs if r["tourapi_ko_id"] in ids]
        h = {r["tourapi_ko_id"] for r in p if r["match_status"] == "HIGH_CONFIDENCE_CANDIDATE"}
        review = {r["tourapi_ko_id"] for r in p if r["match_status"] not in
                  ("HIGH_CONFIDENCE_CANDIDATE", "REVIEW_LOCATION_CONFLICT")}-h
        conflict = {r["tourapi_ko_id"] for r in p}-h-review
        row = dict(scope=scope, tourapi_ko_records=len(ids),
                   exact_name_candidates=len({r["tourapi_ko_id"] for r in p if r["name_evidence"]=="EXACT"}),
                   high_confidence_candidates=len(h),
                   additional_review_candidates=len(review), location_conflict_only=len(conflict),
                   no_candidates=len(ids)-len(h)-len(review)-len(conflict), candidate_pairs=len(p))
        summary.append(row)
        print(row, flush=True)
    write(OUT / "joint_match_summary.csv", summary)
    print("High confidence by TourAPI trilingual status:",
          dict(Counter(r["tourapi_trilingual_status"] for r in high)), flush=True)
    print("Existing full descriptions:", sum(r["existing_three_descriptions"]=="YES" for r in high), flush=True)
    print("Review lists saved. No final gold records accepted without team review.", flush=True)


if __name__ == "__main__":
    main()
