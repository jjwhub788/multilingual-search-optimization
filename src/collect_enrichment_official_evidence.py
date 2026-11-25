"""Collect the publicly displayed KTO detail fields for screened common POIs.

The Korean VISITKOREA page resolves TourAPI content IDs to public content UUIDs
and requests the detail record through its ordinary public frontend. We use the
same two read-only requests, with a delay; no API secret, login, or personal
location is required. We retain only tourism metadata used in the research.
"""

import csv
import json
import re
import time
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"
BASE = "https://korean.visitkorea.or.kr"
DELAY = 0.4


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "li", "tr", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "li", "tr", "div"}:
            self.parts.append("\n")


def plain(value):
    if value is None:
        return ""
    parser = PlainText()
    parser.feed(str(value))
    return " | ".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())


def public_call(cmd, params, referer):
    time.sleep(DELAY)
    body = urlencode({"cmd": cmd, **params}).encode()
    request = Request(BASE + "/call", data=body, headers={
        "User-Agent": "Mozilla/5.0 (compatible; public-tourism-research)",
        "Referer": BASE + referer,
        "Content-Type": "application/x-www-form-urlencoded",
    })
    with urlopen(request, timeout=14) as response:
        data = json.load(response)
    if data.get("header", {}).get("process") != "success":
        raise ValueError("Public page did not return a successful record")
    return data.get("body", {})


def collect(pid, title):
    result = {
        "place_id": pid, "title_ko": title, "fetch_status": "", "official_title": "",
        "official_address": "", "source_url": "", "official_homepage": "", "opening_hours": "",
        "holidays": "", "parking": "", "admission_fee": "", "experience": "", "collected_at_kst": "", "error": "",
    }
    try:
        mappings = public_call("CONTENT_COTID_VIEW", {"cid": pid}, f"/detail/content_view.do?cid={pid}").get("result") or []
        if len(mappings) != 1:
            raise ValueError(f"Expected one public ID mapping, found {len(mappings)}")
        cotid = mappings[0]["cotId"]
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", cotid):
            raise ValueError("Unexpected public content ID")
        ref = f"/detail/ms_detail.do?cotid={cotid}"
        result["source_url"] = BASE + ref
        body = public_call("TOUR_CONTENT_BODY_DETAIL", {"cotId": cotid, "locationx": "", "locationy": "", "stampId": ""}, ref)
        d = body.get("detail") or {}
        if str(d.get("cid")) != pid or str(d.get("cotId")) != cotid:
            raise ValueError("Returned record ID does not match the requested attraction")
        if str(d.get("contentStatus")) not in {"0", "2"}:
            raise ValueError("Public page reports an unavailable/non-display record")
        result["official_title"] = plain(d.get("title"))
        result["official_address"] = plain(d.get("addr1"))
        result["opening_hours"] = plain(d.get("useTime"))
        result["holidays"] = plain(d.get("restDate"))
        result["parking"] = plain(d.get("parking"))
        result["experience"] = plain(d.get("expGuide"))
        match = re.search(r'href\s*=\s*["\'](https?://[^"\']+)', str(d.get("homepage") or ""), re.I)
        result["official_homepage"] = unescape(match.group(1)) if match else ""
        fees = []
        for label in ("useFee", "admissionFee", "useCost", "entranceFee", "fee"):
            if d.get(label):
                fees.append(f"{label}: {plain(d[label])}")
        for sub in body.get("subArticle") or []:
            label = plain(sub.get("displayTitle"))
            content = plain(sub.get("contentBody"))
            if label in {"입장료", "시설이용료", "이용요금", "관람료", "체험프로그램", "체험료", "이용료", "사용료"} and content:
                fees.append(f"{label}: {content}")
            if label == "체험프로그램" and content:
                result["experience"] = " | ".join(v for v in (result["experience"], content) if v)
        result["admission_fee"] = " | ".join(fees)
        result["fetch_status"] = "OK"
    except HTTPError as exc:
        result["fetch_status"] = "HTTP_ERROR"
        result["error"] = str(exc.code)
        if exc.code in {401, 403, 429}:
            # Respect access restrictions and rate limiting.
            raise RuntimeError(f"Public site returned {exc.code}; stop collection") from exc
    except (URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        result["fetch_status"] = "UNAVAILABLE"
        result["error"] = str(exc)[:200]
    result["collected_at_kst"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    return result


def main():
    with (ROOT / "common_enrichment_screening.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    sites = {}
    for row in rows:
        if row["screening_status"] == "SOURCE_CONSISTENT":
            sites[row["place_id"]] = row["title_ko"]
    assert len(sites) == 96
    target = ROOT / "common_enrichment_official_evidence.csv"
    assert not target.exists(), "Evidence output exists; do not overwrite."
    records = []
    for i, (pid, title) in enumerate(sites.items(), 1):
        records.append(collect(pid, title))
        if i % 10 == 0 or i == len(sites):
            print(f"{i}/{len(sites)} official attraction records checked", flush=True)
    with target.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print("OK", sum(r["fetch_status"] == "OK" for r in records), "unavailable", sum(r["fetch_status"] != "OK" for r in records))


if __name__ == "__main__":
    main()
