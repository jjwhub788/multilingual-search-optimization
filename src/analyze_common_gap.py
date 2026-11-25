"""Descriptive, paired information-gap analysis for the 138 common POIs.

Counts are based on AI-assisted reviews of the collected descriptions. They
do not claim absence from all TourAPI endpoints, or establish retrieval effects.
Verified enrichment means an AI Hub value agreed with an official source
within its recorded scope, not an independent human audit of every fact.
"""

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "common_gap_analysis"
LANGS = ("ko", "en", "ja")
LANG_KO = {"ko": "한국어", "en": "영어", "ja": "일본어"}
FIELDS = ("experience", "opening_hours", "holidays", "admission_fee", "parking")
FIELD_KO = dict(zip(FIELDS, ("체험", "운영시간", "휴무일", "입장료", "주차")))
LEVEL = {"MISSING": 0, "PARTIAL": 1, "PRESENT": 2}


def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(r):
    return r["place_id"], r["language"], r["field"]


def summarize(rows, verified):
    c = Counter(r["tourapi_final_status"] for r in rows)
    n = len(rows)
    assert n and set(c) <= set(LEVEL)
    gap = c["MISSING"] + c["PARTIAL"]
    v = sum(key(r) in verified for r in rows)
    return dict(n=n, present=c["PRESENT"], partial=c["PARTIAL"], missing=c["MISSING"], gap=gap,
                missing_rate=c["MISSING"]/n, gap_rate=gap/n, verified=v,
                verified_per_gap=v/gap if gap else 0.0)


def compute():
    rows = read("common_gap_review_final.csv")
    final = read("common_verified_enrichment_final.csv")
    screen = read("common_enrichment_screening_final.csv")
    base = read("common_dataset.csv")
    keys = {key(r): r for r in rows}
    verified = {key(r) for r in final}
    places = {r["place_id"] for r in rows}
    assert len(keys) == len(rows) == 138*3*5 == 2070
    assert set(keys) == {(p, l, f) for p in places for l in LANGS for f in FIELDS}
    assert len(verified) == len(final) == 440 and verified <= set(keys)
    assert all(keys[k]["tourapi_final_status"] != "PRESENT" for k in verified)
    assert len(screen) == 736 and verified == {key(r) for r in screen if r["screening_status"] == "VERIFIED"}
    assert len([r for r in base if r["source"] == "tourapi"]) == 414
    detail = [dict(language=l, language_ko=LANG_KO[l], field=f, field_ko=FIELD_KO[f],
                   **summarize([r for r in rows if r["language"] == l and r["field"] == f], verified))
              for f in FIELDS for l in LANGS]
    language = [dict(language=l, language_ko=LANG_KO[l], **summarize([r for r in rows if r["language"] == l], verified)) for l in LANGS]
    field = [dict(field=f, field_ko=FIELD_KO[f], **summarize([r for r in rows if r["field"] == f], verified)) for f in FIELDS]
    pair = []
    for f in FIELDS:
        for l in ("en", "ja"):
            a = [(keys[p, "ko", f]["tourapi_final_status"], keys[p, l, f]["tourapi_final_status"]) for p in places]
            row = dict(field=f, field_ko=FIELD_KO[f], foreign_language=l, foreign_language_ko=LANG_KO[l], n=len(a),
                       ko_higher=sum(LEVEL[x]>LEVEL[y] for x,y in a), foreign_higher=sum(LEVEL[x]<LEVEL[y] for x,y in a), same_status=sum(x==y for x,y in a),
                       ko_present_foreign_gap=sum(x=="PRESENT" and y!="PRESENT" for x,y in a), foreign_present_ko_gap=sum(x!="PRESENT" and y=="PRESENT" for x,y in a),
                       both_present=sum(x==y=="PRESENT" for x,y in a), both_incomplete=sum(x!="PRESENT" and y!="PRESENT" for x,y in a), both_missing=sum(x==y=="MISSING" for x,y in a))
            assert row["ko_higher"]+row["foreign_higher"]+row["same_status"]==138
            assert row["ko_present_foreign_gap"]+row["foreign_present_ko_gap"]+row["both_present"]+row["both_incomplete"]==138
            pair.append(row)
    tri = []
    for f in FIELDS:
        a = [tuple(keys[p,l,f]["tourapi_final_status"] for l in LANGS) for p in places]
        row=dict(field=f, field_ko=FIELD_KO[f], n=len(a), all_present=sum(all(s=="PRESENT" for s in x) for x in a),
                 mixed_present=sum(0<sum(s=="PRESENT" for s in x)<3 for x in a), all_incomplete=sum(all(s!="PRESENT" for s in x) for x in a),
                 all_missing=sum(all(s=="MISSING" for s in x) for x in a), same_status=sum(len(set(x))==1 for x in a))
        assert row["all_present"]+row["mixed_present"]+row["all_incomplete"]==138
        tri.append(row)
    screening = []
    for f in FIELDS:
        c=Counter(r["screening_status"] for r in screen if r["field"]==f)
        screening.append(dict(field=f,field_ko=FIELD_KO[f],pool=sum(c.values()),verified=c["VERIFIED"],hold=c["HOLD"],exclude=c["EXCLUDE"]))
    total=summarize(rows, verified)
    assert sum(s["verified"] for s in detail)==sum(s["verified"] for s in screening)==440
    assert sum(s["n"] for s in detail)==2070
    return dict(total=total,detail=detail,language=language,field=field,pair=pair,tri=tri,screening=screening,
                facts=440,places=len(places),documents=414,verified_places=len({r["place_id"] for r in final}),
                verified_documents=len({(r["place_id"],r["language"]) for r in final}),
                screen_total=dict(Counter(r["screening_status"] for r in screen)),
                method_note="기준: TourAPI에서 수집한 설명문(overview)의 언어별 정보 존재 여부. PRESENT=충분, PARTIAL=일부, MISSING=없음. 불충분=PARTIAL+MISSING. 보완 가능률=공식 출처 대조 성공 440건 중 해당 행 수/불충분 건수. 보완 후 실제 검색효과와는 별개.")


def draw_figures(data):
    from PIL import Image, ImageDraw, ImageFont
    font_path=Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if not font_path.exists():
        raise RuntimeError("Set a Korean-capable font path before rendering figures.")

    def font(size, bold=False):
        return ImageFont.truetype(str(font_path), size, index=2 if bold else 0)

    def tx(draw, xy, text, size=26, fill="#182B3B", bold=False, anchor=None):
        draw.text(xy, str(text), font=font(size,bold), fill=fill, anchor=anchor)

    ink="#182B3B"; muted="#4D6373"; grid="#DCE4E9"; present="#176F80"; partial="#E6AE4D"; missing="#CFD9E0"

    # Figure 1: status distribution, matched 138-site denominator per bar.
    w,h=1800,1170
    im=Image.new("RGB",(w,h),"white"); d=ImageDraw.Draw(im)
    tx(d,(68,48),"TourAPI 설명문의 언어·정보 유형별 정보 상태",46,bold=True)
    tx(d,(70,112),"동일 관광지 138곳 × 3개 언어 × 5개 유형  |  각 막대: 138건",28,muted)
    for x,c,label in ((70,present,"충분(PRESENT)"),(350,partial,"일부(PARTIAL)"),(620,missing,"없음(MISSING)")):
        d.rounded_rectangle((x,170,x+28,198),radius=4,fill=c)
        tx(d,(x+40,166),label,27)
    x0,x1=350,1690; bw=x1-x0; top=248; bar_h=35; delta=46; group_h=169
    for t in (0,25,50,75,100):
        xx=x0+bw*t/100
        d.line((xx,top-8,xx,top+group_h*4+3*delta),fill=grid,width=2)
        tx(d,(xx,top-18),f"{t}%",23,muted,anchor="mb")
    lookup={(r["field"],r["language"]):r for r in data["detail"]}
    for i,f in enumerate(FIELDS):
        yt=top+i*group_h
        tx(d,(70,yt+47),FIELD_KO[f],32,bold=True,anchor="lm")
        for j,l in enumerate(LANGS):
            r=lookup[f,l]; y=yt+j*delta
            tx(d,(325,y+bar_h/2),LANG_KO[l],23,muted,anchor="rm")
            left=x0
            for name,color in (("present",present),("partial",partial),("missing",missing)):
                v=r[name]; right=left+bw*v/r["n"]
                if v:
                    d.rectangle((left,y,right,y+bar_h),fill=color)
                    if right-left>=55:
                        tx(d,((left+right)/2,y+bar_h/2),str(v),20,"white" if name=="present" else ink,bold=True,anchor="mm")
                left=right
        if i<4:d.line((70,yt+149,x1,yt+149),fill=grid,width=1)
    tx(d,(70,1110),"막대 안 숫자: 해당 정보 상태의 건수. 조사 범위는 수집한 설명문이며, API의 전체 상세정보가 아님.",25,muted)
    im.save(OUT/"01_language_field_gap.png",dpi=(220,220))

    # Figure 2: verified supplements as a fraction of incomplete items.
    w,h=1800,960
    im=Image.new("RGB",(w,h),"white"); d=ImageDraw.Draw(im)
    tx(d,(68,48),"정보 유형별 AI Hub 보완 가능률",46,bold=True)
    tx(d,(70,113),"공식 출처와 대조된 AI Hub 값 ÷ TourAPI 설명문의 불충분 항목(PARTIAL + MISSING)",28,muted)
    x0,x1=380,1345; bw=x1-x0; top=244; bar_h=62; delta=116
    for t in (0,25,50,75,100):
        xx=x0+bw*t/100
        d.line((xx,top-18,xx,top+4*delta+bar_h+15),fill=grid,width=2)
        tx(d,(xx,top-27),f"{t}%",24,muted,anchor="mb")
    for i,r in enumerate(data["field"]):
        y=top+i*delta; frac=r["verified_per_gap"]
        tx(d,(70,y+bar_h/2),r["field_ko"],33,bold=True,anchor="lm")
        d.rounded_rectangle((x0,y,x1,y+bar_h),radius=8,fill="#EBF0F3")
        if frac>0:d.rounded_rectangle((x0,y,x0+bw*frac,y+bar_h),radius=7,fill=present)
        tx(d,(1370,y+bar_h/2),f'{r["verified"]:,} / {r["gap"]:,}건',26,ink,bold=True,anchor="lm")
        tx(d,(1732,y+bar_h/2),f'{frac*100:.1f}%',29,present,bold=True,anchor="rm")
    tx(d,(70,858),"전체 불충분 정보 1,774건 중 440건(24.8%)이 보완 후보로 공식 정보와 대조됨.",30,bold=True)
    tx(d,(70,906),"보완 가능률은 검색 성능 향상률이 아님. 출처의 최신성·적용 범위는 변동될 수 있음.",25,muted)
    im.save(OUT/"02_verified_enrichment_coverage.png",dpi=(220,220))


def main():
    data=compute()
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/"common_gap_summary_data.json"
    assert not target.exists(), "Refusing to overwrite previous analysis."
    with target.open("x",encoding="utf-8") as f:json.dump(data,f,ensure_ascii=False,indent=2)
    # A flat table for anyone who prefers CSV to a report workbook.
    with (OUT/"common_gap_by_language_field.csv").open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(data["detail"][0]));w.writeheader();w.writerows(data["detail"])
    draw_figures(data)
    print("total",data["total"])
    print("three-language group totals",{k:sum(r[k] for r in data["tri"]) for k in ("all_present","mixed_present","all_incomplete","all_missing","same_status")})
    print("pair totals",{l:{k:sum(r[k] for r in data["pair"] if r["foreign_language"]==l) for k in ("ko_higher","foreign_higher","same_status","ko_present_foreign_gap","foreign_present_ko_gap","both_missing")} for l in ("en","ja")})
    print("verified places / docs",data["verified_places"],data["verified_documents"])


if __name__ == "__main__":
    main()
