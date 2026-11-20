"""Record source-versus-output inspection of selected translation examples."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tourapi_full_translation'
NOTES={
 ('125724','en',2):('MAJOR_ERROR','임진왜란을 Great Depression(대공황)으로 오역하고 고씨 표기도 왜곡'),
 ('125724','ja',2):('MAJOR_ERROR','임진왜란과 고씨 가족 피난 서술을 쇼와 연도·일본 도시 개축 이야기로 잘못 생성'),
 ('126998','en',1):('MAJOR_ERROR','정자·인명·연산군 정보를 왜곡하고 인물의 후손 및 군대 표현으로 번역'),
 ('126998','ja',1):('MAJOR_ERROR','정자를 후손으로 번역하고 고유명사 왜곡 및 한국어 글자 혼입'),
 ('126998','en',14):('MAJOR_ERROR','전통 건축의 상부 구조를 배·바지선으로 오역'),
 ('126998','ja',14):('MAJOR_ERROR','전통 건축 용어의 의미를 보존하지 못하고 한국어 글자가 혼입'),
 ('2605528','en',9):('PARTIAL_ERROR','교복이 old costumes로 일반화되고 달고나·딱지 등 일부 문화 표현 오역'),
 ('2605528','ja',9):('PARTIAL_ERROR','옛 교복을 古い服로 일반화하고 달고나·딱지 표현 오역'),
 ('127074','en',6):('PARTIAL_ERROR','회색 벽돌·2층·대칭은 전달하나 설교단과 건축 배치 번역 부정확'),
 ('127074','ja',6):('PARTIAL_ERROR','회색 벽돌·2층·대칭은 대체로 전달하나 설교단 관련 표현 부자연스러움'),
 ('2704694','en',5):('ADEQUATE_FEATURE','해발 107m와 30분 정보 보존; 문장은 다소 부자연스러움'),
 ('2704694','en',6):('ADEQUATE_FEATURE','어린이부터 노인까지 부담이 적은 길이라는 핵심 정보 보존'),
 ('2704694','ja',5):('PARTIAL_ERROR','30분 걷는 정보는 보존하나 해발 107m 누락'),
 ('2704694','ja',6):('ADEQUATE_FEATURE','어린이부터 고령자까지 부담 없는 길이라는 핵심 정보 보존'),
}

with (OUT/'translation_segments.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
selected=[]
for r in rows:
    key=(r['place_id'],r['language'],int(r['segment_index']))
    if key in NOTES:
        a,b=NOTES[key]
        selected.append(dict(**r,ai_spotcheck_status=a,ai_spotcheck_note=b,review_scope='Purposefully selected examples; not a random sample or estimated error rate'))
assert len(selected)==len(NOTES)
with (OUT/'translation_quality_spotcheck.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
