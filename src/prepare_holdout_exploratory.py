"""Create AI-written held-out-place query drafts; human review remains required."""
import csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'data/processed';O=R/'results/common_pilot'
def read(n):
 with (D/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
s={r['place_id']:r['split'] for r in read('common_place_splits_balanced.csv')};d={(r['place_id'],r['language']):r for r in read('common_dataset.csv') if r['source']=='tourapi'}
records=[
('2718685','opening_hours','제천 금수산에서 숲 치유 프로그램을 운영하는 곳의 이용시간','Opening hours for a forest therapy center near Geumsusan in Jecheon','堤川の錦繡山で森林療法プログラムを行う施設の営業時間'),
('2447115','parking','백두대간의 산림 치유 시설 주차 정보','Parking information for a forest therapy facility in the Baekdudaegan mountains','白頭大幹の森林療養施設の駐車情報'),
('2670558','opening_hours','양주에서 천일홍 꽃 축제가 열리는 농원 이용시간','Opening hours for a flower garden in Yangju known for its globe amaranth festival','楊州でセンニチコウの花祭りが開かれる農園の営業時間'),
('2638475','parking','한강대교 중간에 있는 음악 문화공간 주차 정보','Parking at the music and culture space in the middle of Hangang Bridge','漢江大橋の中央にある音楽文化空間の駐車情報'),
('2500687','parking','팔당호 주변 수변 생태공원 주차 정보','Parking information for a waterside ecological park near Paldang Lake','八堂湖周辺にある水辺の生態公園の駐車情報'),
('126498','admission_fee','석촌호수의 실내외 테마파크 입장료','Admission fee for the indoor and outdoor theme park by Seokchon Lake','石村湖にある屋内外テーマパークの入場料金'),
('1329201','opening_hours','제주의 돌과 바람을 주제로 한 미로공원 이용시간','Opening hours for a Jeju maze park themed around stone and wind','済州の石と風をテーマにした迷路公園の営業時間'),
('2555756','parking','대운산 계곡에 있는 산림 치유 시설 주차 정보','Parking information for the forest therapy facility in the Daeunsan valley','大雲山の渓谷にある森林療養施設の駐車情報'),
('126581','admission_fee','거제의 바다로 둘러싸인 섬 식물공원 입장료','Admission fee for an island botanical garden surrounded by the sea in Geoje','巨済の海に囲まれた島の植物公園の入場料金'),
('2743288','opening_hours','선유도에서 고군산군도를 도는 유람선 운영시간','Operating hours for a sightseeing cruise from Seonyudo around the Gogunsan islands','仙遊島から古群山群島を巡る遊覧船の運航時間'),
('128637','parking','경주에 통일신라시대 삼층석탑 두 기가 남은 절터 주차 정보','Parking at a temple site in Gyeongju with two Unified Silla three-story pagodas','慶州の統一新羅時代の三重石塔が二基残る寺跡の駐車情報'),
('125891','experience','계룡산 국립공원 사찰의 체험 프로그램','Experience programs at a temple in Gyeryongsan National Park','鶏龍山国立公園にある寺院の体験プログラム'),
('2569511','parking','경기 감악산 골짜기를 잇는 출렁다리 주차 정보','Parking information for a suspension bridge across a valley on Gamaksan in Gyeonggi','京畿道の紺岳山の渓谷を結ぶ吊り橋の駐車情報'),
('2751854','experience','제주 애월에서 알파카에게 먹이를 주는 목장 체험','An alpaca feeding experience at a ranch in Aewol, Jeju','済州の涯月でアルパカに餌をあげる牧場体験'),
('945824','holidays','서울에 있는 대한민국 임시정부 활동 공간 전시관 휴무일','Closed days of a Seoul exhibition site once used by the Korean provisional government','ソウルにある大韓民国臨時政府の活動場所だった展示館の休業日'),
]
rows=[]
for i,(pid,field,ko,en,ja) in enumerate(records,1):
 assert s[pid]=='test'
 for lang,q in [('ko',ko),('en',en),('ja',ja)]:
  rows.append(dict(intent_id=f'TEST-EXPL-{i:03}',query_id=f'TEST-EXPL-{i:03}-{lang}',place_id=pid,title_ko=d[(pid,'ko')]['title'],field=field,language=lang,split='test',query=q,reference_ko=d[(pid,'ko')]['description'],relevance_review='PENDING',language_review='PENDING',other_relevant_places='',review_note='AI draft after seeing development results; blind independent relevance review required'))
out=O/'holdout_exploratory_query_draft.csv'
with out.open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('exploratory heldout places',len(records),'queries',len(rows))
