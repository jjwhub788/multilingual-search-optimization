"""Development query drafts; evidence and labels require human review."""
import csv
from pathlib import Path
R=Path(__file__).resolve().parents[1]; D=R/'data/processed'; O=R/'results/common_pilot'
def load(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
items=load(D/'common_enrichment_item_pool.csv');docs=load(D/'common_dataset.csv');splits={r['place_id']:r['split'] for r in load(D/'common_place_splits_balanced.csv')}
samples=load(O/'dev_semantic_sample_review.csv');
additional=[
('1921423','parking','실내 시설과 야외 파도풀이 있고 주차할 수 있는 워터파크','A water park with indoor pools, an outdoor wave pool and parking','屋内施設と屋外の波のプールがあり駐車できるウォーターパーク','실내 워터파크와 야외 파도풀'),
('2719004','parking','고창에서 라벤더 정원을 볼 수 있고 주차할 수 있는 농원','A garden farm in Gochang with lavender and parking','高敞でラベンダーの庭園を見られて駐車できる農園','고창의 라벤더 정원'),
('127539','admission_fee','광주에 있는 놀이공원의 입장료','Admission fee for an amusement park in Gwangju','光州にある遊園地の入場料金','광주 소재 놀이공원'),
('129443','opening_hours','호수 주변 습지와 자연학습장이 있는 생태공원 운영시간','Opening hours of a lakeside ecological park with wetlands and nature learning areas','湖畔に湿地と自然学習場がある生態公園の営業時間','호수 인근 습지와 자연학습장'),
('127860','parking','제주에서 돌하르방과 석물을 볼 수 있고 주차할 수 있는 공원','A Jeju park with stone grandfather statues and parking','済州でトルハルバンなどの石像を見られて駐車できる公園','제주 돌하르방과 석물'),
('128627','parking','대구의 타워를 중심으로 한 테마파크 주차 정보','Parking information for a theme park centered around a tower in Daegu','大邱のタワーを中心としたテーマパークの駐車情報','대구 83타워 중심 테마파크'),
('129438','parking','메타세쿼이아 숲을 걸을 수 있는 자연휴양림 주차','Parking at a recreational forest with metasequoia walking trails','メタセコイアの森を散策できる自然休養林の駐車場','메타세쿼이아 숲 자연휴양림'),
('2436233','opening_hours','경기도 광주의 여러 테마원이 있는 수목원 운영시간','Opening hours of an arboretum with themed gardens in Gwangju, Gyeonggi','京畿道広州のテーマ庭園がある樹木園の営業時間','경기도 광주시 테마원 수목원'),
('125583','holidays','한강과 임진강을 조망할 수 있는 통일전망대 휴무일','Closed days of a unification observatory overlooking the Han and Imjin rivers','漢江と臨津江を見渡せる統一展望台の休業日','한강과 임진강 조망 통일전망대'),
('1059479','opening_hours','벚꽃축제가 열리는 여의도 한강 공원 운영시간','Opening hours of a Han River park in Yeouido known for a cherry blossom festival','桜祭りが開かれる汝矣島の漢江公園の営業時間','여의도 한강공원과 벚꽃축제'),
('2554070','experience','부산 영도에서 배를 타고 조선 마을을 둘러보는 체험','A boat tour of a shipbuilding village in Yeongdo, Busan','釜山の影島で船に乗って造船の村を巡る体験','조선 마을 투어 및 해상투어'),
('2668973','opening_hours','부산 해운대의 초고층 전망대 운영시간','Opening hours of a high-rise observatory in Haeundae, Busan','釜山の海雲台にある高層展望台の営業時間','부산 해운대 고층 전망대'),
('127740','admission_fee','영월에 있는 천문대 관람 입장료','Admission fee for an observatory in Yeongwol','寧越にある天文台の入場料金','영월 천문대'),
('126747','holidays','서울에 있는 한옥마을 휴무일','Closed days of a traditional Korean house village in Seoul','ソウルにある韓屋村の休業日','서울 한옥마을'),
('127817','holidays','충북에 있는 수목원의 휴무일','Closed days of an arboretum in North Chungcheong','忠清北道にある樹木園の休業日','충북 수목원'),
('1607565','opening_hours','강원도에 있는 유럽풍 정원의 운영시간','Opening hours of a European-style garden in Gangwon','江原道にあるヨーロッパ風庭園の営業時間','강원도 유럽풍 정원'),
]
keys={(r['place_id'],r['field']) for r in items};title={r['place_id']:r['title_ko'] for r in items};q=[]
for r in samples:q.append(dict(r))
for i,(pid,field,ko,en,ja,clue) in enumerate(additional,5):
 assert splits[pid]=='dev' and (pid,field) in keys
 candidates=[r for r in items if r['place_id']==pid and r['field']==field]
 for lang,text in [('ko',ko),('en',en),('ja',ja)]:
  facts=[r for r in candidates if r['language']==lang]; fact=facts[0] if facts else candidates[0]
  q.append(dict(intent_id=f'SAMPLE-{i:03}',query_id=f'SAMPLE-{i:03}-{lang}',place_id=pid,title_ko=title[pid],field=field,language=lang,split='dev',query=text,feature_evidence=clue,feature_source='TourAPI 설명 (원문 사람 검토 필요)',fact_value=fact['verified_aihub_value'],fact_source=fact['verification_source'],relevance_review='PENDING',language_review='PENDING',other_relevant_places='',review_note='AI 작성 초안; 검색어 조건·출처·복수 정답 사람 검토 필요; 보완정보 존재 언어 '+fact['language']))
assert len({x['query_id'] for x in q})==len(q)
out=O/'dev_semantic_expanded_review.csv'
with out.open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(q[0]));w.writeheader();w.writerows(q)
print('의도',len(q)//3,'검색어',len(q),'개발용 관광지',len({x['place_id'] for x in q}))
