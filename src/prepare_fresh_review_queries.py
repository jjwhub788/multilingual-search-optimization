"""Draft new held-out-place queries without using enrichment selections or outcomes."""
import csv,hashlib,json,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'data/processed';O=R/'results/common_pilot'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
docs=read(D/'common_dataset.csv');splits=read(D/'common_place_splits_balanced.csv');old=read(O/'holdout_exploratory_query_draft.csv')
s={r['place_id']:r['split'] for r in splits};used={r['place_id'] for r in old};ko={r['place_id']:r for r in docs if r['source']=='tourapi' and r['language']=='ko'}
# Each evidence string must appear in the original Korean TourAPI description.
specs=[
('894145','experience','대구의 오래된 마을에서 한옥 숙박을 체험할 수 있는 곳','A traditional house stay in a historic village in Daegu','大邱の歴史ある村で韓屋宿泊を体験できる場所','한옥스테이'),
('127641','admission_fee','대전에서 동물원과 놀이공원을 함께 즐길 수 있는 곳의 입장료','Admission fees for a Daejeon park combining a zoo and amusement rides','大田で動物園と遊園地を楽しめる施設の入場料金','주랜드'),
('125909','opening_hours','청주에 있는 조선시대 지방 교육기관의 관람시간','Visiting hours for a Joseon-era local educational institution in Cheongju','清州にある朝鮮時代の地方教育機関の観覧時間','지방백성의 교육'),
('1928731','experience','전통 장류의 제조와 문화를 알아볼 수 있는 테마공원','A theme park where visitors can learn about traditional fermented sauces','伝統的な発酵調味料の製造と文化を知ることができるテーマ公園','전통 장류 테마공원'),
('2764342','opening_hours','이천에서 온천과 가족탕을 이용할 수 있는 시설의 운영시간','Opening hours for a hot spring facility in Icheon with private family baths','利川で温泉と家族風呂を利用できる施設の営業時間','가족탕'),
('2715684','parking','정읍 내장호 주변의 단풍나무 생태공원 주차 정보','Parking at a maple tree ecological park near Naejang Lake in Jeongeup','井邑の内蔵湖周辺にあるカエデの生態公園の駐車情報','내장호 주변'),
('125807','holidays','횡성에 있는 천주교 성당의 휴무일','Closed days for a Catholic church in Hoengseong','横城にあるカトリック聖堂の休業日','횡성'),
('126738','parking','양양에서 등대와 방파제를 볼 수 있는 항구 주차 정보','Parking at a harbor in Yangyang with lighthouses and a breakwater','襄陽で灯台と防波堤を見られる港の駐車情報','양양'),
('2831210','experience','비무장지대의 역사와 생태를 실감 콘텐츠로 체험하는 실내 공간','An indoor space with immersive exhibits about the history and ecology of the DMZ','非武装地帯の歴史と生態を体感型コンテンツで学べる屋内施設','실감미디어 콘텐츠'),
('2027193','opening_hours','광주 도심 속 수행 사찰의 운영시간','Opening hours for a Buddhist temple in central Gwangju','光州の市街地にある修行寺院の営業時間','도심 속의 수행 사찰'),
]
rows=[]
for i,(pid,field,k,e,j,evidence) in enumerate(specs,1):
 assert s[pid]=='test' and pid not in used and evidence in ko[pid]['description']
 for lang,q in [('ko',k),('en',e),('ja',j)]:
  rows.append(dict(intent_id=f'FRESH-{i:03}',query_id=f'FRESH-{i:03}-{lang}',place_id=pid,title_ko=ko[pid]['title'],field=field,language=lang,split='test',query=q,feature_evidence=evidence,source='TourAPI 한국어 원본 설명',relevance_review='PENDING',language_review='PENDING',other_relevant_places='',review_note='AI 초안; 실제 해당 정보 확인 및 복수 정답·언어 검토 필요'))
out=O/'fresh_holdout_queries_review.csv'
with out.open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
hash_value=hashlib.sha256(out.read_bytes()).hexdigest();m={'query_sha256':hash_value,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'AI_DRAFT_HUMAN_REVIEW_PENDING','feature_check':'Korean evidence substring found in TourAPI description; not independent factual validation','independence':'new places not previously queried; same AI has seen earlier results; not blind independent assessment','not_yet_executed':True}
with (O/'fresh_holdout_queries_manifest.json').open('x',encoding='utf-8') as f:json.dump(m,f,ensure_ascii=False,indent=2)
print('장소',len(specs),'검색어',len(rows),'sha256',hash_value)
