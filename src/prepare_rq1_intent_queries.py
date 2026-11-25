"""Create paired, query-specific RQ1 development labels from original texts.
Labels are AI-reviewed drafts, not independent human annotations.
"""
import csv,json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'data/processed';O=R/'results/rq1_analysis'
def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
docs={(r['place_id'],r['language']):r for r in read(D/'common_dataset.csv') if r['source']=='tourapi'}
splits={r['place_id']:r['split'] for r in read(D/'common_place_splits_balanced.csv')}
# Local evidence tokens are checked in the matching source description.
# ABSENT means no mention in this snapshot, not that the activity does not exist.
specs=[
('2605528','군산 철길 마을에서 옛날 교복을 입고 사진 찍는 체험','A railroad village in Gunsan where visitors wear old school uniforms for photos','群山の線路沿いの村で昔の制服を着て写真を撮る体験',{'ko':'교복','en':'school uniforms','ja':''}),
('2436233','경기도 광주의 수목원에서 휠체어로 숲길 산책','Wheelchair-accessible forest trail at a botanic garden in Gwangju, Gyeonggi','京畿道広州の樹木園で車椅子を使って森の小道を散策',{'ko':'휠체어','en':'wheelchairs','ja':''}),
('125909','청주의 옛 향교에서 서예 체험','A calligraphy class at a historic Confucian school in Cheongju','清州の古い郷校で書道体験',{'ko':'','en':'calligraphy','ja':'書道'}),
('1019041','서울 북악산 근처 공원에서 배드민턴','Badminton at a park near Bugaksan in Seoul','ソウルの北岳山近くの公園でバドミントン',{'ko':'','en':'badminton','ja':'バドミントン'}),
('125891','공주 계룡산의 사찰에서 템플스테이','A temple stay at a temple on Gyeryongsan in Gongju','公州の鶏龍山にある寺院でテンプルステイ',{'ko':'템플스테이','en':'','ja':'テンプルステイ'}),
('125636','강릉 계곡에서 캠핑하고 야영하는 곳','A valley in Gangneung with a campsite for camping','江陵の渓谷でキャンプや野営ができる場所',{'ko':'야영장','en':'','ja':''}),
('126397','남원의 전통 정원에서 열리는 춘향 축제','The Chunhyang festival at a traditional garden in Namwon','南原の伝統庭園で開かれる春香祭',{'ko':'춘향제','en':'Chunhyangjae Festival','ja':''}),
('1621391','경주 향교에서 다도 체험','A tea ceremony experience at a Confucian school in Gyeongju','慶州の郷校で茶道体験',{'ko':'','en':'tea ceremonies','ja':'茶道'}),
('2554070','부산 영도의 조선 마을에서 투어 프로그램 체험','A tour program in a shipbuilding village in Yeongdo, Busan','釜山影島の造船の村でツアープログラムを体験',{'ko':'','en':'tour program','ja':''}),
('125266','청태산 자연휴양림에서 숲속 교육 체험','Forest education activities at a recreational forest on Cheongtaesan','青太山の自然休養林で森林教育を体験',{'ko':'숲 속 교실','en':'Forest Education Facility','ja':''}),
]
queries=[];labels=[]
for i,(pid,ko,en,ja,evidence) in enumerate(specs,1):
 assert set(evidence)=={'ko','en','ja'} and any(evidence.values()) and not all(evidence.values())
 intent=f'RQ1-{i:03}'
 for lang,text in [('ko',ko),('en',en),('ja',ja)]:
  original=docs[(pid,lang)];ev=evidence[lang]
  if ev:assert ev.lower() in original['description'].lower(),(pid,lang,ev)
  # Empty token follows direct review of each original document, but is not human-verified.
  labels.append(dict(intent_id=intent,place_id=pid,title_ko=docs[(pid,'ko')]['title'],document_language=lang,split=splits[pid],query_specific_status='PRESENT' if ev else 'MISSING',evidence=ev,review_method='AI_READ_ORIGINAL_SNAPSHOT',human_review='PENDING',note='MISSING means not mentioned in the available description, not absent in reality'))
  queries.append(dict(intent_id=intent,query_id=f'{intent}-{lang}',place_id=pid,language=lang,query=text,split=splits[pid],field='experience',review_status='PENDING_HUMAN_REVIEW'))
for name,rows in [('rq1_queries.csv',queries),('rq1_intent_labels.csv',labels)]:
 with (O/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('intents',len(specs),'queries',len(queries),'labels',len(labels))
