"""Rewrite the AI draft as short keyword and concise sentence search queries."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROCESSED=ROOT/'data/processed'
INPUT=PROCESSED/'tourapi_blind_query_template.csv'
OUTPUT=PROCESSED/'tourapi_ai_generated_queries_90_short.csv'

QUERIES={
 'B01-1':('대구 두피 마사지','Daegu scalp massage','大邱 頭皮マッサージ'),
 'B01-2':('대구에서 피부 분석해 주는 에스테틱','Where can I get a skin analysis in Daegu?','大邱で肌診断ができるエステは？'),
 'B02-1':('광주 선교사 기념관','Gwangju missionary memorial','光州 宣教師 記念館'),
 'B02-2':('광주에서 옛 서양식 건물 구경할 곳','Where can I see historic Western-style buildings in Gwangju?','光州で歴史ある洋風建築を見られる場所は？'),
 'B03-1':('거제 해금강 전망대','Geoje Haegeumgang viewpoint','巨済 海金剛 展望台'),
 'B03-2':('거제에서 가족과 가볍게 오를 전망 좋은 곳','Where can families take a short scenic hike in Geoje?','巨済で家族で気軽に登れる景色の良い場所は？'),
 'B04-1':('순창 웰니스 체험','Sunchang wellness activities','淳昌 ウェルネス 体験'),
 'B04-2':('순창에서 편백숲 산책하고 쉴 곳','Where can I relax and walk through a cypress forest in Sunchang?','淳昌でヒノキ林を散歩して休める場所は？'),
 'B05-1':('목포 바다 전망 케이블카','Mokpo cable car ocean views','木浦 ケーブルカー 海の眺め'),
 'B05-2':('목포에서 유달산과 섬을 한눈에 볼 곳','Where can I see Yudalsan and the islands together in Mokpo?','木浦で儒達山と島々を一望できる場所は？'),
 'B06-1':('해남 두륜산 사찰','Haenam Duryunsan temple','海南 頭輪山 寺院'),
 'B06-2':('해남에서 전통 차 문화로 유명한 절','Which temple in Haenam is known for traditional tea culture?','海南で伝統茶文化が有名なお寺は？'),
 'B07-1':('강원도 전쟁 피난 동굴','Gangwon wartime refuge cave','江原道 戦時避難 洞窟'),
 'B07-2':('강원도에서 검은색 동굴 생성물을 볼 수 있는 곳','Where can I see black cave formations in Gangwon?','江原道で黒い洞窟生成物が見られる場所は？'),
 'B08-1':('구례 지리산 신라 사찰','Gurye Jirisan Silla temple','求礼 智異山 新羅 寺院'),
 'B08-2':('지리산에서 샘물 전설이 있는 절','Which temple in Jirisan has a legend about a spring?','智異山で泉の伝説があるお寺は？'),
 'B09-1':('여수 일출 사찰','Yeosu sunrise temple','麗水 日の出 寺院'),
 'B09-2':('여수에서 거북 모양 바위가 있는 절','Which temple in Yeosu has rocks shaped like a turtle shell?','麗水で亀の甲羅のような岩があるお寺は？'),
 'B10-1':('군산 레트로 철도 여행','Gunsan retro railway trip','群山 レトロ 鉄道 旅行'),
 'B10-2':('군산에서 옛날 간식과 놀이를 즐길 곳','Where can I enjoy nostalgic snacks and games in Gunsan?','群山で昔懐かしいおやつや遊びを楽しめる場所は？'),
 'B11-1':('곡성 동악산 절','Gokseong Dongaksan temple','谷城 動楽山 寺院'),
 'B11-2':('곡성에서 계곡과 폭포 옆 사찰','Where is a temple near a valley and waterfalls in Gokseong?','谷城で渓谷や滝のそばにあるお寺は？'),
 'B12-1':('포천 허브 아로마 체험','Pocheon herbal aromatherapy','抱川 ハーブ アロマ 体験'),
 'B12-2':('포천에서 나에게 맞는 허브 스파를 받을 곳','Where can I get a personalized herbal spa treatment in Pocheon?','抱川で自分に合ったハーブスパを受けられる場所は？'),
 'B13-1':('영덕 대게 어시장','Yeongdeok snow crab fish market','盈徳 ズワイガニ 魚市場'),
 'B13-2':('영덕에서 BTS 뮤직비디오를 촬영한 항구','Which port in Yeongdeok was used in a BTS music video?','盈徳でBTSのミュージックビデオを撮影した港は？'),
 'B14-1':('여수 야경 크루즈 불꽃','Yeosu night cruise fireworks','麗水 ナイトクルーズ 花火'),
 'B14-2':('여수에서 배 타고 식사하며 바다를 볼 곳','Where can I dine on a boat while enjoying sea views in Yeosu?','麗水で海を眺めながら船上で食事できる場所は？'),
 'B15-1':('강릉 경포 근처 횟집 해변','Gangneung beach near Gyeongpo sashimi restaurants','江陵 鏡浦 近く 刺身 海岸'),
 'B15-2':('강릉에서 스쿠버다이빙 하기 좋은 해변','Which beach in Gangneung is good for scuba diving?','江陵でスキューバダイビングにおすすめの海岸は？'),
}

def main():
    with INPUT.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    assert len(rows)==30 and {r['intent_id'] for r in rows}==set(QUERIES)
    assert all(not r[f].strip() for r in rows for f in ('query_ko','query_en','query_ja'))
    for r in rows:
        for k,v in zip(('query_ko','query_en','query_ja'),QUERIES[r['intent_id']]): r[k]=v
        style='키워드형' if r['intent_id'].endswith('-1') else '짧은 문장형'
        r['annotator_note']=f'AI 생성 / {style} / 검토 전 / 정답은 대표 관광지이며 유사 장소 가능'
    assert len({r['place_id'] for r in rows})==15
    assert len({r[k] for r in rows for k in ('query_ko','query_en','query_ja')})==90
    with OUTPUT.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with OUTPUT.open(encoding='utf-8-sig',newline='') as f:
        assert list(csv.DictReader(f))==rows
    print(OUTPUT,'30 intents; 45 keyword queries; 45 short sentence queries; 90 total')

if __name__=='__main__': main()
