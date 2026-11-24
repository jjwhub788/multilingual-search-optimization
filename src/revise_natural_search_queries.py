"""Normalize the 600-query draft to realistic search-bar phrases.

The two intents per place are representative-feature and detail-feature,
not keyword versus question form. All queries remain AI-authored drafts.
"""
import csv
import statistics
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data/processed'
INPUT=P/'tourapi_ai_generated_queries_600_short.csv'
OUTPUT=P/'tourapi_ai_generated_queries_600_natural.csv'

# Pair ID | natural Korean phrase | equivalent English/Japanese search phrase.
DETAILS='''B01|대구 피부 타입 진단 에스테틱|Daegu skin analysis beauty salon|大邱 肌診断 エステ
B02|광주 회색 벽돌 옛 예배당|Gwangju old gray brick chapel|光州 灰色レンガ 旧礼拝堂
B03|거제 아이랑 가볍게 등산|Geoje easy hike with kids|巨済 子連れ 気軽なハイキング
B04|순창 편백숲 산책 명상|Sunchang cypress forest walk meditation|淳昌 ヒノキ林 散策 瞑想
B05|목포 유달산 섬 한눈에 보는 곳|Mokpo panoramic views of Yudalsan and islands|木浦 儒達山 島々 一望
B06|해남 전통 차 문화 유명한 절|Haenam temple famous for tea culture|海南 伝統茶文化 有名な寺
B07|영월 임진왜란 피난 동굴|Yeongwol Imjin War refuge cave|寧越 文禄・慶長の役 避難洞窟
B08|지리산 샘물 전설 있는 절|Jirisan temple spring legend|智異山 泉の伝説 寺
B09|여수 거북바위 있는 절|Yeosu temple with turtle-shaped rocks|麗水 亀の甲羅のような岩 寺
B10|군산 철길 옛날 교복 체험|Gunsan railway vintage school uniform experience|群山 線路 昔の制服 体験
B11|곡성 계곡 폭포 있는 절|Gokseong temple by valley and waterfall|谷城 渓谷 滝 お寺
B12|포천 체질 맞춤 허브 스파|Pocheon personalized herbal spa|抱川 体質別 ハーブスパ
B13|영덕 BTS 뮤비 찍은 바다|Yeongdeok BTS music video seaside location|盈徳 BTS MV ロケ地 海
B14|여수 배에서 밥 먹는 크루즈|Yeosu cruise with dining onboard|麗水 船上で食事 クルーズ
B15|강릉 스쿠버다이빙 해변|Gangneung scuba diving beach|江陵 スキューバダイビング 海岸
B16|제주 자전거 타기 좋은 섬|Jeju island good for cycling|済州 サイクリングにいい島
B17|강진 옛 성곽 걷기|Gangjin historic fortress wall walk|康津 古い城壁 散策
B18|월출산 녹찻잎 따기 체험|Wolchulsan tea leaf picking|月出山 茶摘み体験
B19|강화해협 지키던 유적지|Historic site guarding the Ganghwa Strait|江華海峡を守った史跡
B20|고창 한옥 꽃정원 카페|Gochang hanok flower garden cafe|高敞 韓屋 花庭園 カフェ
B21|광주 떡갈비 뼛국 같이 나오는 곳|Gwangju tteokgalbi with bone soup|光州 トッカルビ 骨スープ付き
B22|광주 아이랑 눈썰매|Gwangju snow sledding with kids|光州 子連れ 雪そり
B23|완주 인절미 만들기 체험|Wanju injeolmi making experience|完州 インジョルミ 作り体験
B24|춘천 동백꽃 소설 작가 생가|Chuncheon Camellias author birthplace|春川 小説「椿」作家 生家
B25|통영 편백숲 족욕|Tongyeong cypress forest foot bath|統営 ヒノキ林 足湯
B26|통영 바다 전망 야외 미술|Tongyeong outdoor art sea view|統営 海の眺め 屋外アート
B27|서울 근교 산성 트레킹|Mountain fortress hiking near Seoul|ソウル近郊 山城 ハイキング
B28|울산 오래된 팽나무 있는 절|Ulsan temple with old hackberry tree|蔚山 古いエノキのある寺
B29|광주 아이랑 에너지 절약 체험|Gwangju energy saving activities for kids|光州 子供 省エネ体験
B30|청송 약수터 산불 피해|Cheongsong mineral spring wildfire damage|青松 薬水場 山火事 被害
B31|양산 아이랑 계곡 숲속 숙소|Yangsan forest accommodation by a stream for families|梁山 子連れ 渓谷 森の宿
B32|무주 구천동 폭포 계곡|Muju Gucheondong waterfalls valley|茂朱 九千洞 滝 渓谷
B33|고성 아이랑 실내외 수영장|Goseong indoor outdoor pools for kids|高城 子連れ 屋内外プール
B34|단양 강 한가운데 정자|Danyang pavilion in the middle of a river|丹陽 川の真ん中 亭子
B35|여수 다리 건너가는 섬|Island accessible by bridge from Yeosu|麗水 橋で行ける島
B36|해남 땅끝 잔잔한 해수욕장|Haenam Ttangkkeut calm beach|海南 タンクッ 穏やかな海水浴場
B37|부산 스릴 있는 놀이기구|Busan thrilling amusement rides|釜山 スリル系 アトラクション
B38|제주 큰 용암석주 동굴|Jeju cave giant lava column|済州 巨大溶岩石柱 洞窟
B39|안동 김계행 옛 정자|Andong Kim Gye-haeng historic pavilion|安東 金係行 古い亭子
B40|서천 이색 기리는 유적|Seocheon Yi Saek memorial site|舒川 李穡 史跡
B41|평창 계란프라이 모양 석순|Pyeongchang fried egg shaped stalagmites|平昌 目玉焼き形 石筍
B42|대구 반월당 미술품 구경|Daegu Banwoldang art browsing|大邱 半月堂 美術品 鑑賞
B43|서울 궁궐 사이 한옥 동네|Seoul hanok neighborhood between palaces|ソウル 宮殿の間 韓屋の町
B44|분당 청담동 느낌 카페|Bundang upscale Cheongdam-style cafes|盆唐 清潭洞風 おしゃれカフェ
B45|영광 백제 불교 처음 전해진 절|Yeonggwang temple early Baekje Buddhism|霊光 百済 仏教伝来 寺
B46|한탄강 움푹 파인 협곡 폭포|Hantangang recessed gorge waterfall|漢灘江 くぼんだ峡谷 滝
B47|삼척 숲속 트리하우스|Samcheok forest treehouse|三陟 森 ツリーハウス
B48|전주 골동품 다락방 카페|Jeonju antiques attic cafe|全州 骨董品 屋根裏 カフェ
B49|제부도 갯벌 위 공중 여행|Jebudo aerial mudflat sightseeing|済扶島 干潟 空中散歩
B50|제주 큰 화산 분화구|Jeju large volcanic crater|済州 大きな火口
B51|진도 섬 사이 해지는 풍경|Jindo sunset between islands|珍島 島の間に沈む夕日
B52|소백산 숲속 명상 프로그램|Sobaeksan forest meditation program|小白山 森林瞑想 プログラム
B53|소백산 연화봉 망원경|Sobaeksan Yeonhwabong telescope|小白山 蓮花峰 望遠鏡
B54|속초 설악산 바다 같이 보는 놀이기구|Sokcho ride overlooking Seoraksan and sea|束草 雪岳山と海を眺める乗り物
B55|충주 조선 왕이 찾던 온천|Chungju hot spring visited by Joseon king|忠州 朝鮮時代の王が訪れた温泉
B56|양양 가우디 스타일 실내 수영장|Yangyang Gaudí-style indoor pool|襄陽 ガウディ風 屋内プール
B57|아산 언덕 위 근대 성당|Asan historic hilltop Catholic church|牙山 丘の上 近代カトリック教会
B58|정선 송천 골지천 만나는 곳|Jeongseon Songcheon Goljicheon meeting point|旌善 松川と骨只川 合流地点
B59|기장 오래된 소나무 편백숲|Gijang old pine and cypress forest|機張 古い松 ヒノキ林
B60|대구 산 위 시내 전망|Daegu city views from a mountain|大邱 山の上 市街地展望
B61|여수 빅오쇼 스카이타워|Yeosu Big-O show Sky Tower|麗水 ビッグオーショー スカイタワー
B62|진해 하천 옆 벚꽃 야경|Jinhae streamside cherry blossom night views|鎮海 川沿い 夜桜
B63|영월 어린 조선 왕 무덤 역사|Yeongwol young Joseon king tomb history|寧越 朝鮮の若い王の墓 歴史
B64|부산 용호동 바다 지질 명소|Busan Yongho-dong coastal geology spot|釜山 龍湖洞 海岸 地質名所
B65|해남 노을 보이는 야영장|Haenam campsite with sunset view|海南 夕日が見えるキャンプ場
B66|울산 전통 옹기 장인 작품|Ulsan traditional pottery artisan works|蔚山 伝統甕器 職人の作品
B67|완도 해조류 머드 테라피|Wando seaweed mud therapy|莞島 海藻 マッドセラピー
B68|제주 올레길 7코스 바다 바위|Jeju Olle Trail 7 sea rock|済州 オルレ7コース 海の岩
B69|거창 협곡 위 다리 산책|Geochang gorge bridge walk|居昌 渓谷の上 吊り橋 散策
B70|광주 1920년대 서양식 주택|Gwangju 1920s Western-style house|光州 1920年代 洋式住宅
B71|증도 하얀 모래 해변 산책|Jeungdo white sand beach walk|曽島 白い砂浜 散策
B72|광주 시내 불교 법회|Gwangju downtown Buddhist services|光州 市街地 仏教 法会
B73|인제 숲속 힐링 산림욕|Inje healing forest bathing|麟蹄 森林浴 癒し
B74|부산 초량 산복도로 문학 전시|Busan Choryang hillside road literary exhibition|釜山 草梁 山腹道路 文学展示
B75|인천공항 아트 전시 터미널|Incheon airport terminal art exhibitions|仁川空港 アート展示 ターミナル
B76|제주 용암숲 오름 산책|Jeju lava forest oreum walking trail|済州 溶岩の森 オルム 散策
B77|서울 조계종 총본산|Seoul head temple of Jogye Order|ソウル 曹渓宗 総本山
B78|순천 송광사 선암사 사이 등산|Suncheon hike between Songgwangsa and Seonamsa|順天 松広寺と仙岩寺の間 登山
B79|무등산 서쪽 오래된 절|Historic temple west of Mudeungsan|無等山 西側 古刹
B80|진도 화가 허련 살던 집|Jindo painter Heo Ryeon old house|珍島 画家 許鍊の旧宅
B81|부안 해식동굴 층층이 쌓인 바위|Buan sea caves layered rocks|扶安 海食洞窟 層状の岩
B82|경주 금관 날개 달린 말 그림 나온 무덤|Gyeongju tomb gold crown flying horse painting|慶州 金冠 空飛ぶ馬の絵 古墳
B83|봉화 원시림 산속 캠핑|Bonghwa old growth forest camping|奉化 原生林 山 キャンプ
B84|제천 충주호 옛 한옥 복원|Jecheon Chungju Lake restored historic houses|堤川 忠州湖 古民家 復元
B85|서울 외국어 가능한 메이크업 컬러 진단|Seoul multilingual makeup color analysis|ソウル 外国語対応 メイクカラー診断
B86|여수 소나무 숲길 끝 바다|Yeosu sea at end of pine trail|麗水 松林の小道の先 海
B87|파주 오래된 회중시계 전시|Paju antique pocket watch exhibition|坡州 古い懐中時計 展示
B88|제천 계곡 보이는 야외 온수풀|Jecheon outdoor warm pool valley view|堤川 渓谷を眺める屋外の温水プール
B89|강릉 피노키오 인형 전시|Gangneung Pinocchio puppet exhibition|江陵 ピノキオ 人形展示
B90|평창 한옥 숙박 김치 만들기|Pyeongchang hanok stay kimchi making|平昌 韓屋 宿泊 キムチ作り
B91|한라산 기암 윗세오름 코스|Hallasan rock formations Witseoreum trail|漢拏山 奇岩 ウィッセオルム コース
B92|여수 겨울 동백꽃 구경|Yeosu winter camellia flowers|麗水 冬 ツバキ 花見
B93|서울 체질 맞춤 한방 스파|Seoul traditional herbal spa by body type|ソウル 体質別 韓方スパ
B94|제주 야자수 흰 모래 해변|Jeju palm trees white sand beach|済州 ヤシの木 白い砂浜
B95|충남 천주교 박해 순례지|Chungnam Catholic persecution pilgrimage site|忠清南道 カトリック迫害 巡礼地
B96|정선 금 채굴 체험 석회동굴|Jeongseon gold mining experience limestone cave|旌善 金採掘体験 石灰洞窟
B97|수원 무지개 모양 수문|Suwon rainbow arched floodgates|水原 虹形の水門
B98|부산 수영구 전시 대나무 정원|Busan Suyeong exhibitions bamboo garden|釜山 水営区 展示 竹の庭
B99|제주 감귤 바다 테마 실내 포토존|Jeju citrus and sea themed indoor photo spot|済州 ミカンと海テーマ 屋内フォトスポット
B100|서울 홍삼 오일 전신 마사지|Seoul red ginseng oil full body massage|ソウル 紅参オイル 全身マッサージ'''

# Refocus representative queries whose initial phrasing was overly detailed,
# too close to the detail query, or unnatural for a search bar.
REPRESENTATIVE={
 'B07':('영월 석회동굴 여행','Yeongwol limestone cave trip','寧越 石灰洞窟 観光'),
 'B13':('영덕 대게 어시장','Yeongdeok snow crab fish market','盈徳 ズワイガニ 魚市場'),
 'B15':('강릉 경포 근처 해변','Gangneung beach near Gyeongpo','江陵 鏡浦付近 ビーチ'),
 'B30':('청송 탄산 약수터','Cheongsong carbonated mineral spring','青松 炭酸 薬水場'),
 'B55':('충주 온천 족욕','Chungju hot spring foot bath','忠州 温泉 足湯'),
 'B63':('영월 왕릉 여행','Yeongwol royal tomb visit','寧越 王陵 観光'),
 'B70':('광주 옛 선교사 주택','Gwangju historic missionary residence','光州 旧宣教師住宅'),
 'B74':('부산 느린 우체통','Busan slow mailbox','釜山 遅い郵便ポスト'),
 'B75':('인천공항 터미널 관광','Incheon airport terminal attractions','仁川空港 ターミナル 観光'),
 'B87':('파주 이색 박물관','Paju unusual museums','坡州 ユニークな博物館'),
 'B93':('서울 한방 마사지','Seoul herbal massage','ソウル 韓方マッサージ'),
 'B100':('서울 홍삼 스파','Seoul red ginseng spa','ソウル 紅参 スパ'),
}

def main():
    with INPUT.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    details={}
    for line in DETAILS.splitlines():
        pair,*queries=line.split('|');assert len(queries)==3 and pair not in details
        details[pair]=queries
    assert len(rows)==200 and len(details)==100
    assert set(details)=={r['intent_id'].rsplit('-',1)[0] for r in rows}
    source=[dict(r) for r in rows]
    fields_query=('query_ko','query_en','query_ja')
    for r in rows:
        pair,num=r['intent_id'].rsplit('-',1)
        if num=='2': chosen=details[pair];style='세부 특징 검색'
        else: chosen=REPRESENTATIVE.get(pair);style='대표 특징 검색'
        if chosen:
            for key,q in zip(fields_query,chosen):r[key]=q
        r['annotator_note']=f'AI 생성 / {style} / 검토 전 / 대표 정답 관광지이며 유사 장소 가능'
    assert len({r['place_id'] for r in rows})==100
    assert set(Counter(r['place_id'] for r in rows).values())=={2}
    assert all(a[k]==b[k] for a,b in zip(source,rows) for k in ('intent_id','place_id','title_ko','reference_description_ko'))
    assert len({r[k] for r in rows for k in fields_query})==600
    assert all(r[k].strip() and not r[k].endswith(('?', '？')) for r in rows for k in fields_query)
    with OUTPUT.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with OUTPUT.open(encoding='utf-8-sig',newline='') as f:assert list(csv.DictReader(f))==rows
    print(OUTPUT)
    print('100 places, 200 intents, 600 queries')
    print('Mean Korean chars',round(statistics.mean(len(r['query_ko']) for r in source),2),'->',round(statistics.mean(len(r['query_ko']) for r in rows),2))
    print('types',Counter(r['annotator_note'].split(' / ')[1] for r in rows))

if __name__=='__main__': main()
