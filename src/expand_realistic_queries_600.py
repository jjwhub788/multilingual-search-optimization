"""Expand the 15-place realistic query draft to all 100 TourAPI pilot places."""
import csv
import statistics
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data/processed'
STANDARD=P/'tourapi_standardized.csv'
EXISTING=P/'tourapi_ai_generated_queries_90_short.csv'
OUTPUT=P/'tourapi_ai_generated_queries_600_short.csv'
# place_id | keyword ko/en/ja | concise sentence ko/en/ja
DATA='''126446|제주 청보리 섬|Jeju green barley island|済州 青麦 島|제주에서 자전거로 둘러볼 섬 어디야?|Which island off Jeju can I cycle around?|済州で自転車で一周できる島は？
127110|강진 조선 군사 유적|Gangjin Joseon military heritage|康津 朝鮮時代 軍事遺跡|강진에서 옛 성곽을 볼 곳은?|Where can I see old fortress walls in Gangjin?|康津で昔の城壁を見られる場所は？
128151|강진 녹차밭|Gangjin green tea fields|康津 緑茶畑|월출산 근처 찻잎 따는 체험 어디서 해?|Where can I pick tea leaves near Wolchulsan?|月出山の近くで茶摘み体験できる場所は？
127185|강화도 옛 포대|Ganghwa historic artillery fort|江華島 旧砲台|강화해협을 지키던 역사 유적 어디야?|Which historic site guarded the Ganghwa Strait?|江華海峡を守っていた史跡は？
2719004|고창 라벤더 꽃밭|Gochang lavender fields|高敞 ラベンダー畑|고창에서 한옥과 꽃 정원을 함께 볼 곳은?|Where can I see a hanok and flower gardens in Gochang?|高敞で韓屋と花の庭園を楽しめる場所は？
2605353|광주 떡갈비 맛집 골목|Gwangju grilled rib patty street|光州 トッカルビ グルメ通り|광주에서 떡갈비랑 뼛국 먹을 곳은?|Where can I try tteokgalbi and bone broth in Gwangju?|光州でトッカルビと骨スープを食べるなら？
127539|광주 가족 놀이공원|Gwangju family amusement park|光州 家族 遊園地|광주에서 아이랑 눈썰매 탈 곳은?|Where can I go snow sledding with kids in Gwangju?|光州で子供と雪そりを楽しめる場所は？
1977631|완주 황토 한증막|Wanju clay sauna|完州 黄土サウナ|완주에서 인절미 만들고 쉬어 갈 곳은?|Where can I make rice cakes and relax in Wanju?|完州でインジョルミ作りと休息を楽しめる場所は？
127933|춘천 문학 여행|Chuncheon literary trip|春川 文学旅行|춘천에서 소설 동백꽃 작가의 생가를 보고 싶어|Where can I see the birthplace of the author of Camellias in Chuncheon?|春川で小説「椿」の作家の生家を見たい
1623931|통영 편백숲 맨발 산책|Tongyeong barefoot cypress walk|統営 ヒノキ林 裸足 散策|통영에서 숲속 족욕하고 쉴 곳은?|Where can I enjoy a forest foot bath in Tongyeong?|統営で森の足湯を楽しめる場所は？
127621|통영 야외 조각 전시|Tongyeong outdoor sculptures|統営 屋外彫刻 展示|통영에서 조각 작품과 바다 전망을 함께 보고 싶어|Where can I see sculptures and sea views in Tongyeong?|統営で彫刻作品と海の眺めを楽しみたい
125449|경기 광주 세계유산 성곽|Gyeonggi Gwangju World Heritage fortress|京畿道 広州 世界遺産 城壁|서울 근교 조선시대 산성 걷기 좋은 곳은?|Where can I walk along a Joseon fortress near Seoul?|ソウル近郊で朝鮮時代の山城を散策するなら？
298348|울산 대운산 고찰|Ulsan Daeunsan historic temple|蔚山 大雲山 古刹|울산에서 맑은 계곡과 오래된 팽나무가 있는 절은?|Which temple in Ulsan has a clear valley and old hackberry tree?|蔚山で清流と古いエノキがあるお寺は？
2470068|광주 친환경 에너지 체험|Gwangju green energy exhibit|光州 環境 エネルギー 体験|광주에서 아이와 에너지 절약을 배울 곳은?|Where can kids learn about saving energy in Gwangju?|光州で子供と省エネを学べる場所は？
126111|청송 탄산 약수 전통|Cheongsong carbonated mineral spring history|青松 炭酸 薬水 歴史|청송에 탄산 약수로 알려진 곳은 어디야?|Which place in Cheongsong is known for carbonated spring water?|青松で炭酸の薬水で知られる場所は？
4134831|양산 숲속 카라반|Yangsan forest caravan stay|梁山 森 キャンピングカー 宿泊|대운산 근처 가족과 계곡에서 쉴 곳은?|Where can my family relax by a stream near Daeunsan?|大雲山の近くで家族と渓谷で休める場所は？
126238|무주 산악 국립공원|Muju mountain national park|茂朱 山岳 国立公園|무주에서 폭포와 구천동 계곡을 볼 곳은?|Where can I see waterfalls and Gucheondong Valley in Muju?|茂朱で滝と九千洞渓谷を楽しむなら？
128118|고성 온천 워터파크|Goseong hot spring water park|高城 温泉 ウォーターパーク|고성에서 아이랑 실내외 물놀이하고 싶어|Where can kids swim indoors and outdoors in Goseong?|高城で子供と屋内外の水遊びを楽しみたい
125913|단양 남한강 바위 봉우리|Danyang Namhan River rock peaks|丹陽 南漢江 岩峰|단양에서 강 가운데 정자가 있는 명소는?|Which Danyang attraction has a pavilion on a rock in the river?|丹陽で川の中の岩に亭子がある名所は？
126284|여수 섬 해안 드라이브|Yeosu island coastal drive|麗水 島 海岸 ドライブ|여수에서 다리로 갈 수 있는 섬 여행지 추천해줘|Which island near Yeosu can I reach by bridge?|麗水から橋で行ける島の旅行先は？
126302|해남 송림 해변|Haenam pine forest beach|海南 松林 海水浴場|해남 땅끝 근처 잔잔한 바닷가 어디야?|Where is a calm beach near Ttangkkeut in Haenam?|海南のタンクッ付近で波が穏やかな海岸は？
2815627|부산 놀이공원 퍼레이드|Busan amusement park parade|釜山 遊園地 パレード|부산에서 신나는 놀이기구 타고 싶어|Where can I enjoy thrilling rides in Busan?|釜山でスリルのあるアトラクションに乗りたい
126452|제주 용암동굴 관광|Jeju lava tube sightseeing|済州 溶岩洞窟 観光|제주에서 큰 용암석주로 유명한 동굴은?|Which Jeju cave is famous for its giant lava column?|済州で巨大な溶岩石柱で有名な洞窟は？
126998|안동 조선시대 정자|Andong Joseon pavilion|安東 朝鮮時代 亭子|안동에서 김계행이 세운 옛 정자는?|Which historic pavilion in Andong was built by Kim Gye-haeng?|安東で金係行が建てた昔の亭子は？
125986|서천 고려 학자 서원|Seocheon Goryeo scholar academy|舒川 高麗 学者 書院|서천에서 이색을 기리는 유교 유적은?|Which Confucian historic site in Seocheon honors Yi Saek?|舒川で李穡を祀る儒教の史跡は？
1333464|평창 석회동굴 탐방|Pyeongchang limestone cave tour|平昌 石灰洞窟 探訪|평창에서 특이한 달걀프라이 모양 석순을 볼 곳은?|Where can I see fried-egg-shaped stalagmites in Pyeongchang?|平昌で目玉焼き形の石筍を見られる場所は？
126802|대구 화랑 거리|Daegu art gallery street|大邱 画廊 通り|대구 반월당 근처 미술품 구경할 곳은?|Where can I browse art near Banwoldang in Daegu?|大邱の半月堂付近で美術品を鑑賞するなら？
126537|서울 전통 한옥 골목|Seoul traditional hanok alleys|ソウル 伝統韓屋 路地|서울에서 궁궐 사이에 한옥이 많은 동네는?|Which Seoul neighborhood has many hanok between palaces?|ソウルで宮殿の間に韓屋が多い町は？
1053768|분당 분위기 좋은 카페거리|Bundang stylish cafe street|盆唐 おしゃれ カフェ通り|성남에서 카페가 모여 있는 고급스러운 거리는?|Where is a stylish street full of cafes in Seongnam?|城南でおしゃれなカフェが集まる通りは？
126349|영광 상사화 꽃구경|Yeonggwang red spider lilies|霊光 キツネノカミソリ 花見|영광에서 백제 불교의 전래 역사 볼 절은?|Which temple in Yeonggwang tells the story of early Baekje Buddhism?|霊光で百済仏教伝来の歴史を知るお寺は？
2569476|포천 현무암 폭포|Pocheon basalt waterfall|抱川 玄武岩 滝|한탄강 근처 움푹 파인 협곡 명소는?|Where is a bowl-shaped gorge near the Hantangang River?|漢灘江の近くでくぼんだ峡谷の名所は？
2746584|삼척 산림 치유|Samcheok forest therapy|三陟 森林セラピー|삼척에서 트리하우스와 숲길을 즐길 곳은?|Where can I enjoy treehouses and forest trails in Samcheok?|三陟でツリーハウスと森の散策路を楽しむなら？
2804498|전주 옛 정미소 카페|Jeonju historic rice mill cafe|全州 古い精米所 カフェ|전주에서 골동품과 다락방이 있는 카페는?|Which cafe in Jeonju has antiques and an attic?|全州で骨董品と屋根裏部屋のあるカフェは？
2945376|화성 바다 위 케이블카|Hwaseong sea cable car|華城 海上 ケーブルカー|전곡항에서 제부도 갯벌을 내려다보고 싶어|Where can I view the Jebudo mudflats from Jeongok Harbor?|前谷港から済扶島の干潟を眺めたい
126435|제주 바다 일출 화산|Jeju ocean sunrise volcano|済州 海 日の出 火山|제주에서 큰 화산 분화구를 보고 싶어|Where can I see a large volcanic crater in Jeju?|済州で大きな火口を見たい
128224|진도 섬 일몰 명소|Jindo island sunset spot|珍島 島 夕日 名所|진도에서 다도해 섬 사이로 해 지는 곳은?|Where can I see the sun set between islands in Jindo?|珍島で島々の間に沈む夕日を見られる場所は？
3346664|영주 자연 생태 교육|Yeongju nature education|栄州 自然 生態 学習|소백산에서 숲 명상 체험할 곳은?|Where can I try forest meditation in Sobaeksan?|小白山で森林瞑想を体験できる場所は？
2607143|소백산 별 관측|Sobaeksan stargazing|小白山 星 観測|소백산 연화봉 근처 망원경 있는 곳은?|Where is the observatory near Yeonhwabong on Sobaeksan?|小白山の蓮花峰付近に天文台はある？
2814180|속초 바다 대관람차|Sokcho seaside Ferris wheel|束草 海 大観覧車|속초에서 설악산과 바다를 한눈에 볼 놀이시설은?|Which ride in Sokcho overlooks Seoraksan and the sea?|束草で雪岳山と海を一望できる乗り物は？
3343355|충주 온천 족욕|Chungju hot spring foot bath|忠州 温泉 足湯|충주에서 자연 온천에 발 담글 곳은?|Where can I soak my feet in a natural hot spring in Chungju?|忠州で天然温泉の足湯を楽しむなら？
3082578|양양 해변 워터파크|Yangyang seaside water park|襄陽 海辺 ウォーターパーク|양양에서 가우디풍 실내 수영장을 즐길 곳은?|Where can I enjoy a Gaudí-style indoor pool in Yangyang?|襄陽でガウディ風の屋内プールを楽しむなら？
128127|아산 오래된 성당|Asan historic Catholic church|牙山 歴史ある カトリック教会|아산에서 언덕 위 근대 천주교 유적 보고 싶어|Where can I see a historic Catholic site on a hill in Asan?|牙山で丘の上の近代カトリック史跡を見たい
125777|정선 두 강 합류 명소|Jeongseon river confluence|旌善 川 合流 名所|정선에서 송천과 골지천이 만나는 곳은?|Where do Songcheon and Goljicheon meet in Jeongseon?|旌善で松川と骨只川が合流する場所は？
2385686|부산 기장 대나무숲|Busan Gijang bamboo forest|釜山 機張 竹林|부산에서 오래된 소나무와 편백숲을 걷고 싶어|Where can I walk among old pines and cypress trees in Busan?|釜山で古い松とヒノキの森を散策したい
2366916|대구 시내 전망 케이블카|Daegu city view cable car|大邱 市街地 展望 ケーブルカー|대구에서 산 정상에서 시내를 내려다볼 곳은?|Where can I look down over Daegu from a mountain summit?|大邱で山頂から市街地を見下ろせる場所は？
2037308|여수 박람회 수변공원|Yeosu expo waterfront park|麗水 万博 水辺公園|여수에서 빅오쇼와 스카이타워 볼 곳은?|Where can I see the Big-O show and Sky Tower in Yeosu?|麗水でビッグオーショーとスカイタワーを見られる場所は？
128792|진해 벚꽃 터널|Jinhae cherry blossom tunnel|鎮海 桜 トンネル|진해에서 하천 옆 벚꽃 야경을 즐길 곳은?|Where can I enjoy illuminated cherry blossoms by a stream in Jinhae?|鎮海で川沿いの夜桜を楽しむなら？
125794|영월 단종 왕릉|Yeongwol King Danjong royal tomb|寧越 端宗 王陵|영월에서 조선의 어린 왕이 묻힌 유적은?|Which historic site in Yeongwol is the burial place of the young Joseon king?|寧越で朝鮮王朝の若き王が眠る史跡は？
126088|부산 바다 바위섬|Busan rocky sea islets|釜山 海 岩島|부산 용호동 앞바다 독특한 지질 명소는?|What is the geological attraction off Yongho-dong in Busan?|釜山の龍湖洞沖にある地質の名所は？
989046|해남 바다 캠핑|Haenam seaside camping|海南 海辺 キャンプ|해남에서 일몰 보며 야영할 곳은?|Where can I camp and watch the sunset in Haenam?|海南で夕日を見ながらキャンプできる場所は？
3019141|울주 옹기 만들기 체험|Ulju pottery making|蔚州 甕器 作り体験|울산에서 전통 옹기 장인 작품 볼 곳은?|Where can I see traditional pottery by local artisans in Ulsan?|蔚山で伝統甕器職人の作品を見るなら？
3081515|완도 해수 스파|Wando seawater spa|莞島 海水 スパ|완도에서 해조류와 갯벌 치유 체험할 곳은?|Where can I try seaweed and tidal mud therapies in Wando?|莞島で海藻や干潟の泥のセラピーを体験するなら？
126470|서귀포 바다 돌기둥|Seogwipo sea stack|西帰浦 海 岩柱|제주 올레 7코스에 홀로 솟은 바위 어디야?|Where is the lone rock column on Jeju Olle Trail 7?|済州オルレ7コースの海にそびえる一本岩は？
2755856|거창 Y자 현수교|Geochang Y-shaped suspension bridge|居昌 Y字型 吊り橋|거창에서 협곡 위를 걷는 산악 다리 어디야?|Which mountain bridge in Geochang crosses a gorge?|居昌で渓谷の上を歩ける山岳吊り橋は？
127073|광주 옛 선교사 집|Gwangju historic missionary house|光州 旧宣教師住宅|광주에서 1920년대 서양식 주택을 보고 싶어|Where can I see a Western-style house from the 1920s in Gwangju?|光州で1920年代の洋式住宅を見たい
126276|신안 증도 해송 해변|Sinan Jeungdo pine beach|新安 曽島 松林 海岸|증도에서 깨끗한 백사장 산책할 곳은?|Where can I walk along a clean sandy beach on Jeungdo?|曽島できれいな砂浜を散歩するなら？
346906|광주 금남로 도심 사찰|Gwangju Geumnam-ro city temple|光州 錦南路 都心寺院|광주 시내에서 불교 법회에 갈 만한 절은?|Which temple in downtown Gwangju hosts Buddhist gatherings?|光州市街地で仏教の法会に参加できるお寺は？
1818945|인제 하얀 자작나무 산책|Inje white birch walks|麟蹄 白樺 散策|인제에서 숲길 걸으며 산림욕 할 곳은?|Where can I enjoy forest bathing in Inje?|麟蹄で森林浴を楽しめる散策スポットは？
1945300|부산 초량 느린 우체통|Busan Choryang slow mailbox|釜山 草梁 遅い郵便ポスト|부산에서 1년 뒤 도착하는 편지 보낼 곳은?|Where in Busan can I send a letter to be delivered in a year?|釜山で1年後に届く手紙を出せる場所は？
2535506|인천공항 신터미널|Incheon airport newer terminal|仁川空港 新ターミナル|인천공항에서 예술 전시가 있는 여객터미널은?|Which Incheon Airport passenger terminal has art displays?|仁川空港でアート展示のある旅客ターミナルは？
2763739|제주 곶자왈 숲 휴양|Jeju Gotjawal forest retreat|済州 コッチャワル 森 休養|제주에서 용암 지형 숲길과 오름을 걸을 곳은?|Where can I hike forest trails and oreum in Jeju’s lava landscape?|済州で溶岩地形の森とオルムを歩ける場所は？
128144|서울 종로 불교 사찰|Seoul Jongno Buddhist temple|ソウル 鍾路 仏教 寺院|서울 도심에서 조계종 총본산을 방문하고 싶어|Where is the head temple of the Jogye Order in central Seoul?|ソウル中心部で曹渓宗の総本山に行きたい
126240|순천 두 사찰 등산|Suncheon two-temple hike|順天 二つの寺 登山|순천에서 송광사와 선암사를 잇는 산은?|Which mountain in Suncheon lies between Songgwangsa and Seonamsa?|順天で松広寺と仙岩寺のある山は？
126351|광주 무등산 신라 사찰|Gwangju Mudeungsan Silla temple|光州 無等山 新羅 寺院|무등산 서쪽 기슭의 오래된 절 어디야?|Where is a historic temple on the western slope of Mudeungsan?|無等山の西側の麓にある古刹は？
126375|진도 전통 수묵화|Jindo traditional ink painting|珍島 伝統 水墨画|진도에서 소치 허련이 그림을 그렸던 집은?|Where in Jindo did painter Heo Ryeon live and paint?|珍島で画家・許鍊が暮らし絵を描いた家は？
128982|부안 격포 해안 절벽|Buan Gyeokpo sea cliffs|扶安 格浦 海食崖|부안에서 층층이 쌓인 암석과 해식동굴 볼 곳은?|Where can I see layered rocks and sea caves in Buan?|扶安で層状の岩と海食洞窟を見るなら？
126214|경주 신라 공개 고분|Gyeongju open Silla tomb|慶州 新羅 公開古墳|경주에서 금관과 날개 달린 말 그림이 발견된 무덤은?|Which Gyeongju tomb yielded a gold crown and flying horse painting?|慶州で金冠と空飛ぶ馬の絵が発見された古墳は？
127811|봉화 깊은 산 계곡|Bonghwa remote mountain valleys|奉化 山奥 渓谷|봉화에서 원시림과 야영장이 있는 산은?|Which mountain in Bonghwa has old-growth forest and camping?|奉化で原生林とキャンプ場のある山は？
126004|제천 전통문화 유적|Jecheon traditional heritage site|堤川 伝統文化 遺跡|제천에서 충주호 전망과 옮겨 복원한 옛 건축을 볼 곳은?|Where in Jecheon can I see relocated historic buildings and lake views?|堤川で移築復元された古建築と湖の景色を見るなら？
4092567|서울 연희동 퍼스널컬러|Seoul Yeonhui personal color analysis|ソウル 延禧洞 パーソナルカラー|서울에서 외국어로 어울리는 메이크업 색을 진단받고 싶어|Where in Seoul can I get makeup color analysis in a foreign language?|ソウルで外国語対応のメイクカラー診断を受けたい
2931362|여수 숨은 하얀 등대|Yeosu hidden white lighthouse|麗水 穴場 白い灯台|돌산도에서 소나무 숲길 끝 바다를 보고 싶어|Where on Dolsando can I see the sea at the end of a pine trail?|突山島で松林の小道の先に海が見える場所は？
2569503|파주 시계 검 박물관|Paju clocks and swords museum|坡州 時計 刀剣 博物館|파주에서 옛 회중시계를 전시하는 곳은?|Where can I see antique pocket watches in Paju?|坡州で昔の懐中時計を見られる場所は？
2769846|제천 숲속 야외 스파|Jecheon outdoor forest spa|堤川 森 屋外スパ|제천에서 숲과 계곡을 보며 온수풀에서 쉬고 싶어|Where can I relax in a warm pool with forest and valley views in Jecheon?|堤川で森と渓谷を眺めながら温水プールで休みたい
127951|강릉 바다 미술관|Gangneung seaside art museum|江陵 海辺 美術館|강릉에서 피노키오 인형 전시를 볼 곳은?|Where can I see a Pinocchio puppet exhibition in Gangneung?|江陵でピノキオ人形の展示を見るなら？
1081744|평창 한식 만들기 체험|Pyeongchang Korean cooking class|平昌 韓国料理 作り体験|평창에서 한옥에 머물고 김치도 만들고 싶어|Where can I stay in a hanok and make kimchi in Pyeongchang?|平昌で韓屋に泊まりキムチ作りも体験したい
126469|한라산 초보 등산 코스|Hallasan beginner hiking trail|漢拏山 初心者 登山コース|한라산에서 기암과 윗세오름을 볼 코스는?|Which Hallasan trail passes dramatic rock formations and Witseoreum?|漢拏山で奇岩とウィッセオルムを見られるコースは？
128025|여수 동백꽃 섬|Yeosu camellia island|麗水 ツバキ 島|여수에서 겨울에도 붉은 동백꽃 볼 곳은?|Where can I see red camellias in Yeosu even in winter?|麗水で冬にも赤いツバキを見られる場所は？
2589657|서울 맞춤 한방 마사지|Seoul personalized herbal massage|ソウル オーダーメイド 韓方マッサージ|서울에서 체질에 맞는 한방 스파 받을 곳은?|Where can I get a herbal spa tailored to my constitution in Seoul?|ソウルで体質に合った韓方スパを受けるなら？
126451|제주 에메랄드빛 얕은 바다|Jeju shallow turquoise beach|済州 遠浅 エメラルド色 海|제주에서 야자수와 흰 모래를 보고 싶어|Where can I see palm trees and white sand in Jeju?|済州でヤシの木と白い砂浜を見たい
918805|서산 천주교 순교 유적|Seosan Catholic martyrdom site|瑞山 カトリック 殉教 史跡|충남에서 천주교 박해 역사를 알아볼 순례지는?|Where in Chungnam can I learn about the persecution of Catholics?|忠清南道でカトリック迫害の歴史を学べる巡礼地は？
128661|정선 옛 금광 동굴|Jeongseon former gold mine cave|旌善 旧金鉱山 洞窟|정선에서 금 채굴 역사와 석회동굴을 함께 볼 곳은?|Where in Jeongseon can I explore mining history and a limestone cave?|旌善で採鉱の歴史と石灰洞窟を一緒に楽しめる場所は？
264394|수원 화성 수문 야경|Suwon Hwaseong floodgate night view|水原 華城 水門 夜景|수원에서 무지개 모양 수문 일곱 개를 볼 곳은?|Where in Suwon can I see seven arched floodgates?|水原で虹のようなアーチ形の水門を7つ見られる場所は？
2456767|부산 옛 공장 문화공간|Busan former factory cultural space|釜山 旧工場 文化施設|부산 수영구에서 전시와 대나무 정원을 즐길 곳은?|Where in Busan’s Suyeong district can I enjoy exhibitions and a bamboo garden?|釜山の水営区で展示と竹の庭を楽しめる場所は？
4131993|제주 실내 미디어아트|Jeju indoor media art|済州 屋内 メディアアート|제주에서 감귤과 바다 테마로 사진 찍을 실내 전시장은?|Where in Jeju is an indoor exhibit for photos with citrus and sea themes?|済州でミカンや海をテーマに写真を撮れる屋内展示は？
2660731|서울 홍삼 스파|Seoul red ginseng spa|ソウル 紅参 スパ|서울에서 홍삼 오일로 전신 관리 받을 곳은?|Where in Seoul can I get a full-body treatment with red ginseng oil?|ソウルで紅参オイルの全身ケアを受けられる場所は？'''

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);return reader.fieldnames,list(reader)

def main():
    fields,previous=read(EXISTING)
    _,all_docs=read(STANDARD)
    ko_docs={r['place_id']:r for r in all_docs if r['language']=='ko'}
    assert len(ko_docs)==100 and len(previous)==30
    new={}
    for line in DATA.splitlines():
        parts=line.split('|');assert len(parts)==7,parts
        pid,*q=parts;assert pid not in new;new[pid]=q
    prior_ids={r['place_id'] for r in previous}
    assert set(new)==set(ko_docs)-prior_ids,(len(new),set(ko_docs)-prior_ids-set(new),set(new)-set(ko_docs))
    rows=[dict(r) for r in previous]
    for i,(pid,q) in enumerate(new.items(),16):
        doc=ko_docs[pid]
        for j,(style,queries) in enumerate((('키워드형',q[:3]),('짧은 문장형',q[3:])),1):
            rows.append(dict(intent_id=f'B{i:02d}-{j}',place_id=pid,title_ko=doc['title'],reference_description_ko=doc['description'],query_ko=queries[0],query_en=queries[1],query_ja=queries[2],annotator_note=f'AI 생성 / {style} / 검토 전 / 정답은 대표 관광지이며 유사 장소 가능'))
    assert len(rows)==200 and len({r['intent_id'] for r in rows})==200
    assert set(Counter(r['place_id'] for r in rows).values())=={2}
    assert set(r['place_id'] for r in rows)==set(ko_docs)
    assert all(r[f].strip() for r in rows for f in ('query_ko','query_en','query_ja'))
    assert len({r[f] for r in rows for f in ('query_ko','query_en','query_ja')})==600
    assert rows[:30]==previous
    with OUTPUT.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    assert read(OUTPUT)[1]==rows
    print(OUTPUT,'places',len(ko_docs),'intents',len(rows),'queries',3*len(rows))
    print('query mean lengths',[(f,round(statistics.mean(len(r[f]) for r in rows),1)) for f in ('query_ko','query_en','query_ja')])

if __name__=='__main__': main()
