"""Conservative phrase-based feature audit; labels are not ranked by retrieval score.

Each intent has one or more explicit core concepts with language-specific
equivalent phrases. Unmatched phrases are NOT automatically proven absent.
"""
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data/processed'
Q=P/'tourapi_ai_generated_queries_600_natural.csv'
DOC=P/'tourapi_standardized.csv'
DETAIL=P/'tourapi_pilot_details_300.csv'
OUT=ROOT/'results/tourapi_feature_coverage'
LANGS=('ko','en','ja')

# intent | core feature(s) in Korean | KO regex | EN regex | JA regex
# ";" separates multiple required concepts. Within each concept, | means OR.
# Use ~ instead of | for alternatives in regex to preserve the five columns.
RULES=r'''B01-1|두피;마사지|두피;마사지|scalp;massag|頭皮;マッサージ
B01-2|피부;진단·분석|피부;진단~분석|skin;analy[sz]~diagnos|肌~皮膚;分析~診断
B02-1|선교사;기념|선교사;기념|missionar;memorial~memory~commemorat|宣教師;記念
B02-2|회색 벽돌;예배당|회색\s*벽돌;예배당|gr[ae]y.{0,12}brick;chapel~worship|灰色.{0,8}レンガ~灰色.{0,8}煉瓦;礼拝堂
B03-1|해금강;조망|해금강;전망~조망~내려다|haegeumgang;view~overlook~observator|海金剛~ヘグムガン;展望~見渡~眺め
B03-2|부담 없는 짧은 등산|30\s*분~부담.{0,10}없~힘들지~어렵지|30.minute~half an hour~easy hike~not difficult~gentle walk|30分~気軽~負担.{0,8}ない~険しくない
B04-1|웰니스·건강 휴양|웰니스~건강.{0,12}휴양~힐링|wellness~health.{0,25}relax~healing|ウェルネス~健康.{0,15}休養~ヒーリング
B04-2|편백;명상|편백;명상|cypress~hinoki;meditat|ヒノキ~檜;瞑想
B05-1|케이블카;바다 조망|케이블카;바다~다도해~해상|cable car;sea~ocean~archipelago|ケーブルカー;海~多島海
B05-2|유달산;섬|유달산;고하도~섬~다도해|yudalsan;gohado~island~archipelago|儒達山~ユダルサン;高下島~島~多島海
B06-1|두륜산;사찰|두륜산;사찰~절|duryunsan~duryun;temple|頭輪山~トゥリュンサン;寺~仏閣
B06-2|차 문화|차\s*문화~다도~초의선사|tea culture~tea ceremony~tea master~choui|茶文化~茶道~草衣
B07-1|석회동굴|석회암~석회동굴~석회석|limestone|石灰岩~石灰洞窟
B07-2|임진왜란;피난|임진왜란;피난~숨|imjin~japanese invasions~japanese invasion~1592;refuge~shelter~hid|壬辰倭乱~文禄~慶長;避難~身を隠~逃れ
B08-1|신라;사찰|신라;사찰~절|silla;temple|新羅;寺
B08-2|샘물;전설|샘~감로;전설~솟아나지~구렁이|spring~water;legend~snake~stopped flowing~disappear|泉~甘露;伝説~大蛇~出なく~消え
B09-1|일출;사찰|일출~해돋이;사찰~도량~암|sunrise;temple~hermitage~sanctuary|日の出~朝日;寺~庵~道場
B09-2|거북 형태 바위|거북.{0,12}(바위~등)~바위.{0,12}거북|turtle.{0,20}(rock~shell)~rock.{0,30}turtle|亀.{0,15}(岩~甲羅)~岩.{0,15}亀
B10-1|철길;복고|철길~철로~철도;레트로~옛~추억|railway~railroad~tracks;retro~nostalg~old~former|線路~鉄道;レトロ~昔~懐かし~廃線
B10-2|철길;옛 교복|철길~철도~철로;교복|railway~railroad~tracks;school uniform~vintage uniform|線路~鉄道;制服~学生服
B11-1|동악산;사찰|동악산;절~사찰|dongaksan;temple|動楽山~トンアクサン;寺
B11-2|계곡;폭포|계곡;폭포|valley~gorge;waterfall|渓谷~谷;滝~瀑布
B12-1|허브;아로마|허브;아로마|herb;aroma|ハーブ;アロマ
B12-2|체질 맞춤;스파|체질~맞춤;스파|constitution~personaliz~customiz~tailor;spa|体質~一人ひとり~個別;スパ
B13-1|대게;어시장·위판|대게;어시장~위판|snow crab~king crab;fish market~auction|ズワイガニ~大ゲ~テゲ;魚市場~競り~せり
B13-2|BTS;뮤직비디오|bts~방탄;뮤직\s*비디오~뮤비|bts~bangtan;music video~\bmv\b|bts~防弾;ミュージックビデオ~\bmv\b
B14-1|야간 크루즈;불꽃|야경~밤바다~야간;불꽃|night~evening;firework|夜景~夜の海~ナイト;花火
B14-2|유람선 식당|레스토랑~식당~선상.{0,8}식사|restaurant~dining|レストラン~食事
B15-1|강릉·경포 해변|해변~해수욕장~바닷가|beach|海岸~ビーチ~海水浴場
B15-2|스쿠버다이빙|스쿠버~다이빙|scuba~diving|スキューバ~ダイビング
B16-1|청보리|청보리~보리밭|green barley~barley field|青麦~麦畑
B16-2|자전거로 둘러보기|자전거|bicycle~bike~cycling|自転車~サイクリング
B17-1|조선;군사 지휘부·병영|조선;병영~육군~지휘부|joseon;military~army~headquarters|朝鮮;兵営~軍事~軍隊
B17-2|성곽|성곽~성벽|fortress wall~fortification~wall|城壁~城郭
B18-1|녹차밭|차밭~다원~녹차밭|tea plantation~tea field~tea garden|茶畑~茶園
B18-2|찻잎 따기|찻잎.{0,8}따~녹찻잎.{0,8}따~차따기|pick.{0,18}tea~tea leaf picking~tea picking|茶摘み~茶葉を摘
B19-1|포대|포대~진보|artillery~battery~fort|砲台~鎮
B19-2|강화해협;방어·수비|강화해협;지키~방어~요충|ganghwa strait~ganghwahaehyeop;guard~protect~defen~strategic|江華海峡;守~防衛~要衝
B20-1|라벤더|라벤더|lavender|ラベンダー
B20-2|한옥;꽃 정원·카페|한옥;정원~카페|hanok~traditional korean house;garden~cafe~café|韓屋;庭園~カフェ
B21-1|떡갈비;전문점·골목|떡갈비;골목~전문점|tteokgalbi~tteok-galbi~rib patt;street~alley~restaurant|トッカルビ~餅カルビ;通り~専門店~店
B21-2|뼛국|뼛국~뼈국|bone (soup~broth)~soup.{0,20}bone|骨.{0,8}スープ~骨汁
B22-1|놀이공원|놀이공원~테마파크|amusement park~theme park|遊園地~テーマパーク
B22-2|눈썰매|눈썰매|snow sled~snow sleigh~sledding|雪そり~雪ソリ~スノースレッド
B23-1|황토;한증막|황토;한증막|yellow clay~red clay~loess;sauna~steam|黄土;サウナ~汗蒸幕
B23-2|인절미 만들기|인절미|injeolmi~rice cake.{0,25}mak~mak.{0,25}rice cake|インジョルミ~餅つき~餅作り
B24-1|문학·작가|문학~작가|literar~author~writer|文学~作家
B24-2|동백꽃;생가|동백꽃;생가|camellia;birthplace~birth home~birth house~childhood home|椿~ツバキ;生家
B25-1|편백;맨발|편백;맨발|cypress~hinoki;barefoot|ヒノキ~檜;裸足
B25-2|족욕|족욕|foot bath~footbath~foot soak|足湯~足浴
B26-1|야외 조각|조각~조형물|sculptur|彫刻~造形
B26-2|미술·조각;해양 전망|조각~조형물~예술;조망~경관~한려수도~바다|art~sculptur;sea~ocean~island~coastal view|彫刻~芸術;海~島~眺望
B27-1|세계유산 성곽|산성~성곽|fortress|山城~城郭~城壁
B27-2|산성 걷기|산성~성곽~성벽|fortress|山城~城郭~城壁
B28-1|대운산;고찰|대운산;고찰~사찰|daeunsan~daeun;temple~hermitage|大雲山~テウンサン;古刹~寺~庵
B28-2|팽나무 고목|팽나무|hackberry~nettle tree|エノキ~榎
B29-1|녹색 에너지|녹색\s*에너지~신재생\s*에너지|green energy~renewable energy~sustainable energy|グリーンエネルギー~新再生エネルギー~再生可能エネルギー
B29-2|에너지 절약|에너지\s*절약|energy saving~saving energy~energy conservation|省エネ~エネルギー節約~エネルギーの節約
B30-1|탄산 약수|탄산|carbonat~carbon dioxide|炭酸
B30-2|산불 피해|산불~전소|wildfire~forest fire~burned down~burnt down|山火事~山林火災~全焼
B31-1|카라반|카라반~캐러밴|caravan~camper|キャラバン~キャンピングカー
B31-2|계곡;숙소|계곡;숲\s*속.{0,5}집~숙소~휴양관|valley~stream;accommodation~cabin~lodge~forest house|渓谷;宿泊~森の家~休養館
B32-1|국립공원;산|국립공원;산|national park;mountain|国立公園;山
B32-2|구천동;폭포|구천동;폭포|gucheondong~gucheondong;waterfall|九千洞~クチョンドン;滝~瀑布
B33-1|온천;워터파크|온천;워터파크~물놀이|hot spring;water park~waterpark|温泉;ウォーターパーク~水遊び
B33-2|실내외;수영장|실내;실외~야외|indoor;outdoor|屋内~室内;屋外
B34-1|남한강;바위 봉우리|남한강;바위~봉우리|namhan;peak~rock|南漢江~ナムハンガン;岩~峰
B34-2|정자|정자~삼도정|pavilion|亭子~三島亭
B35-1|섬;해안도로|섬;해안.{0,6}도로~일주도로|island;coastal road~scenic drive~road around|島;海岸道路~周遊道路~一周道路
B35-2|다리 연결|대교~다리~이어져|bridge|大橋~橋
B36-1|소나무 해변|노송~소나무~송림|pine|松林~松の木~老松
B36-2|잔잔한 바다|잔잔~수심이 얕|calm~gentle wave~shallow|穏やか~波が静か~遠浅
B37-1|퍼레이드|퍼레이드|parade|パレード
B37-2|놀이기구·스릴|놀이기구~스릴|ride~attraction~thrill|アトラクション~乗り物~スリル
B38-1|용암동굴|용암동굴~용암\s*동굴|lava tube~lava cave|溶岩洞窟~溶岩洞
B38-2|용암석주|용암석주~용암\s*석주|lava column~lava pillar|溶岩石柱~溶岩柱
B39-1|조선;정자|조선;정자|joseon;pavilion|朝鮮;亭子~東屋
B39-2|김계행|김계행|kim.{0,5}gye.{0,4}haeng|金係行~金系行~キムゲヘン
B40-1|고려;서원·학자|고려;서원~학자|goryeo;academy~scholar|高麗;書院~学者
B40-2|이색|이색|yi sae[kc]~lee sae[kc]~yi saek|李穡~イセク
B41-1|석회동굴|석회암~석회동굴|limestone|石灰岩~石灰洞窟
B41-2|계란프라이 석순|에그프라이~계란\s*프라이~달걀\s*프라이|egg.fry~fried.egg~egg.shaped|目玉焼き~目玉焼
B42-1|화랑·미술거리|화랑~미술~화방|art~galler|画廊~美術~ギャラリー
B42-2|반월당;미술품|반월당;미술~화랑|banwoldang;art~galler|半月堂~パノルダン;美術~画廊
B43-1|한옥;골목|한옥;골목~거리|hanok~traditional korean house;alley~street~lane|韓屋;路地~通り
B43-2|경복궁;창덕궁·종묘|경복궁;창덕궁~종묘|gyeongbokgung;changdeokgung~jongmyo|景福宮~キョンボックン;昌徳宮~チャンドックン~宗廟
B44-1|카페거리|카페|cafe~café|カフェ
B44-2|청담동 같은 고급 분위기|청담동~고급스러|cheongdam~upscale~luxurious~sophisticated|清潭洞~高級~洗練
B45-1|상사화|상사화~꽃무릇|spider lil~lycoris~cluster amaryllis|キツネノカミソリ~ヒガンバナ~石蒜~相思華~サンソファ
B45-2|백제;불교 전래|백제;마라난타~불교.{0,8}전래~첫.{0,8}불법|baekje;marananta~buddhism.{0,20}introduc~introduc.{0,25}buddhism|百済;摩羅難陀~仏教.{0,10}伝来
B46-1|현무암;폭포|현무암;폭포|basalt;waterfall|玄武岩;滝~瀑布
B46-2|움푹 들어간 협곡|협곡~주머니~움푹|gorge~canyon~pocket~bowl|峡谷~くぼ~ポケット
B47-1|산림 치유|치유~힐링|forest therapy~healing~forest wellness|森林セラピー~癒し~森林治癒
B47-2|트리하우스|트리하우스|tree.?house|ツリーハウス
B48-1|정미소;카페|정미소;카페|rice mill;cafe~café|精米所;カフェ
B48-2|골동품;다락방|골동품~민속품;다락|antique~vintage;attic|骨董~アンティーク;屋根裏~ロフト
B49-1|케이블카;해상|케이블카;바다~해상|cable car;sea~ocean~marine|ケーブルカー;海~海上
B49-2|갯벌|갯벌|mudflat~mud flat~tidal flat|干潟
B50-1|일출;화산|일출~해돋이;화산~분화구|sunrise;volcan~crater|日の出~朝日;火山~火口
B50-2|분화구|분화구|crater|火口~噴火口
B51-1|낙조·일몰|낙조~일몰~해질|sunset|夕日~夕陽~日没
B51-2|섬;일몰|섬;낙조~일몰~해질|island;sunset|島;夕日~夕陽~日没
B52-1|생태;교육·체험|생태;교육~체험|ecolog~nature;education~experienc~learning|生態~自然;教育~体験~学習
B52-2|숲 명상|명상|meditat|瞑想
B53-1|별 관측|별~천문~관측|star~astronom~observatory|星~天文~観測
B53-2|연화봉;망원경|연화봉;망원경|yeonhwabong;telescope|蓮花峰~ヨンファボン;望遠鏡
B54-1|관람차;바다|관람차;바다|ferris wheel;sea~ocean|観覧車;海
B54-2|설악산;바다|설악산;바다|seoraksan;sea~ocean|雪岳山~ソラクサン;海
B55-1|온천;족욕|온천;족욕|hot spring;foot bath~footbath~foot soak|温泉;足浴~足湯
B55-2|조선 왕 온천|태조~이성계~조선왕조|king taejo~yi seong-gye~joseon dynasty|太祖~李成桂~朝鮮王朝
B56-1|워터파크|워터파크~물놀이|water park~waterpark|ウォーターパーク~水遊び
B56-2|가우디|가우디~구엘공원|gaud[ií]~g[üu]ell|ガウディ~グエル
B57-1|성당·천주교|성당~천주교|catholic~church|カトリック~聖堂~教会
B57-2|언덕;성당|언덕;성당~본당|hill;church|丘;聖堂~教会
B58-1|두 강 합류|합류~어우러|confluence~converge~meet~join~merge|合流~二つの川
B58-2|송천;골지천|송천;골지천|songcheon;goljicheon|松川~ソンチョン;骨只川~コルジチョン
B59-1|대나무숲|대나무|bamboo|竹林~竹
B59-2|소나무;편백|소나무~금강송;편백|pine;cypress~hinoki|松;ヒノキ~檜
B60-1|케이블카;도시 전망|케이블카;대구~도심~시내|cable car;daegu~city|ケーブルカー;大邱~市街地
B60-2|산 정상 전망;도심|정상~전망;시내~도시~대구|summit~view~observator;city~daegu|山頂~展望;市街地~大邱
B61-1|박람회;수변공원|박람회~엑스포;수변공원~바다|expo;waterfront~ocean~sea|博覧会~エキスポ;水辺~海
B61-2|빅오;스카이타워|빅\s*오;스카이\s*타워|big.?o;sky tower|ビッグ.?オー;スカイタワー
B62-1|벚꽃 터널|벚꽃~벚나무|cherry blossom|桜~サクラ
B62-2|하천;야경|하천~여좌천;야경~조명~밤|stream~yeojwacheon;night~light~illuminat|川~余佐川;夜~照明~ライトアップ
B63-1|왕릉·능|왕릉~단종.{0,6}능|royal tomb~king danjong.{0,20}tomb|王陵~端宗.{0,10}陵
B63-2|어린 조선 왕|단종|danjong|端宗~タンジョン
B64-1|바위섬|바위섬~암석섬|rocky island~rock island~islet|岩島~岩の島~岩礁
B64-2|해안 지질|침식~지질~해식~파도|erosion~geolog~wave|侵食~浸食~地質~海食~波
B65-1|해안;캠핑|바다~다도해;캠핑~야영|sea~ocean~coast;camp|海;キャンプ~野営
B65-2|일몰;야영|일몰~노을~해가 질;캠핑~야영|sunset;camp|夕日~夕陽~日没;キャンプ~野営
B66-1|옹기;체험|옹기;체험~빚|onggi~pottery;experienc~mak~workshop|甕器~オンギ~陶器;体験~作り
B66-2|옹기 장인 작품|장인~옹기장~3대째|artisan~craftsman~third.generation~potter|職人~3代目~匠
B67-1|해수;스파·치유|해수;스파~치유~테라피|seawater~sea water;spa~therap~wellness|海水;スパ~セラピー~治癒
B67-2|해조류;머드|해조류;머드~갯벌|seaweed;mud~tidal flat|海藻;泥~マッド~干潟
B68-1|바다 돌기둥|돌기둥~바위.{0,8}솟|sea stack~rock pillar~rock column~rock.{0,20}sea|柱状~岩柱~岩.{0,15}海~海.{0,15}岩
B68-2|올레 7코스|올레.{0,8}7|olle.{0,8}7~course 7|オルレ.{0,8}7
B69-1|Y자형;현수교|y자~y 자;출렁다리~현수교|y.shaped;bridge|y字~y型;吊り橋~つり橋
B69-2|협곡;다리|협곡;다리|gorge~valley~canyon;bridge|峡谷~渓谷;橋
B70-1|선교사;주택|선교사;주택~사택|missionar;house~residence|宣教師;住宅~私宅~家
B70-2|1920년대;서양식|1920;서양식~서양풍|1920;western.style|1920;洋式~洋風
B71-1|해송 해변|해송~소나무|pine|松林~松~海松
B71-2|백사장|백사장~모래사장|white sand~sandy beach|白砂~白い砂~砂浜
B72-1|도심;사찰|금남로~번화가~도심;사찰~절~포교당|downtown~city center~geumnam;temple|繁華街~都心~錦南路;寺~布教
B72-2|법회·포교 활동|법회~포교~참배~모임|buddhist service~dharma~worship~religious gathering~missionary|法会~布教~参拝~集会
B73-1|자작나무|자작나무|birch|白樺~シラカバ
B73-2|산림욕|산림욕~힐링|forest bathing~healing|森林浴~癒し
B74-1|느린 우체통|느린\s*우체통~1년 뒤|slow mailbox~slow postbox~one year later~after a year|遅い郵便~1年後~一年後
B74-2|문학·시인;전시|시인~문학~유치환;작품\s*전시~갤러리|poet~poetry~literar;exhibit~gallery|詩人~文学~詩;展示~ギャラリー
B75-1|여객터미널|여객터미널|passenger terminal~terminal|旅客ターミナル~ターミナル
B75-2|예술 요소|아트포트~예술~아트|artport~art exhibit~artwork|アートポート~芸術~アート
B76-1|곶자왈;휴양림|곶자왈;휴양림~숲|gotjawal;forest~recreation|コッチャワル~ゴッチャワル;森~休養林
B76-2|용암 지형;오름|용암;오름|lava;oreum|溶岩;オルム
B77-1|도심;불교 사찰|종로~도심;불교~사찰|jongno~downtown~central seoul;temple~buddhis|鍾路~都心;寺~仏教
B77-2|조계종 총본산|조계종.{0,10}총본산|head temple.{0,30}jogye~jogye.{0,30}head temple|曹渓宗.{0,10}総本山
B78-1|송광사;선암사|송광사;선암사|songgwangsa;seonamsa|松広寺~ソングァンサ;仙岩寺~ソナムサ
B78-2|송광사;선암사|송광사;선암사|songgwangsa;seonamsa|松広寺~ソングァンサ;仙岩寺~ソナムサ
B79-1|무등산;신라|무등산;신라|mudeungsan;silla|無等山~ムドゥンサン;新羅
B79-2|무등산 서쪽|서쪽~서편|west~western|西側~西麓~西の
B80-1|수묵화·서화|서화~수묵~남화~그림|ink painting~painting~calligraph|水墨画~書画~南画~絵
B80-2|허련|허련~허유|heo ryeon~heo ryeon~heo yu~heoryeon|許鍊~許錬~許維~ホリョン
B81-1|해안 절벽|해식애~절벽|cliff|海食崖~断崖~絶壁
B81-2|해식동굴;퇴적층|해식동굴;퇴적~지층~암층|sea cave;sediment~layer~strata|海食洞窟;堆積~地層~層状
B82-1|신라;고분|신라;고분~무덤|silla;tomb|新羅;古墳~墓
B82-2|금관;말 그림|금관;천마도~하늘을 나는 말|gold crown;heavenly horse~flying horse~horse painting~horse.{0,20}paint|金冠;天馬~空を飛ぶ馬~天を飛ぶ馬
B83-1|산;계곡|산;계곡|mountain;valley~gorge|山;渓谷
B83-2|원시림;캠핑|원시림;야영~캠핑|primeval forest~virgin forest~old.growth;camp|原生林;キャンプ~野営
B84-1|문화유산·유적|문화유산~유적~문화재|heritage~historic site~cultural property|文化遺産~史跡~文化財
B84-2|충주호;이전 복원|충주호;이전~복원~옮겨|chungju lake~lake chungju;relocat~restor~reconstruct|忠州湖~チュンジュ湖;移転~移築~復元
B85-1|퍼스널컬러|퍼스널\s*컬러|personal color~personal colour|パーソナルカラー
B85-2|외국어;메이크업 컬러|외국인~영어~일본어~중국어;메이크업~컬러|english~japanese~chinese~foreign language;makeup~make.up~color|英語~日本語~中国語~外国語;メイク~カラー
B86-1|하얀 등대|하얀\s*등대~흰\s*등대~등대|white.{0,12}lighthouse~lighthouse|白い灯台~灯台
B86-2|소나무 숲길;바다|소나무~숲길;바다~수평선|pine~forest trail;sea~ocean~horizon|松林~松の木~森の小道;海~水平線
B87-1|시계;검 박물관|시계;검~칼|clock~watch;sword~blade|時計;刀剣~剣~刀
B87-2|회중시계|회중시계|pocket watch|懐中時計
B88-1|숲;스파|숲~포레스트;스파|forest;spa|森~フォレスト;スパ
B88-2|계곡;야외 풀|계곡;인피니티풀~야외|valley;infinity pool~outdoor pool|渓谷;インフィニティプール~屋外プール
B89-1|바다;미술관|바다;미술관~예술|sea~ocean;art museum~art park~art gallery|海;美術館~アート
B89-2|피노키오 인형|피노키오|pinocchio|ピノキオ
B90-1|전통음식 만들기|음식.{0,8}만들기~한식~전통음식|korean food~cooking~food making|韓国料理~伝統料理~伝統飲食
B90-2|한옥 숙박;김치 만들기|한옥;김치|hanok;kimchi|韓屋;キムチ
B91-1|초보 등산|초보자~수월~짧은|beginner~easy~shortest|初心者~気軽~最短
B91-2|기암;윗세오름|기암;윗세오름|rock formation~strange rock~rocky cliff;witseoreum~witsaeoreum~witseorum|奇岩;ウィッセオルム~ウィッセオルム
B92-1|동백꽃 섬|동백|camellia|ツバキ~椿
B92-2|겨울;동백|겨울~10월~2월;동백|winter~october~february;camellia|冬~10月~2月;ツバキ~椿
B93-1|한방;마사지|한방;마사지|herbal~traditional korean medicine;massage|韓方;マッサージ
B93-2|체질 맞춤|체질~맞춤|constitution~customiz~tailor|体質~オーダーメイド
B94-1|에메랄드 바다;얕은 물|에메랄드;수심.{0,5}얕|emerald~turquoise;shallow|エメラルド;浅~遠浅
B94-2|야자수;흰 모래|야자수;흰~하얀~백사장|palm;white sand|ヤシ~椰子;白い砂~白砂
B95-1|천주교;순교|천주교;순교~처형~박해|catholic;martyr~persecut|カトリック;殉教~迫害
B95-2|박해 순례 역사|박해|persecut|迫害
B96-1|금광;동굴|금광~광산;동굴|gold mine~gold mining;cave|金鉱~鉱山;洞窟
B96-2|채굴 체험;석회동굴|채굴~금광맥;석회|mining~extraction;limestone|採掘~金鉱;石灰
B97-1|수문;야경|수문;야경~조명|floodgate~water gate~sluice;night~illuminat~lighting|水門;夜景~照明~ライトアップ
B97-2|무지개 아치 수문|무지개~아치|rainbow~arch|虹~アーチ
B98-1|공장;문화시설|공장;문화~예술|factory;art~culture|工場;文化~芸術
B98-2|전시;대나무|전시;대나무|exhibit;bamboo|展示;竹
B99-1|미디어아트|미디어\s*아트|media art|メディアアート
B99-2|감귤;사진|감귤;사진~포즈~촬영|citrus~tangerine~mandarin;photo~pose|ミカン~みかん~柑橘;写真~撮影~ポーズ
B100-1|홍삼 스파·케어|홍삼|red ginseng|紅参~紅蔘
B100-2|홍삼 오일;전신 관리|홍삼\s*오일;전신~풀케어|red ginseng oil;full.body~whole.body~total care|紅参オイル~紅蔘オイル;全身'''

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def normalize(t):
    return unicodedata.normalize('NFKC',t)

def match_groups(text,patterns):
    text=normalize(text)
    parts=[]
    for pattern in patterns:
        m=re.search(pattern.replace('~','|'),text,flags=re.I)
        parts.append((m.group(0),text[max(0,m.start()-45):min(len(text),m.end()+55)]) if m else ('',''))
    return parts

# Manual overrides are grounded in full source text; initially only previously
# inspected cases. Distinguish source-confirmed missing from unmatched regex.
MANUAL={
 ('B02-2','ja'):('PARTIAL','1914년 선교사 기념 서양식 건물과 예배당으로 추정된다는 설명은 있으나 회색 벽돌 특징 없음'),
 ('B02-2','en'):('CONFLICT','영문은 white two-story building, 국문은 회색 벽돌. 예배당은 명시되나 색상 정보 충돌'),
 ('B07-2','en'):('MISSING','영문은 천연기념물 지정 및 석회동굴 구조 중심. 임진왜란 피난 이야기는 없음'),
 ('B07-2','ja'):('MISSING','일문은 천연기념물 지정·다층 구조·생성물 중심. 임진왜란 피난 이야기는 없음'),
 ('B10-2','ja'):('PARTIAL','일문은 철도 건설 및 폐선 역사만 설명하고 교복 체험 없음'),
}

# Full target-description follow-up on every initially unresolved candidate.
# P = present, A = partially present, M = distinctive feature absent,
# C = conflicting source information, R = unresolved semantic equivalence.
FOLLOWUP='''B01-1|en|A|head spa 및 두피 순환·근육 이완은 설명하나 마사지라는 구체적 방법은 명시하지 않음
B03-1|en|P|Haegumgang 철자 변형과 panoramic view 명시
B03-2|en|A|camellia forest path는 있으나 짧은 등산 시간·난이도·가족 이용 가능 정보 없음
B09-2|en|M|일출과 사찰 역사만 설명; 거북 등 모양 바위 언급 없음
B11-1|en|A|사찰과 곡성 위치는 있으나 동악산 산명 없음
B11-2|en|A|nearby valley는 있으나 폭포 정보 없음
B12-1|en|A|herbal plant museum은 있으나 아로마테라피 체험 없음
B12-2|en|M|식물원·전시관 중심; 체질 상담 및 맞춤 스파 없음. space를 spa로 오인한 자동 매칭 배제
B14-2|en|M|유람선 투어와 4층 구조는 설명하나 선내 레스토랑·식사 없음
B17-2|en|P|fortress spans over a 1,060m in length 및 보존된 기초 설명으로 성곽 확인
B18-2|en|M|녹차밭·재배·풍경은 있으나 찻잎 따기 체험 없음
B22-2|en|P|sledge park가 눈썰매장 표현에 대응
B26-2|en|A|조각 작품 설명은 있으나 해양 조망 없음
B30-2|en|M|탄산 약수 발생 설명만 있고 산불 피해 공지 없음
B32-2|en|A|36km valley 및 waterfalls는 있으나 구천동 명칭 없음
B38-2|en|P|7.6-meter-tall stone pillar가 용암동굴 문맥에서 용암석주에 대응
B40-1|en|A|고려 학자 이곡·이색과 서원은 설명하나 Goryeo 왕조명 없음
B42-2|en|A|미술·갤러리 거리는 있으나 반월당역 없음
B43-1|en|A|traditional houses와 실제 주거 neighborhood는 있으나 골목 특성은 명시되지 않음
B44-2|en|M|유럽풍 exotic 분위기는 있으나 청담동 비유나 고급스러운 분위기 정보 없음
B45-1|en|P|magic lily 및 late summer bloom으로 꽃 관람 특징 확인; 식물 종의 정확한 동의어 여부는 별개
B46-1|en|A|columnar joints와 waterfall는 있으나 현무암 암석 종류 명시 없음
B46-2|en|A|small valley와 지형 명칭 유래는 있으나 움푹 파인 형상 설명 없음
B48-2|en|A|antique art works 및 second floor photo zone은 있으나 다락방 명시 없음
B53-2|en|A|modern telescope는 있으나 연화봉 명칭 없음
B57-2|en|A|Catholic Church 및 숲 풍경은 있으나 언덕 위 위치가 명시되지 않음
B58-2|en|A|평창·삼척 물줄기 합류는 설명하나 송천·골지천 두 하천 고유명 없음
B60-1|en|A|도시 전망대는 설명하나 케이블카 운행 정보 없음; 제목에는 cable car 있음
B62-2|en|A|Yeojwacheon Stream 벚꽃은 설명하나 야간 조명 없음
B64-2|en|A|6개 바위섬 및 조수 차이는 설명하나 침식·지질 형성 정보 없음
B67-2|en|P|marine algae와 mud가 해조류·머드에 대응
B68-2|en|M|트레킹은 나오나 제주 올레 7코스 구체 정보 없음
B69-1|en|P|Y-shape 및 three sections arranged 확인
B72-1|en|A|광주 사찰은 명시되나 도심·금남로 위치는 설명에 없음
B72-2|en|P|Buddhist ceremonies가 불교 법회·의식에 대응
B73-2|en|A|자작나무 숲 탐방 및 자연 체험은 있으나 산림욕·치유를 명시하지 않음
B74-1|en|A|mailbox는 명시되나 1년 후 배달하는 느린 우체통 기능 없음
B76-2|en|A|Oreum trail은 있으나 용암 지형 설명 없음
B77-1|en|A|서울 조계종 총본산은 있으나 종로·도심 위치는 설명에 없음
B80-1|en|A|작가의 집과 artworks는 있으나 수묵화·서화 화풍 명시 없음
B82-1|en|A|공개 고분과 5~6세기는 설명하나 Silla 왕조명 명시 없음
B83-2|en|A|campground는 있으나 원시림 정보 없음
B84-2|en|A|댐 건설 전 유산 수집 및 전시를 설명하나 충주호 전망과 건축물 원형 이전 복원 명시 없음
B87-2|en|M|clocks and watches 및 스켈레톤 시계만 언급; 회중시계 명시 없음
B88-2|en|A|Outdoor Spa Zone의 Infinity Pool은 있으나 계곡 조망 없음
B89-1|en|P|Contemporary Gallery 및 Pinocchio & Marionette Gallery와 sea view 명시
B90-2|en|M|요리 체험 시설 일반 내용은 있으나 한옥 숙박·김치 만들기 모두 없음
B91-2|en|M|영실 등산로 일반 설명은 있으나 기암과 윗세오름 지명 없음
B93-1|en|P|medicinal herbs 및 massage techniques가 한방 재료·마사지에 대응
B93-2|en|P|different body types 및 one's body type에 따른 약재·마사지 방법 명시
B94-1|en|A|emerald-blue waters와 흰 모래는 있으나 얕은 수심 정보 없음
B97-1|en|P|watergates 및 nighttime view 명시
B98-2|en|A|exhibition hall은 있으나 대나무 정원 없음
B100-2|en|M|홍삼 스파 일반 설명뿐 홍삼 오일·전신 관리 프로그램 없음
B03-2|ja|A|椿林道는 있으나 30분·쉬운 등산·가족 이용 정보 없음
B11-1|ja|A|사찰이 聖出峰 중턱이라는 설명은 있으나 동악산 명칭 없음
B11-2|ja|M|사찰 소개뿐 계곡·폭포 정보 없음. 谷城 지명의 谷는 계곡을 뜻하는 근거가 아님
B12-1|ja|A|ハーブ植物園은 있으나 아로마 체험 없음
B12-2|ja|M|식물원 중심 설명; 체질 맞춤·스파 정보 없음
B14-2|ja|M|유람선 투어와 4층 구조는 설명하나 레스토랑·선상 식사 없음
B17-2|ja|P|長さ約1,060メートルの城 및 보존 상태가 성곽 설명에 대응
B19-2|ja|A|江華12鎮堡·격전지와 포대는 있으나 강화해협 수비 역할 명시 없음
B22-1|ja|P|アトラクション24種 및 놀이·오락 시설로 놀이공원 의미 확인
B22-2|ja|P|湖南地方最長のゲレンデを備えたソリ場로 눈썰매장 확인
B30-2|ja|M|탄산 약수 유래만 설명; 산불 피해 없음
B36-2|ja|P|静かに寄せる波がまるで湖のようである로 잔잔한 파도 확인
B40-1|ja|A|李穀·李穡 및 서원은 설명하나 고려 왕조명 없음
B42-2|ja|A|갤러리와 미술 거리는 소개하나 반월당역 없음
B44-2|ja|M|ヨーロッパのオープンカフェ와 이국적 분위기는 있으나 청담동 비유·고급 분위기는 없음
B45-1|ja|P|彼岸花が満開로 관광 포인트인 상사화류 꽃 관람 확인; 종 명칭 차이는 별개
B53-2|ja|A|現代式望遠鏡은 있으나 연화봉 명칭 없음
B58-2|ja|A|평창·삼척 두 물줄기의 합류는 있으나 송천·골지천 고유명 없음
B60-1|ja|A|大邱市街地を一望하는 전망대 소개만 있고 케이블카 운행 정보 없음; 제목에는 ケーブルカー 명시
B61-1|ja|A|水辺公園은 명시되나 박람회 개최 정보는 설명에 없음; 제목 엑스포에 해당 정보 있음
B64-2|ja|A|岩島와 조수 차이는 설명하나 지질 형성·침식 정보 없음
B65-2|ja|A|オートキャンプ場은 있으나 일몰 조망 정보 없음
B67-2|ja|P|海草類 및 マッド가 해조류·머드에 대응
B68-1|ja|P|西帰浦の海岸の沖合いに浮かぶ高さ20メートルの奇岩 및 陸と離れて海の中に寂しく立っている로 바다 돌기둥 확인
B68-2|ja|M|삼매봉과 산책 내용은 있으나 올레 7코스 없음
B69-1|ja|P|吊橋와 Y字形 명시; 吊り橋와 띄어쓰기·표기 차이
B73-2|ja|P|山林浴やヒーリング을 즐길 수 있다고 명시
B74-1|ja|A|ポスト는 설명하나 1년 뒤 배달하는 기능 없음
B76-2|ja|A|オルム散策路는 있으나 용암 지형 설명 없음
B82-1|ja|A|公開古墳 및 시대는 설명하나 신라 왕조 명칭 없음
B83-2|ja|A|キャンプ場은 명시되나 원시림 설명 없음
B84-2|ja|A|移転・復元과 忠州ダム은 명시되나 충주호 전망 설명 없음
B87-2|ja|M|시계와 부품 전시만 설명; 회중시계 명시 없음
B88-2|ja|A|森が見渡せる野外インフィニティプール은 있으나 계곡 전망 없음
B91-1|ja|P|登山の難度が低いため子供連れでも登れます로 초보 등산 의미 확인
B91-2|ja|A|ウィッセオルム은 명시되나 기암 정보 없음
B92-2|ja|P|1月に花が咲き始めて3月には満開로 겨울 동백꽃 확인
B93-1|ja|P|薬材とマッサージ方法 및 체질에 따른 치료 설명으로 한방 마사지 의미 확인
B100-2|ja|M|홍삼 전문 스파 일반 설명뿐 홍삼 오일·전신 관리 없음'''
status_map={'P':'PRESENT','A':'PARTIAL','M':'MISSING','C':'CONFLICT','R':'REVIEW'}
for line in FOLLOWUP.splitlines():
    iid,lang,code,note=line.split('|')
    assert (iid,lang) not in MANUAL
    MANUAL[iid,lang]=(status_map[code],note)

# Review broad walking-intent matches conservatively; existence of fortress
# walls is not itself an explicit walking/hiking recommendation.
for l in LANGS:
    MANUAL['B27-2',l]=('PARTIAL','산성·성곽은 설명하나 걷기/트레킹 코스나 체험은 원문 설명에 명시되지 않음')
    MANUAL['B17-2',l]=('PARTIAL','옛 성곽과 길이는 설명하나 성곽 걷기 체험은 원문 설명에 명시되지 않음')

SPOT_CHECK={
 'ko':('B57-2','B75-2','B67-1','B50-1','B38-2','B93-1','B28-1','B92-1'),
 'en':('B25-2','B71-2','B21-1','B74-2','B25-1','B24-1','B54-1','B37-1'),
 'ja':('B76-1','B58-1','B52-2','B27-2','B96-1','B89-2','B48-2','B16-2'),
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    qs=read(Q); docs=read(DOC); details=read(DETAIL)
    assert len(qs)==200 and len(docs)==300 and len(details)==300
    db={(r['place_id'],r['language']):r for r in docs};tb={(r['place_id'],r['language']):r for r in details}
    rules={}
    for line in RULES.splitlines():
        iid,concepts,*patterns=line.split('|')
        assert len(patterns)==3 and iid not in rules
        cs=concepts.split(';');ps={l:p.split(';') for l,p in zip(LANGS,patterns)}
        assert all(len(p)==len(cs) for p in ps.values()),(iid,cs,ps)
        rules[iid]=(cs,ps)
    assert set(rules)=={r['intent_id'] for r in qs}
    rows=[]
    for q in qs:
        iid=q['intent_id'];cs,ps=rules[iid]
        for l in LANGS:
            d=db[q['place_id'],l];t=tb[q['place_id'],l]
            assert d['description']==t['overview']
            hits=match_groups(d['description'],ps[l]);other=match_groups(' '.join([t['title'],t['addr1'],t['addr2']]),ps[l])
            n=sum(bool(h[0]) for h in hits)
            full=n==len(cs)
            status='PRESENT' if full else 'PARTIAL_CANDIDATE' if n else 'REVIEW'
            method='LEXICAL_EVIDENCE' if full else 'UNRESOLVED_RULE_MATCH'
            note='핵심 개념별 근거 단어 확인. 문맥/정확성은 별도 검토 가능' if full else '일부 또는 전체 근거 표현 미포착. 동의어/번역 표현 차이 가능; 실제 누락으로 확정하지 않음'
            if (iid,l) in MANUAL:
                status,note=MANUAL[iid,l];method='MANUAL_SOURCE_REVIEW'
            rows.append(dict(intent_id=iid,query_type='REPRESENTATIVE' if iid.endswith('-1') else 'DETAIL',place_id=q['place_id'],title_ko=q['title_ko'],language=l,query=q['query_'+l],core_features=' / '.join(cs),feature_count=len(cs),description_match_count=n,description_coverage=status,decision_method=method,matched_features=' / '.join(f'{c}: {h[0]}' for c,h in zip(cs,hits) if h[0]),unmatched_features=' / '.join(c for c,h in zip(cs,hits) if not h[0]),description_evidence=' | '.join(f'{c}: {h[1]}' for c,h in zip(cs,hits) if h[0]),title_address_evidence=' | '.join(f'{c}: {h[1]}' for c,h in zip(cs,other) if h[0]),note=note,manual_check_required='YES' if status in ('REVIEW','PARTIAL_CANDIDATE','CONFLICT') else 'NO',source_description=d['description']))
    assert len(rows)==600 and len({(r['intent_id'],r['language']) for r in rows})==600
    write(OUT/'feature_coverage_audit.csv',rows)
    write(OUT/'feature_gap_candidates.csv',[r for r in rows if r['description_coverage']!='PRESENT'])
    sample=[]
    for r in rows:
        if r['intent_id'] in SPOT_CHECK[r['language']]:
            sample.append(dict(**r,spot_check_result='CONTEXT_REVIEWED' if r['description_coverage']=='PRESENT' else 'RECLASSIFIED_AFTER_CONTEXT_CHECK',spot_check_note='AI가 문장 문맥과 근거를 재확인; 독립 사람 검증은 아님'))
    assert len(sample)==24
    write(OUT/'feature_coverage_positive_spotcheck.csv',sample)
    review=[r for r in rows if r['manual_check_required']=='YES']
    write(OUT/'feature_coverage_review_candidates.csv',review)
    summary=[]
    for kind in ('REPRESENTATIVE','DETAIL','ALL'):
        for l in LANGS:
            s=[r for r in rows if r['language']==l and (kind=='ALL' or r['query_type']==kind)]
            counts=Counter(r['description_coverage'] for r in s)
            summary.append(dict(query_type=kind,language=l,intents=len(s),present=counts['PRESENT'],partially_confirmed=counts['PARTIAL'],missing_confirmed=counts['MISSING'],conflict=counts['CONFLICT'],partial_candidate=counts['PARTIAL_CANDIDATE'],review=counts['REVIEW'],present_rate=round(counts['PRESENT']/len(s),6),manual_review_required=sum(r['manual_check_required']=='YES' for r in s),present_lexical_evidence=sum(r['description_coverage']=='PRESENT' and r['decision_method']=='LEXICAL_EVIDENCE' for r in s),source_text_followup=sum(r['decision_method']=='MANUAL_SOURCE_REVIEW' for r in s)))
    write(OUT/'feature_coverage_summary.csv',summary)
    print(Counter(r['description_coverage'] for r in rows));print('review',len(review));print(*summary,sep='\n')

if __name__=='__main__':main()
