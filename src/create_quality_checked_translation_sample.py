"""Create a source-segment-aligned AI translation sample for 20 documents.

This is a source-checked AI draft, not independently human-certified translation.
Each segment is translated once and reused in both experimental methods.
"""
import csv
import random
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tourapi_quality_controlled_sample'
CHANGED=ROOT/'results/tourapi_full_feature_enrichment/changed_documents.csv'
SOURCE=ROOT/'results/tourapi_full_translation/source_segments.csv'
SEED=20261120

TRANSLATIONS='''2569476|en|1|Bidulginangpokpo Falls, in Daehoesan-ri, Yeongbuk-myeon, is located in a basalt gorge carved by erosion, at the end of Bulmucheon Stream, which originates on Bulmusan Mountain.
2569476|en|2|One explanation of the name Bidulginang is that the surrounding terrain forms a recessed, pouch-like shape resembling a pigeon's nest.
2569476|en|3|Another tradition says the name comes from the rock pigeons that have long inhabited caves around the falls.
2569476|en|4|During the Korean War, the heavily wooded, secluded area was used as a shelter by local residents and as a resting place for soldiers.
2569476|en|5|It was designated a Natural Monument in 2012 and is now protected, offering visitors its striking scenery and natural beauty.
2569476|en|6|The falls offer opportunities to observe fluvial erosion landforms such as river caves, gorges, headward erosion and waterfalls, as well as geological structures including columnar and sheeting joints.
2569476|en|7|The units of lava that flowed along the Hantangang River can also be observed together, making the site academically valuable.
2569476|en|8|(Source: Hantangang Geopark.) Korean Wave travel information — television drama: the falls, where the character Jang Ho-woo and his group went on an outdoor field investigation in a drama, form a geological and ecological park in Pocheon.
2569476|en|9|The site features spectacular columnar joints and a waterfall with clear blue water, and is considered one of the eight scenic sights of the Hantangang River.
3023048|en|1|The Herb Healing Center opened in 2012 with the vision of a happy world through herbs.
3023048|en|2|It offers health and beauty products and experiences using herbs known for their distinctive fragrances, flavors and perceived therapeutic qualities.
3023048|en|3|The center grows and harvests herbs itself using its own pesticide-free cultivation technology, and offers wellness spa experiences described as health and immune care programs.
3023048|en|4|In the spa program, professional aromatherapists consult each visitor about their body type and then provide herbs, fragrances, colors, music and herbal tea tailored to their constitution, symptoms and biorhythm.
3023048|en|5|The center offers other experiences using herbs harvested in different seasons.
3023048|en|6|Seasonal programs include jasmine vitality care in spring, peppermint detox care in summer, hibiscus beauty care in autumn and rosemary restorative care in winter.
3023048|en|7|The center describes itself as a health and beauty facility using natural herbs and aromatherapy, with a high proportion of returning visitors.
2725257|en|1|Yeosu Minam Cruise is described as Korea's largest coastal cruise vessel, built using Korean shipbuilding technology.
2725257|en|2|The four-deck sightseeing ship carries up to 920 passengers, measures 66 meters long and 13 meters wide, and weighs 1,321 tons; it features the Mermaid Hall on deck one, a restaurant on deck two, a snack corner and VIP rooms on deck three, and an ocean observation deck on deck four.
2725257|en|3|Its programs include daytime seascape city tours, relaxing night-view cruises and onboard fireworks tours, allowing passengers to see major Yeosu attractions from the water.
2725257|en|4|The sunset and night-view tour departs before sunset on Saturdays and lasts about 70 to 80 minutes, taking in the sunset and night scenery of Yeosu's sea.
2725257|en|5|Departure times differ by month; visitors are advised to consult the reservation site for the scheduled sailing time and arrive at least 30 minutes before departure.
2725257|en|6|The vessel is equipped with safety and emergency medical equipment, and the crew has received lifesaving and emergency response training to help make passengers' journeys safe and comfortable.
1818945|en|1|Wondae-ri Birch Forest, also called the Whispering Birch Forest, is a natural ecotourism destination.
1818945|en|2|Created by planting 690,000 birch trees across 138 hectares between 1974 and 1995, the forest offers seven trail routes, a forest classroom, an observatory, an ecological pond, a children's Indian-style house, wooden bridges and stairs, where visitors can enjoy forest bathing and relaxation.
1818945|en|3|Visitors reach the birch grove by walking along the forest trail beginning at the birch forest information center.
1818945|en|4|There are two forest roads: the Wonjeong Forest Road, which takes about 80 minutes on foot, and the Wondae Forest Road, which takes about an hour.
1818945|en|5|Climbing along the trail leads to a dense forest of birch trees more than twenty years old.
1818945|en|6|The birches' shining white trunks create an exotic atmosphere, as if snow had fallen.
1818945|en|7|A range of trails with different lengths and difficulty levels is available inside the forest.
1818945|en|8|The most popular is Trail 1, the Birch Trail, which passes through a concentration of pure-white birch trees; it is about 0.9 kilometers long and takes around 50 minutes.
1818945|en|9|In winter the snow-covered woods and white birch trees form a beautiful scene popular for photography.
1818945|en|10|Families with young children can enjoy nature-themed activities at the children's forest experience area, including the forest classroom and the Indian-style house.
127715|en|1|Hyangiram Hermitage in Yeosu is a branch of the 19th district headquarters of the Jogye Order of Korean Buddhism, a popular destination with around one million annual visitors and a sanctuary for prayer to Avalokitesvara.
127715|en|2|The monk Wonhyo is said to have encountered Avalokitesvara and founded Wontongam there; in the reign of Joseon King Sukjong, the monk Inmuk named it Hyangiram.
127715|en|3|The temple features restored and newly built structures including Wontongbojeon Hall, Samseonggak, Gwaneumjeon, Yongwangjeon, a bell pavilion and a sea-facing Avalokitesvara statue. The main hall, temple office and bell pavilion destroyed by fire on December 20, 2009, were restored and celebrated with a consecration ceremony on May 6, 2012.
127715|en|4|It was named Hyangiram for the spectacular sunrise above the southern sea horizon, and is also called Yeongguam because the surrounding rocks resemble the backs of turtles.
126469|en|1|The Yeongsil hiking trail is 5.8 kilometers long and takes about two hours and thirty minutes; it is considered Hallasan's shortest and most beautiful route.
126469|en|2|Visitors can drive as far as the trail entrance at an elevation of 1,280 meters.
126469|en|3|The Yeongsil route to the Nambyeok junction below Baengnokdam is relatively easy to climb and recommended even for beginners.
126469|en|4|Although the section leading to Baengnokdam is restricted for safety reasons, there are numerous sights, including the Yeongsil rock formations, dense forest, the Five Hundred Arhats, Seonjakjiwat, meaning a gravel plain with stones, and Witseoreum, meaning the three volcanic cones above.
126469|en|5|Visitors who arrive by bus or taxi can descend by the Eorimok or Donnaeko trails to enjoy varied scenery.
126469|en|6|(Source: Jeju Culture and Tourism website.)
126214|en|1|Cheonmachong Tomb, within the Daereungwon Ancient Tombs in Gyeongju, is a pre-unification Silla tomb believed to have been built in the late fifth or early sixth century.
126214|en|2|Excavated in 1973, it takes the form of a stone-mounded wooden-chamber tomb typical of Silla.
126214|en|3|The mound measures 12.7 meters high and 50 meters in diameter. Inside is a layer of piled stream stones surrounding a wooden chamber measuring 6.5 by 4.2 meters and 2.1 meters high, with a wooden coffin at its center.
126214|en|4|Cheonmachong is the only tomb in the Daereungwon complex open to the public and was excavated as a trial investigation before the nearby Hwangnamdaechong Tomb.
126214|en|5|The technology of the time made excavating a mound as enormous as Hwangnamdaechong immediately difficult.
126214|en|6|A total of 11,526 artifacts were unearthed, including ornaments worn by the deceased such as a gold crown, gold cap, bird-wing-shaped crown ornaments, gold belt and gilt-bronze shoes.
126214|en|7|The gold crown is representative of Silla crowns: three tree-branch-shaped projections rise at the front and two antler-shaped projections at the back.
126214|en|8|The branches feature bud-shaped ornaments and curved jade beads symbolizing vitality; it is considered the largest and most elaborate gold crown excavated to date.
126214|en|9|The name Cheonmachong comes from the Heavenly Horse Painting discovered in the tomb.
126214|en|10|The painting of a flying horse was made on birch bark used as a saddle flap, a piece hanging from the saddle to protect the rider from mud. It was the first painting excavated from a Korean ancient tomb and is designated a National Treasure as a representative Silla painting of great value.
126214|en|11|The artifacts excavated from Cheonmachong are now exhibited at Gyeongju National Museum.
2366916|en|1|Apsan Park, which features a cable car, is one of Daegu's representative urban nature parks.
2366916|en|2|It is close enough for a short stroll away from the fatigue of everyday life, making it a popular urban retreat.
2366916|en|3|The cable car is located on the foothill path in Keungol Valley near the Nakdong River Victory Memorial Hall in the park.
2366916|en|4|Riding the cable car to the summit offers a breathtaking panoramic view of Daegu.
2366916|en|5|Visitors can breathe the fresh mountain air amid the dense forest, unwind and regain their vitality.
126088|en|1|Oryukdo is a group of six rocky islets rising out of the rough sea off Yongho-dong, named Bangpaeseom, Solseom, Suriseom, Songgotseom, Gulseom and Deungdaeseom in order from the mainland.
126088|en|2|Until around 120,000 years ago, the islets are believed to have formed a small peninsula connected to the mainland, before prolonged wave erosion separated them from it.
126088|en|3|This is also supported by the identical geological composition of the mainland's Seungdumal and the Bangpaeseom and Solseom islets.
126088|en|4|The name Oryukdo is recorded in the mountains and streams section of the Dongnae Gazetteer compiled in 1740: “Oryukdo lies to the east of Jeoryeongdo.
126088|en|5|Its peaks are unusually shaped and stand in a row in the sea; seen from the east there are six peaks, and from the west there are five, hence the name.”
126088|en|6|Accordingly, the name is said to derive from the different number of peaks visible depending on the observer's position and direction.
126088|en|7|Bangpaeseom, the closest to the mainland, is named for shielding the coast from strong winds and waves; Solseom has naturally growing pines; Suriseom attracted eagles hunting seagulls; Songgotseom is small and pointed; Gulseom is the largest and has a cave with enough water dripping from its ceiling for one person to drink; and flat Deungdaeseom, furthest from shore, was once called Batseom and renamed after a lighthouse was built. It is the only inhabited islet.
126088|en|8|Because the currents near Oryukdo are very swift, the waters were once hazardous for boats; sailors are said to have prayed for safe voyages by casting rice into the sea as an offering to the sea deity and performing ceremonies honoring the dragon spirit.
126088|en|9|Oryukdo is a gateway to Korea and a symbol of Busan Port, through which various incoming and outgoing vessels pass.
126088|en|10|(Source: Busan Nam-gu Office Culture and Tourism.)
2569503|en|1|The Time & Blade Museum opened in 2008 to let visitors see and experience the evolution of civilization and culture through historic timepieces and swords.
2569503|en|2|It exhibits pocket watches from the eighteenth and nineteenth centuries and around 1,000 swords from Macedonia, Rome and the Middle East.
2569503|en|3|The first floor contains timepieces dating from the eighteenth to twentieth centuries, as well as clock parts, manufacturing tools, equipment and other materials related to horology.
2569503|en|4|Visitors can see sundials used to measure time astronomically, astronomical clocks showing the motion of the planets, mechanical pocket watches made by Swiss artisans, lavishly decorated French wall and mantel clocks, and technically precise German timepieces.
2569503|en|5|The second floor displays numerous blades, objects that have existed as tools of survival throughout human history and represent the civilizations and cultures of many periods and countries.
2569503|en|6|From Stone Age blades to the large swords of the Iron Age, blades appeared as humans began using tools and offer a way to explore the origins of civilization.
2569503|en|7|The history of blades also reflects the history of power and conflict.
2569503|en|8|The collection includes swords from Persia under Darius, Alexander's Macedonia, the Roman Empire, medieval Crusaders and Spain, as well as examples from the era of Genghis Khan and numerous Asian and Middle Eastern countries including Korea, Japan, India, Nepal, Uzbekistan, Syria, Yemen and Oman.
127185|ja|1|高麗時代に江華海峡を守っていた外城の要衝です。
127185|ja|2|丙子胡乱の後、江華島を守るために築かれた内城、外城、墩台、鎮堡などの12鎮堡の一つです。
127185|ja|3|顕宗7年（1666年）、国防強化のため水営（海軍の駐屯地）に属していた徳津鎮が徳浦に移されました。粛宗5年（1679年）には龍頭墩台と徳津墩台を傘下に置き、徳津砲台と南障砲台を管轄し、江華海峡で最も強力な砲台として知られ、江華12鎮堡の中でも重要な場所を守っていました。
127185|ja|4|1866年の丙寅洋擾では楊憲洙の軍が徳津鎮を経て鼎足山城に入り、フランス軍を破りました。1871年の辛未洋擾ではアメリカ艦隊との激しい砲撃戦が繰り広げられました。
127185|ja|5|しかし、草芝鎮に上陸したアメリカ軍によって占領されました。
127185|ja|6|この時、兵士を守るために築かれた低い塀はすべて破壊されました。
127185|ja|7|1976年には城郭、墩台、南障砲台が修復され、正面3間、側面2間の門楼も再建され、当時の大砲が復元・設置されました。
127185|ja|8|（出典：国家遺産庁）
126004|ja|1|忠清北道堤川の清風地域は美しい自然景観と豊富な文化遺産を有する地域でしたが、忠州ダムの建設により村と遺産が水没する危機に直面しました。
126004|ja|2|そこで1980年代初めから水没予定地域にあった文化遺産を元の姿のまま移築・復元し、現在の清風文化遺産団地が造成されました。
126004|ja|3|団地では郷校、官衙、民家、石造物など約40点の文化遺産と生活用品が展示され、清風寒碧楼をはじめとする主要建築物や遺跡を見学できます。
126004|ja|4|特に高麗時代の楼閣である寒碧楼からは、忠州湖の素晴らしい風景を眺めることができます。
126004|ja|5|清風文化遺産団地は歴史的価値と自然景観が調和した空間で、かつての清風の生活や伝統文化を体験できる場所です。
3023048|ja|1|ハーブヒーリングセンターは「ハーブで幸せな世界を」というビジョンのもと、2012年に開館しました。
3023048|ja|2|独特の香りや味、癒しの作用があるとされるハーブを利用した健康・美容関連の商品や体験を提供しています。
3023048|ja|3|独自に開発した無農薬ハーブ栽培技術を用いてハーブを自ら栽培・収穫し、健康・免疫管理をテーマとするスパ体験プログラムを運営しています。
3023048|ja|4|スパでは専門のアロマセラピストが一人ひとりの体質を相談し、体質、症状、バイオリズムに合ったハーブ、香り、色、音楽、ハーブティーを提供しています。
3023048|ja|5|このほか、季節ごとに収穫したハーブを用いたさまざまな体験も提供しています。
3023048|ja|6|春はジャスミンの活力プログラム、夏はペパーミントのデトックスプログラム、秋はハイビスカスの美容プログラム、冬はローズマリーの滋養プログラムなどがあります。
3023048|ja|7|天然ハーブとアロマを利用した健康・美容施設で、リピーターが多いと紹介されています。
126469|ja|1|霊室登山道は全長5.8キロメートル、所要時間2時間30分のコースで、漢拏山の登山コースの中でも最も短く美しい区間とされています。
126469|ja|2|登山道の入口がある標高1,280メートル地点まで車で行くことができます。
126469|ja|3|白鹿潭の下にある南壁分岐点まで続く霊室探訪路は比較的登りやすく、初心者にもおすすめのコースです。
126469|ja|4|白鹿潭までの区間は安全上の理由で通行が規制されていますが、霊室奇岩、豊かな森、五百羅漢、「石のある砂利の平地」を意味するソンジャクチワッ、「上にある三つのオルム」を意味するウィッセオルムなど、多くの見どころがあります。
126469|ja|5|バスやタクシーで来た場合は、オリモクやトンネコ方面へ下山し、さまざまな景色を楽しむこともできます。
126469|ja|6|（出典：済州文化観光ホームページ）
126111|ja|1|※2025年3月の山火事で達基湧水の薬水場が全焼したとの案内があります。訪問前にご留意ください。
126111|ja|2|達基湧水は今から約130年前、水路工事の途中で発見されました。
126111|ja|3|口の中ではじける炭酸のほかさまざまな成分が含まれ、昔から胃腸病、婦人病や眼病などに効果があると伝えられています。
126111|ja|4|旧地名が青松郡府内面達基洞であったことから、達基薬水と呼ばれるようになりました。
126111|ja|5|上湯、中湯、元湯など5～6か所から薬水が湧き、原文では周辺に約30軒の飲食店街が形成され営業していると紹介されています。
126111|ja|6|薬水の神秘性に感謝するため、毎年旧暦3月末には「達基薬水霊泉祭」が行われます。
127811|ja|1|青玉山は標高1,277メートルで、太白山の望鏡台山頂から南東に位置しています。
127811|ja|2|太白山を中心に標高1,000メートルを超える山が9つもあるほど山勢の険しい山間部にあります。
127811|ja|3|白頭大幹のカッテベギ峰から南東に分かれる尾根に位置し、召川面のヌッチェから尾根沿いに望鏡台へ続く登山道があります。
127811|ja|4|周辺約1億53万平方メートルは青玉山自然休養林に指定され、韓国最大の休養林とされます。水遊び場、運動施設、山小屋、キャンプ場、キャンプファイヤー場などが整備されています。
127811|ja|5|山腹には惟政が修行したと伝わる弘済寺があり、山沿いを流れる古善渓谷には100里にも及ぶ深い原生林が広がっています。
127811|ja|6|太白山文殊峰と青玉山の間から始まりチョロク岩峰に至る全長12キロメートルの白川渓谷は、洛東江の上流であり、冷水性の淡水魚ヨルモゴの世界最南端の生息地として知られています。
127811|ja|7|周辺には5万の霊を慰める塔のある現仏寺、午前薬水、梧谷薬水場、仏影寺、多徳薬水観光地、斗内薬水場、清涼山道立公園、四未亭などの観光地もあります。
126802|ja|1|鳳山文化通りは、大邱都市鉄道2号線・半月堂駅9番出口前の「メディアスカイ（鳳山の空）」のオブジェから鳳山五差路まで、全長600メートルにわたって整えられた文化芸術の通りです。
126802|ja|2|画廊、古美術店、古書店、画材店などの美術関連の店が集まり、大邱の仁寺洞にたとえられています。
126802|ja|3|文化芸術の通りとして整備が始まった1980年代半ばには作品や画材の販売、表具の制作が中心でしたが、現在は多様な作品が展示・販売されています。
126802|ja|4|多くのギャラリーが開設されたことで大邱を代表する地元画家の作品に接する機会が増え、ほかにも多様な美術作品を鑑賞できます。
126802|ja|5|通りに設置されたオブジェは撮影スポットとして活用され、2004年に建てられた鳳山文化会館ではさまざまな展覧会や公演が行われています。
126802|ja|6|毎年4月には鳳山工芸デザイン展、10月には鳳山美術祭が開催されます。
126802|ja|7|周辺には中区の路地ツアー、東城路、キム・グァンソク再び描く道などの見どころがあります。
126802|ja|8|文化通りの入口と終点にはインスタレーションや大きな人物のオブジェがあり、遠くからでも見つけやすい場所です。
126470|ja|1|済州オルレ7コースの出発地点であるウェドルゲは、海から高さ20メートルもそびえる石柱状の奇岩です。
126470|ja|2|海の上に一本だけ立っていることから「ウェドルゲ」と名付けられました。
126470|ja|3|「将軍石」や「ハルマン岩」とも呼ばれ、崔瑩将軍が元と戦った際、この岩を将軍のように見せかけて敵に打撃を与えたという伝承があります。
126470|ja|4|「ハルマン岩」の名の由来については、次の伝説を参照してください。
126470|ja|5|目の前の岩だけでなく、海に出た人々を待っていた済州の人々の暮らしも垣間見ることができます。
126470|ja|6|（出典：済州文化観光ホームページ）
126088|ja|1|五六島は龍湖洞の沖で荒波の中にそびえる六つの岩島で、陸地に近い順にパンペソム、ソルソム、スリソム、ソンゴッソム、クルソム、灯台ソムと呼ばれます。
126088|ja|2|約12万年前までは陸続きの小さな半島でしたが、長い年月にわたる激しい波の侵食によって陸から切り離されて形成されたものと考えられています。
126088|ja|3|陸地のスンドゥマルとパンペソム、ソルソムの地質構成が同一であることからも、これがうかがえます。
126088|ja|4|1740年に編纂された『東萊府誌』の山川条には、五六島について「五六島は絶影島の東にある。
126088|ja|5|奇妙な形の峰が海中に並び、東から見れば六つ、西から見れば五つの峰となるため、この名が付いた」と記されています。
126088|ja|6|このように、見る人の位置や方向によって見え方が異なることが名称の由来とされています。
126088|ja|7|陸地に最も近いパンペソムは強い風と波を防ぐ盾のような島、ソルソムは頂上に松が自生する島、スリソムはカモメを狙うワシが集まった島、ソンゴッソムは小さく尖った島、クルソムは天井から飲み水になる水が滴る大きな洞窟のある島です。陸から最も遠い平坦な灯台ソムは昔は畑の島とも呼ばれましたが、灯台が建てられてから灯台ソムと呼ばれ、五六島で唯一人が暮らす島です。
126088|ja|8|五六島付近は潮流が速く船の航行には危険な場所だったため、昔の船乗りたちは航海の無事を祈り、米を海の神に供えたり、龍神祭を行ったりしたと伝えられています。
126088|ja|9|五六島は韓国の玄関口であり、釜山港を出入りする船舶が通るため、釜山港の象徴でもあります。
126088|ja|10|（出典：釜山南区庁文化観光）
2569503|ja|1|タイムアンドブレイド博物館は、長い歴史を持つ時計と刀剣を通して文明と文化の流れを直接見て体験できる博物館を目指し、2008年に開館しました。
2569503|ja|2|18～19世紀の懐中時計と、マケドニア、ローマ、中東などの刀剣約1,000点が展示されています。
2569503|ja|3|1階には18世紀から20世紀までの多様な時計や、時計の部品、製作工具、設備などの資料が多数展示されています。
2569503|ja|4|日時計、太陽系の惑星の動きを示す天文時計、スイスの職人が作った機械式懐中時計、華麗に装飾されたフランスの掛け時計や置き時計、精巧なドイツの時計など、さまざまな時計を見ることができます。
2569503|ja|5|2階には多くの刀剣が展示されており、刃物は人類の誕生以来、生存のための道具であると同時に、それぞれの時代や国の文明と文化を表すものでもあります。
2569503|ja|6|石器時代の石の刃物から鉄器時代の大剣まで、刃物は人類が道具を使い始めるとともに登場し、文明の起源をたどる手がかりになります。
2569503|ja|7|刀剣の歴史は、権力と争いの歴史を物語るものでもあります。
2569503|ja|8|ペルシャのダリウス時代、アレクサンドロス大王のマケドニア、ローマ帝国、中世の十字軍、スペインの刀剣のほか、チンギス・ハンの時代や韓国、日本、インド、ネパール、ウズベキスタン、シリア、イエメン、オマーンなどアジア・中東諸国の多様な刀剣が展示されています。'''

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pool=read(CHANGED);rng=random.Random(SEED);selected=[]
    for lang in ('en','ja'):
        selected+=rng.sample(sorted([r for r in pool if r['language']==lang],key=lambda r:r['place_id']),10)
    assert len(selected)==20
    source=defaultdict(list)
    for r in read(SOURCE):source[r['place_id']].append(r)
    trans={}
    for line in TRANSLATIONS.splitlines():
        pid,lang,idx,text=line.split('|',3);key=(pid,lang,int(idx))
        assert key not in trans and text.strip();trans[key]=text
    expected={(r['place_id'],r['language'],int(x['segment_index'])) for r in selected for x in source[r['place_id']]}
    assert set(trans)==expected,(len(trans),expected-set(trans),set(trans)-expected)
    rows=[]
    for d in selected:
        for s in source[d['place_id']]:
            key=(d['place_id'],d['language'],int(s['segment_index']))
            rows.append(dict(place_id=key[0],language=key[1],segment_index=key[2],source_text=s['source_text'],translation=trans[key],review_status='AI_SOURCE_CHECKED_NOT_HUMAN_CERTIFIED'))
    assert len(rows)==147
    write(OUT/'sample_documents.csv',[dict(place_id=d['place_id'],language=d['language'],title=d['title'],intent_ids=d['intent_ids'],selection_seed=SEED,selection_method='uniform random within target language among 68 gap documents') for d in selected])
    write(OUT/'source_checked_translation_segments.csv',rows)
    print('translated',len(rows),'segments',Counter(d['language'] for d in selected))

if __name__=='__main__':main()
