"""Expand source-grounded description supplementation to 73 eligible gaps.

The 68 changed foreign-language documents are a fixed experimental copy.
All original data and previous experiments are preserved. The evaluation
queries were used to select the gaps, so the result is exploratory.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from run_bge_m3 import DATA, MODEL, ROOT, encode, evaluate, metrics_by_group, load_baseline_model
from run_natural_query_evaluation import summarize

STANDARD=ROOT/'data/processed/tourapi_standardized.csv'
AUDIT=ROOT/'results/tourapi_feature_coverage/feature_coverage_audit.csv'
ASSOC=ROOT/'results/tourapi_coverage_retrieval_analysis/coverage_rank_joined.csv'
BASE=ROOT/'results/bge_m3_dense_full'
NAT=ROOT/'results/tourapi_natural_query_eval_600'
OUT=ROOT/'results/tourapi_full_feature_enrichment'
OUTPUT_DATA=ROOT/'data/processed/tourapi_standardized_enriched_68_experiment.csv'
LANGS=('ko','en','ja')

# intent | target language | concise translation of information in the Korean
# source (not a direct copy of the test search query). No new facts are inferred.
SUPPLEMENT_LINES='''B01-1|en|The head spa also uses massage to promote scalp circulation and relax the muscles.
B02-2|ja|建物は灰色のレンガをオランダ式に積んだ二階建てで、礼拝のための説教壇を中心に左右対称の構造を備えています。
B03-2|en|The summit is 107 meters above sea level, and the relatively easy walk takes about 30 minutes even at a leisurely pace, making it suitable for a range of ages.
B03-2|ja|標高107メートルで、ゆっくり歩いても約30分で到着する比較的負担の少ない道のため、子供から高齢者まで楽しめます。
B07-2|en|The cave takes its name from the Go family, who are said to have taken shelter here during the Japanese invasions of Korea in the late sixteenth century.
B07-2|ja|高氏窟という名称は、壬辰倭乱（文禄・慶長の役）の時に高氏一族がこの洞窟に身を隠したという伝承に由来します。
B09-2|en|The surrounding rocks resemble the backs of turtles, which is why the hermitage is also known as Yeongguam.
B10-2|ja|現在は1970～80年代の街並みが再現され、線路沿いの古い商店で昔懐かしいお菓子などを楽しめます。当時の制服を着て記念撮影する体験も人気です。
B11-1|en|The temple sits halfway up Seongchulbong Peak on a ridge of Dongaksan Mountain.
B11-1|ja|寺がある聖出峰は、谷城の動楽山から伸びる山並みの一部です。
B11-2|en|The nearby Dorim Valley has flowing water throughout the year, with old pine trees and waterfalls creating a scenic landscape.
B11-2|ja|近くには動楽山の南側から流れる道林渓谷があり、年中絶えない水流と老松、滝が美しい景観を作ります。
B12-1|en|The Korean description also introduces wellness spa experiences using locally cultivated herbs and professional aromatherapy.
B12-1|ja|韓国語の案内では、自家栽培したハーブを使った専門的なアロマセラピーのスパ体験も紹介されています。
B12-2|en|A therapist discusses each visitor's constitution and then offers herbs, scents, colors, music and herbal tea suited to the visitor as part of a personalized spa program.
B12-2|ja|スパプログラムでは専門のアロマセラピストが体質を相談し、一人ひとりに合ったハーブ、香り、色、音楽、ハーブティーを提供します。
B14-2|en|The ship has a restaurant on the second deck, a snack area on the third deck, and an ocean-view observation deck on the fourth.
B14-2|ja|船内には2階のレストラン、3階のスナックコーナー、4階の海を見渡す展望デッキがあります。
B18-2|en|Besides enjoying the scenery and taking photographs, visitors can experience picking fresh green tea leaves.
B19-2|ja|高麗時代から江華海峡を守る外城の要衝であり、後に江華島防衛の重要な砲台を管轄しました。
B26-2|en|The walking paths provide views of the Hallyeosudo sea and nearby islands including Hansando.
B30-2|en|A notice in the Korean source reports that the Dalgi mineral spring site was destroyed by the wildfire in March 2025; the notice does not establish its current reopening status.
B30-2|ja|韓国語の案内には、2025年3月の山火事で達基湧水の薬水場が焼失したとの注意書きがあります。現在の再開状況はその記載だけでは確認できません。
B32-2|en|Gucheondong Valley, with its numerous waterfalls, pools and rock formations, is a particularly renowned part of Deogyusan.
B40-1|en|Lee Gok and Lee Saek were scholars of the late Goryeo dynasty, and the academy pays tribute to them.
B40-1|ja|この書院は高麗時代後期の学者、李穀と李穡の学問と徳をたたえるためのものです。
B42-2|en|The Korean guide places one end of the art street by the Media Sky sculpture near Exit 9 of Banwoldang Station on Daegu Metro Line 2.
B42-2|ja|韓国語の案内によると、大邱地下鉄2号線・半月堂駅9番出口付近のメディアスカイのオブジェから続く美術の通りです。
B43-1|en|Visitors can walk through narrow alleys lined with traditional hanok houses and their sweeping roof eaves.
B44-2|en|The Korean guide likens the upscale atmosphere of the cafes to Cheongdam-dong in Seoul.
B44-2|ja|韓国語の案内では、高級感のあるカフェ街の雰囲気はソウルの清潭洞にも例えられています。
B46-1|en|The falls are located in a gorge formed by the erosion of basalt.
B46-2|en|One explanation for the name is that the surrounding terrain forms a recessed, pouch-like shape resembling a pigeon's nest.
B48-2|en|A distinctive feature of the cafe is its second-floor attic, which evokes a journey back in time.
B53-2|en|The observatory is located near Yeonhwabong Peak, which visitors reach on foot at an elevation of approximately 1,400 meters.
B53-2|ja|天文台へは標高約1,400メートルの蓮花峰付近まで歩いて向かいます。
B57-2|en|The Catholic church stands on a hill in Gongse-ri, Inju-myeon, Asan.
B58-2|en|The two waterways that meet here are the Songcheon and Goljicheon streams.
B58-2|ja|ここで合流する二つの河川は松川（ソンチョン）と骨只川（コルジチョン）です。
B60-1|en|A cable car in Apsan Park takes visitors up toward the summit, where they can enjoy the city panorama.
B60-1|ja|アプ山公園にはケーブルカーがあり、山の上から大邱市街地の眺めを楽しむことができます。
B61-1|ja|この水辺公園は、かつて麗水で開催された博覧会の会場とその施設を活用した観光スポットです。
B62-2|en|Landscape lighting along the stream allows visitors to enjoy the cherry blossoms at night as well.
B64-2|en|The rock islets are believed to have formed as waves eroded what was once a small peninsula connected to the mainland.
B64-2|ja|かつて陸続きだった小さな半島が長い年月の波の侵食によって陸から離れ、岩島になったと考えられています。
B65-2|ja|海を眺めながらキャンプができ、日が沈む時間帯には特に美しい景観が楽しめます。
B68-2|en|The rock is also the starting point of Jeju Olle Trail Course 7.
B68-2|ja|この海岸の奇岩は、済州オルレ7コースの出発地点でもあります。
B72-1|en|The temple is located in the busy Geumnam-ro area of central Gwangju.
B73-2|en|Its birch groves and various walking trails offer opportunities for forest bathing and relaxation.
B74-1|en|The Korean description notes that the observatory operates a slow mailbox intended to deliver letters one year later.
B74-1|ja|韓国語の案内では、投函した手紙が1年後に届く「遅い郵便ポスト」が運営されていると紹介されています。
B76-2|en|Gotjawal refers to a distinctive Jeju forest formed where vegetation and rock intermingle on volcanic lava terrain.
B76-2|ja|コッチャワルとは、火山が噴出した溶岩地形の上に樹木と岩が混在してできた済州独特の森です。
B77-1|en|The temple is situated in the heart of Jongno, central Seoul, offering a peaceful space in the city.
B80-1|en|Heo Ryeon was a renowned late-Joseon artist associated with Southern School ink landscape painting.
B82-1|en|Cheonmachong is believed to be a tomb of the pre-unification Silla kingdom, built in the late fifth or early sixth century.
B82-1|ja|天馬塚は5世紀末から6世紀初めに造られた統一以前の新羅の古墳と考えられています。
B83-2|en|The nearby Goseon Valley is a deep valley characterized by its untouched forest.
B83-2|ja|山のそばを流れる古善渓谷は、深い原生林の景観を残しています。
B84-2|en|Historic structures were relocated and restored in their original forms, and the Hanbyeongnu Pavilion commands a view of Chungju Lake.
B84-2|ja|高麗時代の楼閣である清風寒碧楼からは、忠州湖の美しい景色を眺めることができます。
B87-2|en|The collections include pocket watches from the eighteenth and nineteenth centuries.
B87-2|ja|18世紀から19世紀にかけての懐中時計も展示されています。
B88-2|en|The Valley Spa area is set amid trees and a valley, complementing the outdoor infinity pool and forest setting.
B88-2|ja|屋外のインフィニティプールに加え、木々と渓谷に囲まれたバレースパゾーンで自然を身近に感じられます。
B90-2|en|The center offers accommodation in traditional hanok houses and hands-on cooking experiences in which visitors can prepare Korean foods such as kimchi.
B91-2|en|Highlights on the route include the striking Yeongsil rock formations and the Witseoreum volcanic cones.
B91-2|ja|登山路には霊室奇岩などの独特な岩の景観も広がっています。
B94-1|en|The clear, shallow water makes the beach suitable for families.
B98-2|en|The outdoor green spaces also include a bamboo grove and a kitchen garden.
B100-2|en|The Korean guide describes treatments using red ginseng oil, including facial and body care and a full-body premium treatment program.
B100-2|ja|韓国語の案内では、紅参オイルを使ったフェイシャル・ボディケアや全身を対象とするプレミアムトリートメントが紹介されています。'''

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    docs=read(DATA);standard=read(STANDARD);queries=read(NAT/'queries.csv');audit=read(AUDIT);assoc=read(ASSOC)
    assert len(docs)==len(standard)==300 and len(queries)==600 and len(audit)==600
    meta=json.loads((BASE/'run_metadata.json').read_text())
    assert meta['corpus_sha256']==hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert meta['model']==MODEL and meta['max_seq_length']==1024
    assert all((d['place_id'],d['language'],d['title'],d['overview'])==(s['place_id'],s['language'],s['title'],s['description']) for d,s in zip(docs,standard))
    prior_idx=read(BASE/'embedding_index.csv')
    assert len(prior_idx)==len(docs) and all((a['place_id'],a['language'],a['source_contentid'])==(b['place_id'],b['language'],b['source_contentid']) for a,b in zip(docs,prior_idx))
    baseline_vec=np.load(BASE/'document_embeddings.npy');qvec=np.load(NAT/'query_embeddings.npy')
    assert baseline_vec.shape==(300,1024) and qvec.shape==(600,1024)
    gaps={(r['intent_id'],r['target_language']):r for r in assoc if r['eligible_primary']=='YES' and r['coverage_group']=='GAP'}
    source={(r['intent_id'],r['language']):r for r in audit}
    supplements={}
    for line in SUPPLEMENT_LINES.splitlines():
        intent,lang,text=line.split('|');key=(intent,lang)
        assert key not in supplements;supplements[key]=text
    assert set(supplements)==set(gaps),(set(gaps)-set(supplements),set(supplements)-set(gaps))
    by_doc=defaultdict(list);changes=[]
    for (intent,lang),text in supplements.items():
        gap=gaps[intent,lang];ko=source[intent,'ko'];target=source[intent,lang]
        assert ko['description_coverage']=='PRESENT' and target['description_coverage'] in ('PARTIAL','MISSING')
        assert ko['place_id']==target['place_id']==gap['place_id'] and ko['description_evidence']
        by_doc[gap['place_id'],lang].append((intent,text))
        changes.append(dict(intent_id=intent,place_id=gap['place_id'],title_ko=gap['correct_title_ko'],language=lang,query_type=gap['query_type'],original_coverage=target['description_coverage'],core_features=target['core_features'],original_gap_note=target['note'],korean_source_evidence=ko['description_evidence'],added_text=text,verification='AI translation of supplied Korean description; independent factual verification not performed'))
    assert len(changes)==73 and len(by_doc)==68
    write(OUT/'description_changes.csv',changes)
    updated=[];changed_docs=[];indices=[];texts=[]
    for i,(d,s) in enumerate(zip(docs,standard)):
        key=(d['place_id'],d['language']);row=dict(s)
        if key in by_doc:
            snippets=by_doc[key];addition=' '.join(t for _,t in snippets)
            row['description']+=' '+addition
            indices.append(i);texts.append(d['search_text']+' '+addition)
            changed_docs.append(dict(place_id=key[0],language=key[1],title=d['title'],intent_ids='|'.join(x for x,_ in snippets),original_chars=len(s['description']),new_chars=len(row['description']),added_text=addition))
        updated.append(row)
    assert len(indices)==68 and len(updated)==300 and sum(a!=b for a,b in zip(standard,updated))==68
    assert all(a[k]==b[k] for a,b in zip(standard,updated) for k in ('source','place_id','language','title'))
    assert all(b['description'].startswith(a['description']) for a,b in zip(standard,updated))
    write(OUTPUT_DATA,updated);write(OUT/'changed_documents.csv',changed_docs)
    model=load_baseline_model(meta)
    tokens=[len(model.tokenizer(t,add_special_tokens=True,truncation=False)['input_ids']) for t in texts]
    assert max(tokens)<=1024,max(tokens)
    after_vec=baseline_vec.copy();after_vec[indices]=encode(model,texts,4,'enriched foreign documents',np)
    unchanged_indices=[i for i in range(len(docs)) if i not in indices]
    assert np.array_equal(after_vec[unchanged_indices],baseline_vec[unchanged_indices])
    np.save(OUT/'document_embeddings_enriched.npy',after_vec)
    before=evaluate(docs,queries,baseline_vec,qvec,np);after=evaluate(docs,queries,after_vec,qvec,np)
    assert len(before)==len(after)==2400
    original=read(NAT/'query_results_original.csv')
    assert len(original)==len(before) and all(str(a['rank'])==b['rank'] and a['query_id']==b['query_id'] and a['target_language']==b['target_language'] for a,b in zip(before,original))
    write(OUT/'query_results_enriched.csv',after)
    place_titles={d['place_id']:d['title'] for d in docs if d['language']=='ko'}
    comp=[];modified_places={p for p,l in by_doc};modified_intents={i for i,l in gaps}
    for a,b in zip(before,after):
        assert (a['query_id'],a['target_language'])==(b['query_id'],b['target_language'])
        iid=a['query_id'].rsplit('-',1)[0];target=a['target_language'];gain=a['rank']-b['rank']
        comp.append(dict(intent_id=iid,query_id=a['query_id'],query_type=a['query_type'],query_language=a['query_language'],target_language=target,place_id=a['place_id'],correct_title_ko=place_titles[a['place_id']],query=a['query'],target_document_changed='YES' if (a['place_id'],target) in by_doc else 'POSSIBLE' if target=='ALL_PLACES' and a['place_id'] in modified_places else 'NO',gap_target='YES' if (iid,target) in gaps else 'YES_MIXED' if target=='ALL_PLACES' and iid in modified_intents else 'NO',rank_before=a['rank'],rank_after=b['rank'],rank_gain=gain,change='IMPROVED' if gain>0 else 'WORSE' if gain<0 else 'UNCHANGED',top1_before=a['top1_title'],top1_after=b['top1_title'],correct_score_before=a['correct_score'],correct_score_after=b['correct_score']))
    write(OUT/'rank_comparison.csv',comp)
    targeted=[r for r in comp if r['gap_target']=='YES'];assert len(targeted)==219
    write(OUT/'gap_target_rank_comparison.csv',targeted)
    groups=[]
    for kind in ('ALL','REPRESENTATIVE','DETAIL'):
        cohort=[r for r in comp if kind=='ALL' or r['query_type']==kind]
        for scope,pred in (
            ('MIXED',lambda r:r['target_language']=='ALL_PLACES'),
            ('SINGLE_LANGUAGE',lambda r:r['target_language'] in LANGS),
            ('SAME_LANGUAGE',lambda r:r['target_language']==r['query_language']),
            ('CROSS_LANGUAGE',lambda r:r['target_language'] in LANGS and r['target_language']!=r['query_language']),
            ('GAP_TARGETS',lambda r:r['gap_target']=='YES'),
            ('GAP_TARGETS_CROSS',lambda r:r['gap_target']=='YES' and r['target_language']!=r['query_language']),
            ('OTHER_SINGLE_TARGETS',lambda r:r['target_language'] in LANGS and r['gap_target']=='NO'),
            ('GAP_INTENT_MIXED',lambda r:r['gap_target']=='YES_MIXED'),
        ):
            subset=[r for r in cohort if pred(r)]
            if subset:groups.append(summarize(subset,kind+'_'+scope))
    write(OUT/'summary.csv',groups)
    m=[]
    for a,b in zip(metrics_by_group(before),metrics_by_group(after)):
        k={t:a[t] for t in ('query_type','query_language','target_language','queries')}
        assert all(a[x]==b[x] for x in k)
        for t in ('recall_at_1','recall_at_5','recall_at_10','mrr'):
            k.update({t+'_before':a[t],t+'_after':b[t],t+'_delta':round(b[t]-a[t],6)})
        m.append(k)
    write(OUT/'language_metrics_comparison.csv',m)
    (OUT/'metadata.json').write_text(json.dumps(dict(eligible_intent_language_gaps=73,changed_documents=68,modified_places=len(modified_places),unchanged_documents=232,baseline_corpus_sha256=meta['corpus_sha256'],query_embeddings_reused=True,max_updated_doc_tokens=max(tokens),limitations=['Supplements use sourced Korean facts and are AI-translated; do not treat as verified current travel guidance.','Same evaluation queries informed the audit; not blinded or independently evaluated.','Raw descriptions preserved in original file; conflicts and two reference-partial walking intents excluded.','Single-target metrics do not represent exhaustive relevance.']),ensure_ascii=False,indent=2),encoding='utf-8')
    for g in groups:
        print(g['group'],g['query_evaluations'],'R@1',g['recall_at_1_before'],g['recall_at_1_after'],'R@5',g['recall_at_5_before'],g['recall_at_5_after'],'MRR',g['mrr_before'],g['mrr_after'],'improved/worse',g['improved'],g['worse'],flush=True)
    print(OUT,flush=True)

if __name__=='__main__':main()
