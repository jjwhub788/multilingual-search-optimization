"""Follow-up on new queries, authored after the initial experiment.

Not an independent/blinded holdout: queries are new, but informed by source
descriptions and the known intervention. Compare two frozen embedding sets.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from run_bge_m3 import DATA, MODEL, ROOT, encode, evaluate, metrics_by_group, load_baseline_model
from run_description_enrichment import SUPPLEMENTS

BASE=ROOT/'results/bge_m3_dense_full'
AUG=ROOT/'results/tourapi_description_enrichment'
OUT=ROOT/'results/tourapi_description_enrichment_retest'
LANGS=('ko','en','ja')

INTENTS=[
 ('O1','OMITTED_FEATURE','125724','고씨 가족이 전쟁을 피해 숨어 지내 이름이 붙은 강원도 동굴','A Gangwon cave named after a family that hid there during a war','戦乱から逃れた一族にちなんで名付けられた江原道の洞窟'),
 ('O2','OMITTED_FEATURE','2605528','옛날 교복을 빌려 입고 철도 선로에서 사진을 찍고 싶어','Where can I pose on railway tracks in an old-fashioned school uniform?','昔風の学生服を着て線路のそばで記念写真を撮りたい'),
 ('O3','OMITTED_FEATURE','127074','광주에 있는 예배당으로 쓰였던 네덜란드식 회색 벽돌 건축물','A Dutch-style gray brick building in Gwangju once used for worship','光州にある礼拝堂として使われたオランダ式の灰色レンガの建物'),
 ('E1','EXISTING_FEATURE','125724','여러 층으로 되어 있고 아래쪽에 지하수가 흐르는 천연기념물 동굴','A protected natural monument cave with multiple levels and a stream of groundwater below','多層構造で最下層に地下水が流れる天然記念物の洞窟'),
 ('E2','EXISTING_FEATURE','2605528','1944년 제지공장에 원료를 나르던 군산의 옛 철도 주변 마을','A Gunsan neighborhood along a railway built in 1944 to transport materials for a paper mill','1944年に製紙工場へ原料を運んだ群山の鉄道沿いの町'),
 ('E3','EXISTING_FEATURE','127074','기독간호대학 안에 자리한 선교사 기념관으로 지금은 강당으로 쓰는 건물','A missionary memorial on a Christian nursing college campus now used as an auditorium','キリスト教系看護大学の敷地内にあり現在は講堂として利用される宣教師の記念館'),
 ('C1','UNCHANGED_CONTROL','128661','정선의 옛 금광과 자연 동굴을 함께 둘러보는 체험 관광','An experiential tour combining an old gold mine and a natural cavern in Jeongseon','旌善で昔の金鉱山と天然洞窟を一緒に巡る体験観光'),
 ('C2','UNCHANGED_CONTROL','127073','광주에 남아 있는 1920년대 미국 선교사의 서양식 주택','A surviving Western-style residence of an American missionary from the 1920s in Gwangju','光州に現存する1920年代のアメリカ人宣教師の西洋式住宅'),
 ('C3','UNCHANGED_CONTROL','126451','제주에서 야자수와 하얀 모래, 수심 얕은 에메랄드빛 해변을 찾고 싶어','A Jeju beach with palm trees, white sand and shallow turquoise water','済州でヤシの木と白い砂、遠浅のエメラルド色の海を楽しめるビーチ'),
 ('C4','UNCHANGED_CONTROL','125913','강 가운데 바위 위 정자와 강 건너편 계절 꽃 정원이 있는 단양 명소','A Danyang attraction with a riverside rock pavilion and a seasonal flower garden across the river','丹陽で川の岩の上の亭子と対岸の季節の花畑を見られる名所'),
 ('C5','UNCHANGED_CONTROL','3081515','전복 요리와 해조류 스파, 머드 랩핑을 체험할 수 있는 해양 웰니스 시설','A marine wellness facility offering abalone meals, seaweed spa treatments and mud wraps','アワビ料理と海藻スパ、泥ラッピングを体験できる海洋ウェルネス施設'),
 ('C6','UNCHANGED_CONTROL','126537','가회동과 계동의 오래된 기와집 처마가 이어지는 서울 주거 골목','A historic Seoul residential neighborhood in Gahoe-dong and Gye-dong with rows of tiled houses','嘉会洞と桂洞で昔ながらの瓦屋根の家が軒を連ねるソウルの住宅路地'),
]

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    docs=read(DATA); prior=read(BASE/'queries.csv')
    before_vec=np.load(BASE/'document_embeddings.npy');after_vec=np.load(AUG/'document_embeddings_enriched.npy')
    assert before_vec.shape==after_vec.shape==(300,1024)
    assert len(docs)==300
    changed={(d['place_id'],d['language']) for i,d in enumerate(docs) if not np.array_equal(before_vec[i],after_vec[i])}
    assert changed==set(SUPPLEMENTS)
    queries=[]
    for iid,kind,pid,*texts in INTENTS:
        for lang,txt in zip(LANGS,texts):
            queries.append(dict(query_id=f'new-{iid}-{lang}',query_type=kind,query_language=lang,place_id=pid,query=txt))
    assert len(queries)==36 and not ({r['query'] for r in queries}&{r['query'] for r in prior})
    write(OUT/'new_queries.csv',queries)
    meta=json.loads((BASE/'run_metadata.json').read_text())
    model=load_baseline_model(meta)
    qvec=encode(model,[q['query'] for q in queries],4,'new queries',np)
    np.save(OUT/'new_query_embeddings.npy',qvec)
    before=evaluate(docs,queries,before_vec,qvec,np);after=evaluate(docs,queries,after_vec,qvec,np)
    comp=[]
    for a,b in zip(before,after):
        assert (a['query_id'],a['target_language'])==(b['query_id'],b['target_language'])
        gain=a['rank']-b['rank']
        comp.append(dict(query_id=a['query_id'],query_type=a['query_type'],query_language=a['query_language'],target_language=a['target_language'],place_id=a['place_id'],query=a['query'],rank_before=a['rank'],rank_after=b['rank'],rank_gain=gain,change='IMPROVED' if gain>0 else 'WORSE' if gain<0 else 'UNCHANGED',correct_score_before=a['correct_score'],correct_score_after=b['correct_score'],top1_before=a['top1_title'],top1_after=b['top1_title']))
    assert len(comp)==144
    write(OUT/'new_query_comparison.csv',comp)
    groups=[]
    for kind in ('OMITTED_FEATURE','EXISTING_FEATURE','UNCHANGED_CONTROL','ALL'):
        rows=[r for r in comp if kind=='ALL' or r['query_type']==kind]
        for scope in ('ALL_PLACES','ENRICHED_TARGETS','SINGLE_LANGUAGE'):
            sub=[r for r in rows if (scope=='ALL_PLACES' and r['target_language']=='ALL_PLACES') or (scope=='ENRICHED_TARGETS' and (r['place_id'],r['target_language']) in SUPPLEMENTS) or (scope=='SINGLE_LANGUAGE' and r['target_language'] in LANGS)]
            if not sub: continue
            rec=dict(query_type=kind,scope=scope,queries=len(sub),improved=sum(r['change']=='IMPROVED' for r in sub),worse=sum(r['change']=='WORSE' for r in sub),unchanged=sum(r['change']=='UNCHANGED' for r in sub))
            for s in ('before','after'):
                ranks=[r['rank_'+s] for r in sub]
                rec['top1_'+s]=sum(r==1 for r in ranks)
                rec['recall1_'+s]=round(sum(r==1 for r in ranks)/len(sub),6)
                rec['recall5_'+s]=round(sum(r<=5 for r in ranks)/len(sub),6)
                rec['mrr_'+s]=round(sum(1/r for r in ranks)/len(sub),6)
            groups.append(rec)
    write(OUT/'retest_summary.csv',groups)
    (OUT/'metadata.json').write_text(json.dumps(dict(new_queries=36,intents=12,model=MODEL,device=meta['device'],max_length=meta['max_seq_length'],evaluation='two frozen corpus embedding sets',limitation='New strings and some alternate intents, but authored after seeing original failures and corrections; not independent or blinded.'),ensure_ascii=False,indent=2),encoding='utf-8')
    for r in groups: print(r,flush=True)

if __name__=='__main__': main()
