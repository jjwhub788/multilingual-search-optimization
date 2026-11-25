"""Dense BGE-M3 retrieval baseline for the aligned 100-place TourAPI corpus.

Run from the project root: .venv/bin/python src/run_bge_m3.py
Queries and document embeddings are evaluated by true place_id, not name.
"""
import argparse
import csv
import hashlib
import json
import os
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed/tourapi_pilot_details_300.csv'
OUT = ROOT/'results/bge_m3_dense_full'
LANGS = ('ko','en','ja')
MODEL = 'BAAI/bge-m3'

# Independently worded queries, aligned by intent rather than title. The place
# identity is derived from the supplied Korean overview, not retrieval outputs.
SEMANTIC = [
 ('126446','제주에서 봄에 청보리밭을 걷기 좋은 평탄한 섬','A flat island off Jeju for strolling through green barley fields in spring','済州島の沖で春に青麦畑を散歩できる平坦な島'),
 ('128151','월출산 기슭에서 녹차밭을 구경하고 찻잎을 따볼 수 있는 곳','A tea plantation at the foot of Wolchulsan where visitors can pick tea leaves','月出山のふもとで緑茶畑を眺め茶摘み体験ができる場所'),
 ('2605528','군산에서 옛 철로 옆 레트로 상점과 추억의 교복 사진을 즐기는 마을','An old railway neighborhood in Gunsan with nostalgic shops and vintage school uniforms','群山で廃線沿いのレトロな店や昔の制服で写真撮影を楽しむ町'),
 ('125724','영월에서 임진왜란 피난처로 쓰였던 석회암 동굴','A limestone cave in Yeongwol used as a refuge during the Japanese invasions','寧越で文禄・慶長の役の際に避難場所となった石灰洞窟'),
 ('127933','단편소설 봄봄과 동백꽃 작가의 생가를 복원한 문학 여행지','A literary village built around the restored childhood home of the author of Spring Spring and Camellias','短編小説「春・春」と「椿」の作家の生家を復元した文学の村'),
 ('125449','조선 시대 한양을 지키던 산성 성곽을 따라 걷고 싶어','Walk along the mountain fortress walls that once protected Joseon-era Hanyang','朝鮮時代に漢陽を守った山城の城壁に沿って歩きたい'),
 ('125913','단양 남한강 한가운데 솟은 세 바위 봉우리와 정자','Three rock peaks in the Namhan River in Danyang with a pavilion on the largest one','丹陽の南漢江に浮かぶ三つの岩峰と亭子が見られる場所'),
 ('126452','제주에서 거대한 용암석주가 남아 있는 용암 동굴','A Jeju lava tube with a giant lava column inside','巨大な溶岩石柱が残る済州の溶岩洞窟'),
 ('1333464','평창 석회동굴에서 계란프라이처럼 생긴 석순을 볼 수 있는 곳','A limestone cave in Pyeongchang with stalagmites shaped like fried eggs','平昌で目玉焼きのような形の石筍が見られる石灰洞窟'),
 ('126537','서울 경복궁과 창덕궁 사이 전통 한옥 골목 산책','A walk through traditional Korean houses between Gyeongbokgung and Changdeokgung in Seoul','ソウルの景福宮と昌徳宮の間にある伝統韓屋の路地を散策'),
 ('126435','제주 바다 위 화산 분화구 정상에서 해돋이 보는 곳','A sunrise hike to the crater of a volcanic cone overlooking Jeju’s sea','済州の海に面した火山の火口の頂上から日の出を眺める場所'),
 ('2814180','속초 바다와 설악산을 동시에 볼 수 있는 대형 관람차','A giant Ferris wheel overlooking both Sokcho’s sea and Seoraksan','束草の海と雪岳山を一望できる大観覧車'),
 ('3343355','충주에서 천연 온천수로 발을 담그며 쉴 수 있는 산책길','A walking path in Chungju where you can soak your feet in natural hot spring water','忠州で天然温泉の足湯を楽しみながら休める散歩道'),
 ('2385686','부산 기장에서 오래된 대나무와 편백 숲을 만날 수 있는 사유림','A centuries-old private forest in Gijang, Busan, with bamboo and cypress trees','釜山機張で竹林やヒノキ林を楽しめる歴史ある私有林'),
 ('127074','광주 양림동에서 선교사를 기리기 위해 1914년 지은 회색 벽돌 건물','A grey-brick memorial building constructed in 1914 for a missionary in Yangnim-dong, Gwangju','光州楊林洞で宣教師を記念して1914年に建てられた灰色レンガの建物'),
 ('3081515','완도에서 해수와 해조류, 갯벌을 활용한 테라피를 체험하는 시설','A wellness center in Wando offering therapies with seawater, seaweed and tidal mud','莞島で海水・海藻・干潟の泥を利用したセラピーが体験できる施設'),
 ('2755856','거창 산악 협곡 위 세 방향으로 연결된 Y자 현수교','A Y-shaped suspension footbridge spanning a mountain gorge in Geochang','居昌の山間の渓谷に架かる三方向のY字型吊り橋'),
 ('126214','경주에서 신라 금관과 하늘을 나는 말 그림이 발견된 공개 고분','An excavated Silla tomb in Gyeongju famous for a gold crown and a painting of a flying horse','慶州で新羅の金冠と天を飛ぶ馬の絵が発見された公開古墳'),
 ('127951','강릉에서 바다가 보이는 조각공원과 피노키오 인형 미술관','An art park in Gangneung with sea views, sculptures and a Pinocchio puppet museum','江陵で海が見える彫刻公園とピノキオの人形美術館'),
 ('2456767','부산에서 옛 와이어 공장을 전시와 공연 공간으로 바꾼 복합문화시설','A former wire factory in Busan transformed into an arts and performance complex','釜山で旧ワイヤー工場を展示や公演の複合文化施設に改装した場所'),
]

def csv_read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def native_title(title,lang):
    text=unicodedata.normalize('NFKC',title).strip()
    if lang=='ko': return text
    # Drop an appended Korean alias, including nested parentheses, to avoid
    # giving foreign-language test queries Korean tokens for free.
    if text.endswith(')'):
        depth=0
        for i in range(len(text)-1,-1,-1):
            if text[i]==')': depth+=1
            elif text[i]=='(':
                depth-=1
                if depth==0:
                    inside=text[i+1:-1]
                    if any('가'<=c<='힣' for c in inside) and text[:i].strip():
                        return text[:i].strip()
                    break
    return text

def query_rows(docs):
    queries=[]
    for d in docs:
        queries.append(dict(query_id=f"name-{d['place_id']}-{d['language']}",query_type='NAME',query_language=d['language'],place_id=d['place_id'],query=native_title(d['title'],d['language'])))
    for pid,*texts in SEMANTIC:
        for lang,txt in zip(LANGS,texts):
            queries.append(dict(query_id=f'semantic-{pid}-{lang}',query_type='SEMANTIC',query_language=lang,place_id=pid,query=txt))
    assert len(queries)==360 and len({q['query_id'] for q in queries})==360
    return queries

def encode(model,texts,bs,label,np):
    outputs=[]
    for i in range(0,len(texts),bs):
        out=model.encode(texts[i:i+bs],batch_size=bs,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        outputs.append(out.astype(np.float32))
        if (i//bs+1)%10==0 or i+bs>=len(texts): print(f'{label}: {min(i+bs,len(texts))}/{len(texts)}',flush=True)
    result=np.concatenate(outputs)
    assert result.shape==(len(texts),1024) and np.isfinite(result).all()
    return result

def load_baseline_model(metadata):
    """Use the device, precision and sequence length of the saved baseline.

    The original experiment used MPS/float16. On other computers, generate a
    new baseline on CPU before generating any additional embeddings.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    device = metadata['device']
    precision = metadata['precision']
    if metadata['model'] != MODEL or metadata['max_seq_length'] != 1024:
        raise ValueError('Generate the full BGE-M3 baseline with --max-length 1024 first.')
    if (device, precision) not in {('mps', 'float16'), ('cpu', 'float32')}:
        raise ValueError(f'Unsupported baseline device/precision: {device}/{precision}')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError(
            'The saved baseline uses Apple MPS. Re-run the complete experiment '
            'in a separate copy with --device cpu; do not mix CPU and MPS embeddings.'
        )
    options = {}
    revision = metadata.get('model_revision')
    if revision and revision != 'None':
        options['revision'] = revision
    model = SentenceTransformer(MODEL, device=device, model_kwargs={'torch_dtype':precision}, **options)
    model.max_seq_length = metadata['max_seq_length']
    return model

def metrics_by_group(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[row['query_type'],row['query_language'],row['target_language']].append(row['rank'])
        groups[row['query_type'],'ALL',row['target_language']].append(row['rank'])
    result=[]
    for (qtype,q_lang,t_lang), ranks in sorted(groups.items()):
        n=len(ranks)
        result.append(dict(query_type=qtype,query_language=q_lang,target_language=t_lang,queries=n,
                           recall_at_1=round(sum(r<=1 for r in ranks)/n,6),recall_at_5=round(sum(r<=5 for r in ranks)/n,6),recall_at_10=round(sum(r<=10 for r in ranks)/n,6),mrr=round(sum(1/r for r in ranks)/n,6)))
    return result

def evaluate(docs,queries,doc_vec,query_vec,np):
    rows=[]
    score=query_vec@doc_vec.T
    for i,q in enumerate(queries):
        for lang in (*LANGS,'ALL_PLACES'):
            indices=sorted((j for j,d in enumerate(docs) if lang=='ALL_PLACES' or d['language']==lang),key=lambda j:(-float(score[i,j]),docs[j]['place_id'],docs[j]['language']))
            if lang=='ALL_PLACES':
                seen=set(); filtered=[]
                for j in indices:
                    pid=docs[j]['place_id']
                    if pid not in seen:
                        seen.add(pid);filtered.append(j)
                indices=filtered
            assert len(indices)==100
            rank=next(k for k,j in enumerate(indices,1) if docs[j]['place_id']==q['place_id'])
            top=docs[indices[0]]
            rows.append(dict(**q,target_language=lang,rank=rank,top1_place_id=top['place_id'],top1_title=top['title'],top1_language=top['language'],top1_score=round(float(score[i,indices[0]]),6),correct_score=round(float(score[i,indices[rank-1]]),6),top5_place_ids='|'.join(docs[j]['place_id'] for j in indices[:5])))
    return rows

def main():
    global OUT
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true',help='create test queries without loading model')
    p.add_argument('--device',choices=('mps','cpu'),default=None)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--max-length',type=int,default=1024)
    p.add_argument('--revision',default=None,help='Optional Hugging Face model revision')
    p.add_argument('--output-dir',type=Path,default=OUT)
    args=p.parse_args()
    OUT=args.output_dir.resolve()
    # The short exploratory baseline was retired; downstream stages require 1024.
    if OUT == (ROOT/'results/bge_m3_dense_full').resolve() and args.max_length != 1024:
        p.error('Use --max-length 1024 for results/bge_m3_dense_full, or choose a separate --output-dir.')
    OUT.mkdir(parents=True,exist_ok=True)
    docs=csv_read(DATA)
    assert len(docs)==300 and len({(d['place_id'],d['language']) for d in docs})==300
    assert all(d['collection_status']=='COMPLETE' and d['overview'].strip() for d in docs)
    assert {d['place_id'] for d in docs if d['language']=='ko'}=={d['place_id'] for d in docs if d['language']=='en'}=={d['place_id'] for d in docs if d['language']=='ja'}
    queries=query_rows(docs)
    csv_write(OUT/'queries.csv',queries)
    if args.prepare:
        print(f'{len(docs)} documents, {len(queries)} queries prepared');return
    import numpy as np
    import torch
    import importlib.metadata
    from sentence_transformers import SentenceTransformer
    device=args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Model: {MODEL}, device: {device}, max length: {args.max_length}',flush=True)
    # Float16 on Apple GPU limits memory; CPU uses float32.
    model=SentenceTransformer(MODEL,device=device,revision=args.revision,model_kwargs={'torch_dtype':'float16' if device=='mps' else 'float32'})
    model.max_seq_length=args.max_length
    doc_text=[d['search_text'] for d in docs]
    query_text=[q['query'] for q in queries]
    metadata=dict(model=MODEL,model_revision=args.revision or str(getattr(model[0].auto_model.config,'_commit_hash',None)),device=device,precision='float16' if device=='mps' else 'float32',max_seq_length=args.max_length,batch_size=args.batch_size,document_count=len(docs),query_count=len(queries),corpus_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),queries_sha256=hashlib.sha256((OUT/'queries.csv').read_bytes()).hexdigest(),sentence_transformers=importlib.metadata.version('sentence-transformers'),torch=torch.__version__,aggregation='per-language: one document/place; ALL_PLACES: maximum document score per place')
    token_counts=[len(model.tokenizer(t,add_special_tokens=True,truncation=False,verbose=False)['input_ids']) for t in doc_text]
    metadata['truncated_documents']=sum(n>args.max_length for n in token_counts)
    metadata['truncated_documents_by_language']={l:sum(n>args.max_length for n,d in zip(token_counts,docs) if d['language']==l) for l in LANGS}
    metadata['max_document_tokens']=max(token_counts)
    doc_vec=encode(model,doc_text,args.batch_size,'documents',np)
    query_vec=encode(model,query_text,args.batch_size,'queries',np)
    np.save(OUT/'document_embeddings.npy',doc_vec)
    np.save(OUT/'query_embeddings.npy',query_vec)
    csv_write(OUT/'embedding_index.csv',[dict(embedding_row=i,place_id=d['place_id'],language=d['language'],source_contentid=d['source_contentid'],title=d['title']) for i,d in enumerate(docs)])
    ranks=evaluate(docs,queries,doc_vec,query_vec,np)
    csv_write(OUT/'query_results.csv',ranks)
    metrics=metrics_by_group(ranks)
    csv_write(OUT/'metrics.csv',metrics)
    (OUT/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in metrics:
        if r['query_language']=='ALL' and r['target_language']=='ALL_PLACES': print(r,flush=True)
    print(f'RESULTS {OUT}',flush=True)

if __name__=='__main__': main()
