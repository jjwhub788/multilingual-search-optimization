"""Analyze description length and semantic retrieval errors from the pilot."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/tourapi_standardized.csv'
EXPERIMENT=ROOT/'results/bge_m3_dense_full'
OUTPUT=ROOT/'results/tourapi_gap_analysis'
LANGS=('ko','en','ja')

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# Source-text audit of all mixed-corpus top-1 errors and the eight ko->ja errors.
AUDIT={
 ('en','ALL_PLACES','125724'):('FEATURE_OMISSION','국문에 임진왜란 당시 고씨 가족의 피난 사실이 있으나 영문·일문에는 해당 사실이 없음. 화암동굴 영문은 금광 및 역사·교육 내용을 포함.','원문과 각 언어 설명의 핵심 역사 사실 누락 여부 비교'),
 ('ja','ALL_PLACES','125724'):('FEATURE_OMISSION','고씨굴 일문에 임진왜란 피난 사실이 없으며 백룡동굴 일문에는 영월댐·석회동굴이 언급됨.','국문 고유 역사 특징을 보완한 번역 설명으로 비교'),
 ('en','ALL_PLACES','126435'):('SIMILAR_ATTRACTION','성산일출봉 영문에는 분화구·일출이 명시됨. 함덕해수욕장 영문에도 제주·화산 오름·정상 전망·해안이 등장.','분화구와 일출 같은 구별 특징의 순위 영향 비교'),
 ('ja','ALL_PLACES','3343355'):('CROSS_LANGUAGE_RANKING','수안보온천 일문에는 충주시·천연 온천·족욕 정보가 있으나 제주 교래자연휴양림 영문이 1위. 구체적 족욕 장소보다 휴식 관련 의미가 우세했을 가능성.','동일 언어 제한 및 핵심 특징 중심 검색 비교'),
 ('ja','ALL_PLACES','127074'):('PARTIAL_OMISSION_AND_SIMILARITY','오웬기념각 일문에는 1914년·선교사·양림동은 있으나 회색 벽돌 설명이 없음. 우일선 선교사 사택도 광주 서양식 선교사 관련 건축물.','건축 특징과 건립 연도를 함께 유지한 설명 비교'),
 ('ko','ja','126446'):('CROSS_LANGUAGE_RANKING','가파도 일문에는 청보리 축제와 오르막이 없다는 내용이 모두 있음. 그럼에도 오동도가 1위. 핵심 사실 누락만으로 설명할 수 없음.','교차언어 의미 정렬 및 섬 관련 경쟁 문서 비교'),
 ('ko','ja','2605528'):('FEATURE_OMISSION','경암동 철길마을 일문에 철로와 군산의 역사는 있으나 국문 후반부 레트로 상점·교복 체험 내용이 없음.','누락된 체험 요소를 보완한 설명 비교'),
 ('ko','ja','125724'):('FEATURE_OMISSION','일문에 임진왜란 피난 사실이 없고 화암동굴 일문에는 일제강점기 채굴 역사가 있음.','역사적 특징 보완 전후 비교'),
 ('ko','ja','1333464'):('CROSS_LANGUAGE_RANKING','백룡동굴 일문에 평창의 동굴 및 눈에 띄는 계란프라이 형태 석순 정보가 포함됨(평창 지역은 주소에서 확인). 정강원이 1위.','형태 묘사와 음식 표현 간 의미 혼동 여부 비교'),
 ('ko','ja','126537'):('CROSS_LANGUAGE_RANKING','북촌한옥마을 일문에는 경복궁·창덕궁·전통 한옥·골목이 모두 있으나 정강원이 1위. 정답 13위.','핵심 지명과 단락 단위 검색을 비교'),
 ('ko','ja','3343355'):('CROSS_LANGUAGE_RANKING','일문에 충주·천연 온천수·족욕 내용이 있으나 델피노 오션플레이가 1위.','지역/활동 키워드 중심 접근과 비교'),
 ('ko','ja','127074'):('PARTIAL_OMISSION_AND_SIMILARITY','일문에 선교사·양림동·1914년은 있으나 회색 벽돌 특징이 빠짐. 우일선 선교사 사택과 주제 유사.','벽돌 특징 보완 전후 성능 및 유사 건축물 비교'),
 ('ko','ja','127951'):('CROSS_LANGUAGE_RANKING','하슬라아트월드 일문에 조각공원·바다 정원·피노키오 미술관이 명시되나 속초 대관람차가 1위.','지명·문화 체험 핵심 구절 검색과 전체 문서 검색 비교'),
}

def stats(values):
    return dict(mean=round(statistics.mean(values),2),median=round(statistics.median(values),2),min=min(values),max=max(values),std=round(statistics.stdev(values),2),p25=round(float(np.percentile(values,25)),2),p75=round(float(np.percentile(values,75)),2))

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    docs=read(DATA)
    results=read(EXPERIMENT/'query_results.csv')
    assert len(docs)==300 and len(results)==1440
    assert all(d['source']=='tourapi' and d['description'].strip() for d in docs)
    tokenizer=AutoTokenizer.from_pretrained('BAAI/bge-m3',local_files_only=True)
    desc=[]
    for d in docs:
        text=d['description']
        tokens=len(tokenizer(text,add_special_tokens=False,truncation=False)['input_ids'])
        desc.append(dict(source='tourapi',place_id=d['place_id'],language=d['language'],title=d['title'],description_chars=len(text),description_chars_no_spaces=sum(not c.isspace() for c in text),description_bge_tokens=tokens))
    write(OUTPUT/'document_description_lengths.csv',desc)
    idx={(d['place_id'],d['language']):d for d in desc}
    assert len(idx)==300
    summaries=[]
    for lang in LANGS:
        group=[d for d in desc if d['language']==lang]
        assert len(group)==100
        rec=dict(language=lang,document_count=len(group))
        for f in ('description_chars','description_chars_no_spaces','description_bge_tokens'):
            rec.update({f'{f}_{k}':v for k,v in stats([r[f] for r in group]).items()})
        summaries.append(rec)
    write(OUTPUT/'language_description_stats.csv',summaries)
    # Pair each place, preserving the 100 aligned units.
    paired=[]
    for pid in sorted({d['place_id'] for d in desc}):
        row=dict(place_id=pid,title_ko=idx[pid,'ko']['title'])
        for lang in LANGS:
            for f in ('description_chars','description_bge_tokens'):
                row[f'{f}_{lang}']=idx[pid,lang][f]
        for lang in ('en','ja'):
            row[f'{lang}_to_ko_token_ratio']=round(row[f'description_bge_tokens_{lang}']/row['description_bge_tokens_ko'],4)
        paired.append(row)
    write(OUTPUT/'paired_description_lengths.csv',paired)
    semantic=[r for r in results if r['query_type']=='SEMANTIC']
    assert len(semantic)==240
    enriched=[]; audits=[]
    for row in semantic:
        out=dict(row)
        rank=int(row['rank'])
        qlang,tlang,pid=row['query_language'],row['target_language'],row['place_id']
        out['correct_title_ko']=idx[pid,'ko']['title']
        out['top1_title_ko']=idx[row['top1_place_id'],'ko']['title']
        out['correct_target_description_tokens']=idx[pid,tlang]['description_bge_tokens'] if tlang in LANGS else ''
        out['top1_description_tokens']=idx[row['top1_place_id'],row['top1_language']]['description_bge_tokens']
        out['score_gap_top1_minus_correct']=round(float(row['top1_score'])-float(row['correct_score']),6)
        out['outcome']='HIT' if rank==1 else 'MISS'
        enriched.append(out)
        key=(qlang,tlang,pid)
        if key in AUDIT:
            assert rank>1
            cat,note,action=AUDIT[key]
            audits.append(dict(**out,observed_pattern=cat,evidence_note=note,next_comparison=action))
    assert len(audits)==len(AUDIT)
    write(OUTPUT/'semantic_results_with_lengths.csv',enriched)
    write(OUTPUT/'semantic_top1_failures.csv',[r for r in enriched if r['outcome']=='MISS'])
    write(OUTPUT/'semantic_failure_review.csv',audits)
    group_stats=[]
    # Restrict token/rank association to one target language per group; n=20.
    for qlang in LANGS:
        for tlang in LANGS:
            group=[r for r in enriched if r['query_language']==qlang and r['target_language']==tlang]
            assert len(group)==20
            hits=[r for r in group if r['outcome']=='HIT'];misses=[r for r in group if r['outcome']=='MISS']
            tokens=[r['correct_target_description_tokens'] for r in group];ranks=[int(r['rank']) for r in group]
            corr=None if len(set(ranks))==1 else float(spearmanr(tokens,ranks).statistic)
            group_stats.append(dict(query_language=qlang,target_language=tlang,queries=len(group),top1_hits=len(hits),top1_misses=len(misses),top1_rate=round(len(hits)/len(group),4),mean_target_tokens_hit=round(statistics.mean(r['correct_target_description_tokens'] for r in hits),2) if hits else '',mean_target_tokens_miss=round(statistics.mean(r['correct_target_description_tokens'] for r in misses),2) if misses else '',spearman_tokens_vs_rank=round(corr,4) if corr is not None else '',interpretation='탐색적 연관성; 토큰 수가 실제 정보량 또는 원인을 증명하지 않음'))
    write(OUTPUT/'semantic_length_rank_stats.csv',group_stats)
    for s in summaries: print(s['language'],s['description_chars_mean'],s['description_bge_tokens_mean'],s['description_bge_tokens_median'])
    for s in group_stats: print(s['query_language'],s['target_language'],s['top1_rate'],s['mean_target_tokens_hit'],s['mean_target_tokens_miss'],s['spearman_tokens_vs_rank'])
    print('misses',len([r for r in enriched if r['outcome']=='MISS']),'reviewed',len(audits),OUTPUT)

if __name__=='__main__': main()
