import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'processed'
OUT = ROOT / 'tourapi_manual_review_completed.csv'

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def distance(a, b):
    lat1, lat2 = map(math.radians, [float(a['mapy']), float(b['mapy'])])
    dlat = lat2-lat1
    dlon = math.radians(float(b['mapx'])-float(a['mapx']))
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371000*2*math.asin(min(1, math.sqrt(h)))

# Row-specific semantic review in original candidate order. Default status is MATCH.
# Every note is an assessment of supplied source records, not an external verification.
notes = [
 '전남 지역명 추가 외 강진다원 명칭 동일',
 '고씨굴 및 강원고생대 국가지질공원 동일; 하이픈·중첩 괄호 차이',
 '구이 지역명 생략 및 띄어쓰기 차이; 안덕 건강힐링체험마을 동일',
 '통영 및 편백나무숲 수식 차이; 나폴리농원 동일',
 '남한산성도립공원 동일; 세계유산/세계문화유산 분류 표기 차이',
 '울산 내원암 동일; 중첩 괄호 차이',
 '녹색에너지체험관 및 광주전남지역본부 동일; 중첩 괄호 차이',
 '달기약수탕 및 청송 국가지질공원 동일; 중첩 괄호 차이',
 '영어는 담양대나무숲-담양습지 복합 명칭으로 대상 범위 불명확; 일문은 담양습지 동일',
 '대흥사 동일; 해남 지역 표기와 세계유산 수식 차이',
 '덕유산국립공원 본소·적상분소까지 세 언어 동일',
 '곡성 도림사 동일; 중첩 괄호 차이',
 '땅끝 지역명 생략 외 송호해수욕장 동일',
 '만장굴 및 제주도 국가지질공원 동일; 중첩 괄호 차이',
 '서천 지역명 생략 외 문헌서원 동일',
 '백룡동굴 및 강원고생대 국가지질공원 동일; 중첩 괄호 차이',
 '영광 불갑사 동일; 띄어쓰기·중첩 괄호 차이',
 '국문은 붕어섬과 황톳길을 함께 지칭하나 영어·일본어는 붕어섬만 지칭',
 '비둘기낭폭포 동일; 세계/국가 지질공원 수식 차이',
 '성산일출봉 동일; 영어 괄호의 세계문화유산 표기는 자연유산과 불일치하므로 분류 정보는 정정 필요',
 '국문·영문은 속리산 관광특구, 일문은 보은 법주사라는 개별 시설을 지칭',
 '속초아이와 속초해수욕장 대관람차(속초아이)는 같은 명칭 요소를 공유',
 '순창 지역명 생략 외 쉴랜드 동일',
 '식장산 문화공원·해돋이전망대 명칭은 동일하나 영문 도로명 주소와 국·일문 지번 주소의 대응 미확인',
 '앞산 케이블카 및 앞산 전망대까지 세 언어 동일',
 '법인명 에스투뷰텍 생략; 뷰라운지 동일하고 강산관 2층까지 주소 일치',
 '여좌천 및 로망스다리 동일; 영어 번역 제목에는 다리 생략되나 괄호에 포함',
 '나옹왕사 선명상 치유센터와 인문힐링센터여명은 치유센터라는 공통 기능만 확인; 명칭 변경 또는 별도 시설 가능',
 '영월 장릉·단종 동일; UNESCO 세계유산 표기 차이',
 '오륙도 및 부산 국가지질공원 동일; 중첩 괄호 차이',
 '화원 지역명 추가 외 오시아노 관광단지 동일',
 '국문·일문은 왕인박사유적지와 성기동국민관광지 병기; 영어는 유적지만 지칭',
 '제주 외돌개 동일; 중첩 괄호 차이',
 '우두산 출렁다리 및 거창 Y자형 출렁다리 동일',
 '증도 지역명 추가 외 우전해수욕장 동일',
 '광주 원각사 동일; 중첩 괄호 차이',
 '원대리 자작나무 숲과 속삭이는 자작나무 숲 동일',
 '국문·일문은 유달유원지, 영문은 목포 스카이워크라는 개별 명소를 지칭',
 '세 언어 모두 인천국제공항 제2여객터미널 명시; 영어 한국어 괄호 생략',
 '제주 지역명 생략 외 교래자연휴양림 동일',
 '서울 조계사 동일; 중첩 괄호 차이',
 '광주 증심사 동일; 중첩 괄호 차이',
 '채석강 동일; 전북 서해안/서해안권 지질공원 표기 차이',
 '천마총 및 대릉원 동일; 중첩 괄호 차이',
 '구례 천은사 동일; 중첩 괄호 차이',
 '국문 천장굴과 영·일문 독립문바위는 서로 다른 지형 명칭 및 유형(굴/바위)',
 '청산도 명칭은 동일하나 영어 주소가 도청4길 19-3으로 국·일문 도청3길 30-1과 다름',
 '봉화 청옥산 동일; 국문은 석포면·소천면, 외국어는 석포면 대현리로 범위 상세도 차이',
 '청풍문화유산단지와 청풍문화재단지는 청풍·문화·단지를 공유하며 영·일 번역 의미 동일',
 '코코리컬러 및 코코리색채연구소 명칭 차이; 양쪽 모두 연희본점 명시, 건물 주소 동일',
 '국문은 태평염전 해양힐링스파, 영·일문은 태평염전 전체를 지칭하여 시설 범위 다름',
 '포레스트 리솜 해브나인 스파 동일; 웰니스 수식어 추가',
 '포천/허브아일랜드 수식 차이; 허브힐링센터 및 포천 소재 주소 동일',
 '국문 한국의료관광비즈니스센터와 영·일문 MJ FLB는 시설명 및 층·호수가 달라 별도 입주 시설',
 '한국전통음식문화체험관 및 정강원 동일; 괄호·어순 차이',
 '한려해상국립공원 오동도 동일; 괄호/하이픈 차이',
 '한방스파 여용국/여용국 한방스파 어순 차이',
 '함덕해수욕장 및 함덕 서우봉 해변 동일',
 '여수 향일암 동일; 중첩 괄호 차이',
 '화암동굴 및 강원고생대 국가지질공원 동일; 중첩 괄호 차이',
 '화홍문 동일; 한자 병기·한국어 괄호 차이',
 'F1963 세 언어 동일; 영문에 한글 괄호 없음',
 'SPA 1978 세 언어 동일; 대소문자 차이',
]

# Address review compares locality, road/lot, and building/floor when provided.
addr_exceptions = {
 9: ('CONFLICT', '국·일문 응용리 104-2, 영문 태목리 656-2로 지번 불일치'),
 19: ('PARTIAL', '대회산리 동일; 국문만 415-2 기재'),
 24: ('UNVERIFIED_EQUIVALENCE', '국·일문 세천동 산43-5, 영문 세천공원로 32-834; 지번↔도로명 대응 미확인'),
 33: ('PARTIAL', '서홍동 동일; 국문만 지번 791 기재'),
 47: ('CONFLICT', '국·일문 도청3길 30-1, 영문 도청4길 19-3'),
 48: ('PARTIAL', '봉화군 석포면 일치; 국문은 소천면도 병기, 영·일문은 대현리 기재'),
 54: ('CONFLICT', '청계천로 40 동일하나 국문 서울센터 1층, 영·일문 8층 809호'),
 57: ('PARTIAL', '석촌호수로12길 60 동일; 국문만 2층 명시'),
 63: ('PARTIAL', '영동대로 416 동일; 국문만 지하 2층 명시'),
}
address_notes = {
 14: '만장굴길 182 동일; 국문 addr2에 구좌읍 병기',
 15: '서원로172번길 66 동일; 일문 addr2에 기산면 병기',
 25: '앞산순환로 574-114 동일; 국문만 봉덕동 병기',
 26: '달구벌대로528길 15 및 강산관 2층 동일',
 36: '중앙로 197 동일; 국문만 금남로4가 병기',
 37: '자작나무숲길 760 동일; 국문만 안내소~자작나무숲 메모',
 38: '해양대학로 59 동일하나 유원지와 스카이워크의 시설 단위가 다름',
 41: '우정국로 55 동일; 국문만 견지동 병기',
 50: '연희로 193-3 동일; 영어만 Cocory Building 병기',
 51: '지도증도로 1083-4 동일하나 염전과 스파는 시설 범위 다름',
 55: '금당계곡로 2010-13 동일; 국문 addr2 용평면 병기',
}
review = {
 9: ('REVIEW','MATCH','영문 상세설명과 습지·대나무숲 지번 및 운영 주체 확인'),
 18: ('REVIEW','REVIEW','황톳길이 붕어섬과 동일 관광 콘텐츠인지 3개 언어 상세설명 확인'),
 24: ('REVIEW','MATCH','도로명주소↔지번주소 동일 필지 여부 확인'),
 28: ('REVIEW','REVIEW','시설의 공식 명칭 변경 이력 또는 운영 주체 및 상세설명 확인'),
 32: ('REVIEW','MATCH','두 시설이 공동 관광 콘텐츠인지 및 영문 소개 범위 확인'),
 47: ('REVIEW','MATCH','청산도 소개의 대상 범위와 대표주소 변경 여부 확인'),
}
unmatch = {
 21: ('MATCH','UNMATCH','관광특구와 법주사를 분리하고 일본어 관광특구 ID를 다시 탐색'),
 38: ('UNMATCH','MATCH','유원지와 스카이워크를 분리하고 해당 영어 ID 재탐색'),
 46: ('UNMATCH','UNMATCH','천장굴과 독립문바위를 별도 지형으로 관리하고 해당 외국어 ID 재탐색'),
 51: ('UNMATCH','UNMATCH','염전 전체와 개별 스파를 분리하고 스파 외국어 ID 재탐색'),
 54: ('UNMATCH','UNMATCH','1층 센터와 8층 809호 입주사를 분리하고 센터 외국어 ID 재탐색'),
}

alternatives = {
 21: [('ko','125906','별도 보은 법주사 국문 레코드'),('en','264271','별도 보은 법주사 영문 레코드')],
 38: [('en','4007363','유달유원지 영문 대체 후보'),('ko','2714513','목포 스카이워크 별도 국문 레코드')],
 46: [('ko','2614617','독립문바위 별도 국문 레코드')],
}

def main():
    candidates = read(ROOT/'tourapi_manual_review.csv')
    source = read(ROOT/'tourapi_clean.csv')
    assert len(candidates) == len(notes) == 63 and len(source) == 18038
    index = defaultdict(list)
    coords = defaultdict(list)
    for r in source:
        index[r['language'], r['source_contentid']].append(r)
        try:
            x,y = float(r['mapx']), float(r['mapy'])
            if 124<=x<=132 and 33<=y<=39.5:
                coords[f'{round(x,5)}_{round(y,5)}'].append((r['language'],r['source_contentid']))
        except ValueError:
            pass
    langs = ('ko','en','ja')
    output = []
    for i,row in enumerate(candidates,1):
        records = {l:index[l,row['id_'+l]] for l in langs}
        assert all(len(v)==1 for v in records.values()), (i,records)
        r = {l:records[l][0] for l in langs}
        assert all(r[l]['title']==row['title_'+l] for l in langs), i
        assert Counter(coords[row['coord']])==Counter((l,row['id_'+l]) for l in langs), i
        assert len(set(row['id_'+l] for l in langs))==3, i
        e,j,ej=distance(r['ko'],r['en']),distance(r['ko'],r['ja']),distance(r['en'],r['ja'])
        astatus,anote = addr_exceptions.get(i, ('MATCH',address_notes.get(i,'시군구·도로명/지번 및 건물번호가 언어 간 대응')))
        es,js,action = review.get(i,unmatch.get(i,('MATCH','MATCH','최종 매칭 데이터 포함')))
        status = 'UNMATCH' if 'UNMATCH' in (es,js) else 'REVIEW' if 'REVIEW' in (es,js) else 'MATCH'
        out = dict(row)
        out.update(review_status=status, review_en=es, review_ja=js,
                   decision_mode='MANUAL_REQUIRED' if status=='REVIEW' else 'AI_SOURCE_REVIEW',
                   manual_check_required='YES' if status=='REVIEW' else 'NO',
                   review_note=notes[i-1], address_status=astatus, address_note=anote,
                   id_check='PASS', id_note='언어+contentID 기준 각 1건 연결, 원본 제목 일치; 언어 간 ID는 별개이며 동일성 근거는 아님',
                   coordinate_check='PASS', coord_group_ko_en_ja='1/1/1',
                   distance_ko_en_m=f'{e:.6f}',distance_ko_ja_m=f'{j:.6f}',distance_en_ja_m=f'{ej:.6f}',
                   max_distance_m=f'{max(e,j,ej):.6f}',
                   action=action, external_verification='NOT_PERFORMED', manual_final_status='', manual_final_note='')
        evidence = []
        for l, cid, description in alternatives.get(i, []):
            alt = index[l,cid]
            assert len(alt)==1
            alt = alt[0]
            evidence.append(f"{description}: {l}/{cid} {alt['title']} | {alt['addr1']} | {alt['mapx']},{alt['mapy']} | 기존 국문 좌표와 {distance(r['ko'],alt):.1f}m 차이")
        out['alternative_source_evidence']='; '.join(evidence)
        if i==38:
            out['action']='기존 영문 스카이워크 ID 제외; 영문 4007363 대체 후보 상세 확인 후 재매칭'
        if i in (21,46,51,54):
            out['action']+='; 전체 원본 제목의 해당 국문 핵심 명칭 검색에서 필요한 외국어 대체 후보 미확인'
        for l in langs:
            for col in ('addr1','addr2','mapx','mapy'):
                out[f'{col}_{l}']=r[l][col]
            out[f'overview_available_{l}']='YES' if r[l]['overview'].strip() else 'NO'
        output.append(out)
    # Never overwrite an existing review that may contain manual confirmations.
    with OUT.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    check=read(OUT)
    assert check==output and len({x['coord'] for x in check})==63
    assert all(all(x[k]==c[k] for k in c) for x,c in zip(check,candidates))
    print(Counter(x['review_status'] for x in check))
    print(Counter(x['address_status'] for x in check))
    print('max distance',max(float(x['max_distance_m']) for x in check))
    print('overview available',sum(x[f'overview_available_{l}']=='YES' for x in check for l in langs))
    for x in check:
        if x['review_status']!='MATCH': print(x['review_status'],x['title_ko'],x['review_note'])

if __name__=='__main__': main()
