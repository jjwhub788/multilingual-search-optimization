import csv
import random
from collections import Counter
from review_tourapi import ROOT, read

source = read(ROOT/'tourapi_final_matches.csv')
auto = [r for r in source if r['match_source']=='AUTO_NAME_MATCH']
assert len(auto)==279
sample = random.Random(20261119).sample(auto,20)
details = {
    '2931362': ('MATCH','큰끝등대와 등대 유형 동일; 영어는 평사리까지, 국·일문은 산 1-1까지 기재'),
    '126469': ('MATCH','한라산 영실 동일; 서귀포시 영실로 246 일치'),
    '126802': ('MATCH','봉산문화거리 동일; 대구 중구 봉산문화길 38 일치'),
    '127110': ('MATCH','강진 전라병영성 동일; 강진 병영성로 175 일치'),
    '2945376': ('MATCH','서해랑 제부도 해상케이블카 동일; 화성 전곡항로 1-10 일치'),
    '126375': ('MATCH','진도 운림산방 동일; 진도 운림산방로 315 일치'),
    '127074': ('MATCH','오웬기념각 동일; 광주 남구 백서로70번길 6 일치'),
    '2605353': ('MATCH','송정동 떡갈비 골목 동일; 광산구 광산로29번길 일치'),
    '3346664': ('MATCH','소백산 생태탐방원 동일; 영주 영단로 253 일치'),
    '2719004': ('MATCH','고창 청농원 동일; 고창 청천길 41-27 일치'),
    '126998': ('MATCH','만휴정 동일; 안동 묵계하리길 42 일치'),
    '3343355': ('MATCH','수안보온천 족욕길 동일; 충주 주정산로 35 일치'),
    '947611': ('REVIEW','개항장 거리 명칭·구역·좌표 일치. 영문 제물량로232번안길 27과 국·일문 관동1가의 실제 주소 대응은 미확인'),
    '2569503': ('MATCH','타임앤블레이드 박물관 동일; 파주 헤이리마을길 41 일치'),
    '127951': ('MATCH','하슬라아트월드 동일; 강릉 율곡로 1441 일치'),
    '2037308': ('MATCH','엑스포해양공원 동일; 여수 박람회길 1 일치'),
    '3082578': ('MATCH','쏠비치 양양 오션플레이 동일; 양양 선사유적로 678 일치'),
    '127539': ('MATCH','광주패밀리랜드 동일; 광주 북구 우치로 677 일치'),
    '126446': ('MATCH','가파도 동일; 서귀포 대정읍 가파로 일치'),
    '126537': ('MATCH','북촌한옥마을 동일; 서울 종로구 계동길 37 일치'),
}
assert {r['id_ko'] for r in sample}==set(details)
results=[]
for i,r in enumerate(sample,1):
    status,note=details[r['id_ko']]
    out=dict(r)
    out['sample_review_status']=status
    out.update(sample_order=i,sample_seed=20261119,sample_review_note=note,
               sample_action='상세정보 수집 후보로 유지' if status=='MATCH' else '대표주소 대응 확인 후 수집 후보 확정')
    results.append(out)
dest=ROOT/'tourapi_sample_review_20.csv'
with dest.open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
check=read(dest)
assert len(check)==20 and len({r['id_ko'] for r in check})==20
assert {r['id_ko'] for r in check}==set(details)
print(dest)
print(Counter(r['sample_review_status'] for r in check))
print('max distance',max(float(r['max_distance_m']) for r in check))
