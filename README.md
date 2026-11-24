# K-Search Gap | 다국어 관광정보 검색 가시성 연구

**연구 주제:** 다국어 관광정보의 불균형이 의미 기반 검색 가시성에 미치는 영향과 적응형 정보 보완 전략 연구

한국관광공사 TourAPI의 한국어·영어·일본어 관광정보를 동일 관광지 단위로 연결하고, 다국어 임베딩 모델 BGE-M3에서 정보 부족에 따른 검색 순위 격차를 측정한다. 이후 누락 정보 선별 보완, 기술적으로 동일한 번역 모델을 사용한 전체 번역, 정보 부족 정도에 따른 혼합 보완 전략을 비교한다.

## 1. 연구 데이터와 실험 설계

| 구분 | 실험 조건 |
| --- | --- |
| 데이터 | 한국관광공사 TourAPI에서 선정한 관광지 100곳 |
| 대상 문서 | 한국어 100개, 영어 100개, 일본어 100개 (총 300개) |
| 초기 검색 실험 | 관광지 이름 300개 + 의미 기반 검색어 60개 = 360개 |
| 본 실험 검색어 | 관광지당 대표 특징 1개, 세부 특징 1개 × 100곳 × 3개 언어 = 600개 |
| 검색 범위 | 한국어, 영어, 일본어 각각의 문서 및 세 언어 통합 검색 |
| 검색 모델 | BAAI/bge-m3, Dense Retrieval, 1,024차원 벡터 |
| 입력 정보 | 관광지 이름, 주소 1·2, 관광지 설명을 합친 `search_text` |
| 모델 입력 길이 | 최대 1,024 토큰 (초기 본 실험에서 잘린 문서 0개) |
| 검색 순위 | L2 정규화된 벡터 간 내적(코사인 유사도), 내림차순 |
| 평가 기준 | 정답 관광지 `place_id`, R@1·R@5·R@10, MRR |
| 통계 검증 | 관광지 단위 군집 부트스트랩, 대응 순열 검정, Holm 다중검정 보정 |
| 번역 비교 | 원본 / 선별 보완 / 전체 기계번역 / 정보 부족도에 따른 혼합 방식 |

각 검색어는 단일 지정 정답을 가진다. 세 언어 통합 검색 시 동일 관광지의 문서 세 개 중 최고 점수를 사용해 100개 관광지의 순위를 산출한다. 실제 포털을 검색한 것이 아니라 자체적으로 구축한 의미 기반 검색 실험이다.

BGE-M3: [공식 모델](https://huggingface.co/BAAI/bge-m3), [개발 논문](https://arxiv.org/abs/2402.03216). 전체 기계번역 비교 모델: [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (연구·비상업적 비교 실험).

## 2. 폴더 구조

```text
multilingual-search-optimization/
├── README.md
├── requirements.txt
├── src/                        # 관광지 검토, 검색, 분석, 번역 비교 코드
├── data/
│   ├── raw/
│   │   ├── aihub/              # 별도 팀원 데이터 영역
│   │   └── tourapi/            # TourAPI 원본/수집·전처리 코드
│   └── processed/              # 정제 데이터, 평가 검색어, 보완 실험 데이터
├── results/                   # 단계마다 생성된 CSV, JSON, NPY
└── notebooks/                 # 노트북용 폴더
```

VS Code에서 **파일 → 폴더 열기 → multilingual-search-optimization**을 선택하면 연구 코드를 한 프로젝트 안에서 볼 수 있다.

일반화된 관광지 설명 파일 `data/processed/tourapi_standardized.csv`는 팀의 공통 형식인 `source, place_id, language, title, description`을 사용한다. 주소, 좌표, 언어별 원본 ID와 상세정보를 포함한 검색 실험 원본은 `tourapi_pilot_details_300.csv`로 보존한다. 실험에 사용한 보완 데이터는 별도 이름으로 저장하고 원본 설명을 수정하지 않는다.

## 3. 실행 환경

원본 실험은 Python 3.12.16, Apple Silicon의 MPS 가속(float16), BGE-M3 배치 크기 4를 사용했다. 실제 모델과 수치 계산에 사용된 라이브러리 버전은 `requirements.txt`와 각 실험의 메타데이터에 기록되어 있다.

새로운 컴퓨터에서는 Python 3.12를 설치한 후 프로젝트 폴더에서 실행한다.

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

VS Code에서 Python 인터프리터를 해당 프로젝트의 `.venv`로 선택한다. 최초 BGE-M3 및 NLLB 모델 실행 시 모델 파일 다운로드와 추가 저장 공간이 필요할 수 있다.

MPS가 없는 컴퓨터는 새 실험 폴더에서 BGE-M3 기준 실험을 `--device cpu`로 실행할 수 있다. 후속 임베딩 단계는 기준 실험의 장치 및 정밀도를 이어서 사용한다. CPU(float32) 재실행과 원본 MPS(float16) 실행 중 작은 수치 차이로 순위가 달라질 수 있다. 원본 임베딩과 새 임베딩을 섞지 않는다. 새로운 기준 실험에서 특정 BGE-M3 리비전을 고정하려면 `--revision` 옵션을 사용할 수 있다.

## 4. 기존 결과를 재현·확인하는 가장 빠른 방법

기존 연구 수치를 확인할 목적이라면 모델을 다시 다운로드하거나 API를 재호출할 필요가 없다. 기존 데이터와 임베딩을 포함한 프로젝트 폴더에서 다음 명령을 실행한다.

```bash
python src/verify_research.py
```

검증 스크립트는 다음 내용을 계산해 비교하며 어떤 데이터나 결과 파일도 변경하지 않는다.

- 100개 관광지의 언어별 문서 300개와 600개 평가 검색어 정합성
- 원본 문서 및 검색어의 SHA-256과 저장된 실험 기록 일치 여부
- 저장된 BGE-M3 임베딩으로 검색 순위를 재계산했을 때 저장 결과와 일치하는지
- 68개 문서를 선별 보완한 전후의 2,400개 검색 평가 조합 및 주요 성능 지표
- 선별/전체 번역 및 혼합 방식 비교의 저장된 순위와 성능 지표 일치 여부
- 확장 검색 결과가 저장되어 있으면 18,038건 전체의 저장 벡터로 14,400개 평가 순위와 확장 번역 비교 지표도 재검증

정상이라면 원본 실험의 정보 부족 집단 1위 검색 성공률 `45.66% -> 56.62%`와 검증 완료 메시지가 출력된다. 코드 통합 시 별도 복사본에서 기존 원자료의 전처리, 매칭·검토, 캐시 기반 300개 상세정보 생성, 표준화와 질의 생성을 재실행해 기존 CSV와 바이트 단위로 일치함을 확인했다. 전체 모델 추론과 신규 API 요청은 다시 실행하지 않았다.

## 5. 데이터 준비 및 코드 실행 순서

**보존된 원본 결과가 있는 폴더에서 아래 명령을 반복 실행하지 않는다.** 일부 스크립트는 기존 파일을 보호하기 위해 덮어쓰기를 거부하며, 나머지는 실험 결과를 새로 저장한다. 처음부터 실행할 때는 별도 복사한 작업 위치에서 진행한다. 아래의 `python` 명령은 가상환경이 활성화된 프로젝트 루트에서 실행한다.

### 5-1. TourAPI 수집 및 동일 관광지 연결 (원자료부터 재현할 때)

현재 보존된 연구용 파일이 있으면 이 단계의 새로운 API 호출은 필요하지 않다. 실험에 사용한 원자료는 특정 시점의 스냅샷이다. 현재 API로 다시 수집한 자료는 관광지나 설명이 변경되어 원본 실험과 달라질 수 있다.

| 실행 순서 | 코드 | 수행 사항 |
| --- | --- | --- |
| 1 | `data/raw/tourapi/tour_download.py` | TourAPI의 언어별 목록 수집, tour_kr/en/jp CSV·JSON 저장 |
| 2 | `data/raw/tourapi/preprocess_tourapi.py` | 형식 통일, 결측·중복 등 정리, `tourapi_clean.csv` 생성 |
| 3 | `data/raw/tourapi/match_tourapi.py` | 좌표를 기준으로 한·영·일 매칭 후보 342개 생성 |
| 4 | `data/raw/tourapi/verify_tourapi_matches.py` | 이름 자동 매칭 279개와 검토 후보 63개 분리 |
| 5 | `src/review_tourapi.py` | 63개 후보의 이름·주소·좌표·원본 ID를 대조한 보수적 판정 기록 반영 |
| 6 | `src/merge_tourapi_final.py` | 자동 매칭 및 검토 MATCH를 합쳐 331개 관광지 확정 |
| 7 | `src/audit_tourapi_sample.py` | 자동 매칭 중 고정 난수 시드로 선정한 20개 후보의 검토 |
| 8 | `data/raw/tourapi/collect_matched_details.py` | 100곳 선정, 3개 언어별 상세설명 300개 수집 |
| 9 | `src/standardize_tourapi.py` | 팀 공통 5개 컬럼 형식으로 표준화 |

수집 보조 코드: `test_detail_api.py`는 상세 API 응답 테스트, `enrich_tourapi.py`는 초기 상세설명 수집 및 캐시 생성에 사용했다. 최종 100곳은 검토된 후보를 우선 선정하고 기존 수집 캐시와 고정 난수 시드를 사용했다. 동일 표본이 필요하면 반드시 보존된 `tourapi_pilot_100.csv` 및 `tourapi_pilot_details_300.csv`를 사용한다.

5~7번 코드에는 원본 63개 후보 및 20개 표본에 대한 AI 보조 검토 결과가 기록되어 있다. 다른 데이터로 실행하면 검토 내용 및 순번을 새로 대조해야 하며 범용 자동 판정 모델로 해석하지 않는다.

기존 원자료와 캐시로 재구성하는 명령 예시:

```bash
python data/raw/tourapi/preprocess_tourapi.py
python data/raw/tourapi/match_tourapi.py
python data/raw/tourapi/verify_tourapi_matches.py
python src/review_tourapi.py
python src/merge_tourapi_final.py
python src/audit_tourapi_sample.py
python data/raw/tourapi/collect_matched_details.py --prepare
python src/standardize_tourapi.py
```

`--prepare`는 API 재호출 없이 기존 상세정보 캐시를 사용한다. 300개 문서가 모두 COMPLETE가 되어야 이후 분석이 가능하다. 상세설명 누락 시 별도로 수집을 완료한다. `tourapi_manual_review_completed.csv`는 기존 판정 결과를 보존하기 위해 덮어쓰지 않는다.

### 5-2. 초기 검색 및 본 평가 검색어 생성

원본 실험을 확실하게 재현하려면 최소 입력으로 원본 `data/processed/tourapi_pilot_details_300.csv`를 준비한다. 아래 단계는 본 실험의 질의 설계와 초기 소규모 설명 보완을 재현한다.

```bash
# 표준화 CSV가 없는 새 복사본에서 실행
python src/standardize_tourapi.py

# 최종 기준 실험: 360개 초기 검색어, 최대 입력 길이 1,024
python src/run_bge_m3.py --max-length 1024

# 초기 언어별 검색 실패 및 설명 길이 분석 (원래 결과의 탐색적 분석)
python src/analyze_tourapi_gap.py

# 4개 외국어 문서 정보 보완과 36개 새로운 문자열 검색 평가
python src/run_description_enrichment.py
python src/retest_description_enrichment.py

# 15개 관광지 예비 질의에서 시작하여 전체 100개 관광지로 확장
python src/prepare_blind_evaluation.py
python src/fill_realistic_query_draft.py
python src/expand_realistic_queries_600.py
python src/revise_natural_search_queries.py
```

최종 검색어 파일: `data/processed/tourapi_ai_generated_queries_600_natural.csv`. 200개 행에 한국어·영어·일본어 검색어가 각 1개씩 들어 있어 600개 질의를 구성한다. 질의 및 보완 문장은 원본 설명을 바탕으로 AI가 작성한 고정 문구이며, 범용 질의 생성 API 호출 코드는 아니다. 원본 출력에서 확인한 일부 실패 사례를 검토한 초기 단계는 하드웨어가 달라 순위가 변하면 해당 사례 주석을 재확인해야 한다.

### 5-3. 본 실험: 600개 검색어 및 정보 격차 분석

```bash
python src/run_natural_query_evaluation.py
python src/audit_query_feature_coverage.py
python src/analyze_coverage_retrieval.py
python src/run_full_feature_enrichment.py
python src/summarize_feature_gap_reduction.py
python src/statistical_validate_enrichment.py
```

| 코드 | 역할 |
| --- | --- |
| `run_natural_query_evaluation.py` | 600개 검색어의 기준 검색 결과 생성 (초기 4개 문서 보완과도 비교) |
| `audit_query_feature_coverage.py` | 대표·세부 특징의 문서 내 존재 여부를 언어별 대조 |
| `analyze_coverage_retrieval.py` | 필요한 정보가 있는 문서와 부족한 문서의 기준 검색 가시성 비교 |
| `run_full_feature_enrichment.py` | 73개 정보 부족 의도-언어 조합에 해당하는 외국어 문서 68개를 선별 보완 |
| `summarize_feature_gap_reduction.py` | 보완 전후 격차의 변화 요약 |
| `statistical_validate_enrichment.py` | 관광지 단위의 통계 검정 및 신뢰구간 계산 |

### 5-4. 번역 방식 비교 및 혼합 전략 평가

```bash
# 처음 실행 시 NLLB 모델을 다운로드한다. 기존 번역 캐시는 재사용 가능하다.
python src/full_translate_tourapi.py --revision f8d333a098d19b4fd9a8b18f94170487ad3f821d
python src/spotcheck_full_translation.py
python src/compare_enrichment_methods.py
python src/create_quality_checked_translation_sample.py
python src/evaluate_quality_controlled_sample.py
python src/evaluate_hybrid_policy.py

# 최종 생성 결과 검증
python src/verify_research.py
```

| 코드 | 역할 |
| --- | --- |
| `full_translate_tourapi.py` | 한국어 설명을 NLLB로 영·일 번역, 외국어 문서 200개 전체 번역 실험 |
| `spotcheck_full_translation.py` | 원본 실험의 일부 번역 사례를 출처와 비교, 오역 기록 |
| `compare_enrichment_methods.py` | 원본·선별 보완·동일 NLLB 엔진 선별/전체 번역 비교 |
| `create_quality_checked_translation_sample.py` | 20개 외국어 문서에 대해 동일한 AI 출처 대조 번역 문장 준비 |
| `evaluate_quality_controlled_sample.py` | 동일 번역 문장, 동일 문서를 기반으로 선택적/전체 번역 비교 |
| `evaluate_hybrid_policy.py` | MISSING이 있는 문서는 전체 번역, PARTIAL만 있는 문서는 선별 번역하는 탐색적 규칙 평가. 무작위 번역 배분과 비교 |

NLLB 원본 실험의 모델 리비전은 위 명령에 기록된 스냅샷이다. 이미 다운로드한 모델 폴더를 지정하는 경우 `--model-path`와 `--local-files-only` 옵션도 사용할 수 있다. 원본 NLLB의 번역 기록과 일부 번역 오류 검토는 저장된 원번역을 대상으로 작성되었으므로, 다른 번역 결과 생성 시 동일 주석을 그대로 사용하지 않는다. 원본 수치의 재확인은 4절의 저장 임베딩 검증을 권장한다.

## 6. 주요 결과 파일 안내

| 파일 (프로젝트 루트 기준) | 내용 |
| --- | --- |
| `data/processed/tourapi_manual_review_completed.csv` | 63개 매칭 후보별 자동·검토 판정 및 근거 |
| `data/processed/tourapi_final_matches.csv` | 최종 연결된 관광지 331개 |
| `data/processed/tourapi_pilot_100.csv` | 연구 대상 관광지 100개 및 선정 근거 |
| `data/processed/tourapi_pilot_details_300.csv` | 검색 원본 데이터 (상세설명 및 주소 포함) |
| `data/processed/tourapi_standardized.csv` | 공통 형식으로 통일한 3개 언어 데이터 |
| `data/processed/tourapi_ai_generated_queries_600_natural.csv` | 본 실험 검색어와 지정 정답 |
| `results/bge_m3_dense_full/metrics.csv` | 초기 검색 성능 |
| `results/tourapi_gap_analysis/language_description_stats.csv` | 초기 언어별 설명 길이 분석 |
| `results/tourapi_feature_coverage/feature_coverage_summary.csv` | 언어별 관광정보 포함률 |
| `results/tourapi_coverage_retrieval_analysis/coverage_performance_summary.csv` | 정보 포함 여부별 검색 지표 |
| `results/tourapi_full_feature_enrichment/summary.csv` | 68개 문서 보완 전후의 검색 성능 |
| `results/tourapi_full_feature_enrichment/rank_comparison.csv` | 600개 질의 × 4개 검색 범위의 순위 변화 |
| `results/tourapi_full_feature_enrichment/coverage_gap_reduction.csv` | 정보 포함 여부에 따른 검색 가시성 격차 변화 |
| `results/tourapi_statistical_validation/statistical_tests.csv` | 통계적 유의성, p값, 95% 신뢰구간 |
| `results/tourapi_enrichment_method_comparison/method_metrics.csv` | 여러 번역 방식의 검색 성능 |
| `results/tourapi_enrichment_method_comparison/translation_workload.csv` | 결과 번역 토큰 수 기반 작업량 비교 |
| `results/tourapi_quality_controlled_sample/method_performance.csv` | 동일 출처 대조 번역 20개 문서 비교 |
| `results/tourapi_hybrid_policy/policy_performance.csv` | 혼합 보완 전략의 성능 및 번역 토큰 수 |
| `results/tourapi_hybrid_policy/random_allocation_comparison.csv` | 동일 개수의 전체 번역 문서 무작위 배분과 비교 |

CSV는 UTF-8 BOM 형식으로 저장되어 Excel, Numbers, VS Code에서 열 수 있다. `.npy`는 `numpy.load`로 읽는 저장 벡터이며 원본 검색 순위를 재계산하는 데 사용된다. 각 결과 폴더의 JSON에는 실험 설정, 방법 또는 한계가 기록된다.

## 7. 현재 주요 결과와 해석

- 확인한 대표·세부 검색 특징의 전체 포함률은 한국어 99%, 영어 77%, 일본어 84%로 나타났다.
- 연구에서 정의한 정보 부족 집단은 기준 1위 검색 성공률 45.66%로, 정보가 충분한 집단(66.05%)보다 20.38%p 낮았다.
- 부족한 정보를 선별 보완한 219개 검색 평가에서 1위 검색 성공률은 45.66% → 56.62%, MRR은 0.593 → 0.685로 증가했다. 1위 지표의 Holm 보정 p값은 0.0002였다.
- 전체 다국어 통합 검색에서는 기존 83.33%, AI 선별 보완 84.17%, NLLB 전체 번역(68개 문서) 81.83%, NLLB 전체 번역(200개 문서) 78.67%였다. 전체 대상 선별 보완의 개선은 다중검정 보정 후 유의하지 않았다.
- 출처 대조 번역 20개 문서 표본에서 혼합 방식은 전체 번역 대비 약 52% 적은 토큰으로 정보 부족 집단의 1위 성공률 40.58% → 60.87%를 기록했다. 이는 탐색적 결과이며 혼합 방식이 항상 우수하다는 뜻은 아니다.

**연구 결론:** 정보 부족에 따라 검색 가시성 격차가 관찰되었으며, 이 연구에서는 부족한 정보의 보완 효과를 확인했다. 번역량과 검색 성능을 함께 고려하는 적응형 보완 전략이 추가 검증할 만한 방향이다.

### 해석상 한계

- 관광지 100개는 전체 관광지에서 무작위로 추출한 대표 표본이 아니다.
- 질의 작성과 정보 부족 확인·보완에 한국어 설명 및 같은 검색 의도가 사용되었다. 독립적인 사용자 검색어 검증이 필요하다.
- 품질 대조와 매칭 검토는 제공된 출처에 근거한 AI 보조 검토로, 독립적인 전문가 확인이나 실제 여행지 현장 검증은 아니다.
- 번역 토큰 수는 작업량의 대리 지표이며 실제 비용이나 인력 투입 시간을 직접 측정한 값이 아니다.
- BGE-M3 자체의 언어별 검색 성능 차이를 완전히 제거할 수 없으며 실제 검색 엔진에서 같은 순위가 나타남을 보장하지 않는다.
- 원본 MPS와 CPU 환경의 결과가 완벽히 같다고 가정하지 않는다. 원본 BGE-M3 메타데이터에는 정확한 모델 리비전이 기록되지 않아 저장된 임베딩과 스냅샷을 함께 보관하는 것이 중요하다.

## 8. 제출 및 버전 관리

대회 제출 시 `README.md`, `requirements.txt`, `src/` 전체와 `data/raw/tourapi/`의 Python 수집·전처리 코드 7개를 포함한다. 데이터 및 결과의 제출이 허용된다면 `data/processed/`와 `results/`를 원본 실험 스냅샷으로 함께 제공하는 것이 재현 검증에 유리하다. 원자료까지 재현하려면 TourAPI 원본 CSV·JSON 및 상세설명 캐시도 필요하다. 가상환경 `.venv/`, 모델 캐시, Python 캐시 `__pycache__/`는 제출할 필요가 없다.

Git에는 분석·검토 코드와 오프라인 전처리 코드, 환경 설정 및 실행 안내를 관리한다. 로컬에만 있는 API 호출 코드 및 데이터는 별도 제출 범위에 따라 관리한다. 기존 결과는 자동으로 원격 저장소에 올리지 않는다.

## 9. 검색 후보 확대 실험: 전체 18,038건 검색, 100곳 참조 평가셋

초기 연구의 100개 관광지를 그대로 지정 정답을 가진 평가셋으로 사용하면서 검색 후보를 전체 정제 원본으로 확장한다. 18,038건은 **언어별 원본 문서 수**이며 서로 다른 관광지 18,038곳이라는 뜻은 아니다. 600개 검색어도 그대로 유지하여 검색 후보 증가 및 정보 보완 전후의 변화를 비교한다.

| 구분 | 예비 실험 | 확장 실험 |
| --- | ---: | ---: |
| 평가용 관광지 | 100곳 | 100곳 |
| 평가용 검색어 | 600개 | 600개 |
| 한국어 검색 후보 | 100건 | 12,581건 |
| 영어 검색 후보 | 100건 | 2,822건 |
| 일본어 검색 후보 | 100건 | 2,635건 |
| 전체 검색 후보 문서 | 300건 | 18,038건 |
| 통합 검색 후보 단위 | 100곳 | 17,378개 그룹 |

통합 검색에서는 이전에 연결한 330개의 한·영·일 관광지 묶음으로만 중복을 통합한다. 추가 검토가 필요했던 1개 묶음과 연결되지 않은 레코드는 별개 후보로 남긴다. 따라서 17,378개는 보수적으로 정의한 검색 후보 그룹 수이지, 검증된 고유 관광지 수가 아니다.

### 9-1. 실험 조건

전체 원본 문서의 공통 필드는 이름과 주소이며, 이름은 18,038건 모두, 주소는 18,022건에 있다. 상세설명을 실제 확보한 문서는 363건(2.01%)에 불과하다. 363건 중 300건은 평가셋의 문서이고, 나머지 63건은 기존 한국어 설명 캐시이다. 상세설명이 없는 문서에 임의의 내용을 생성하거나 새로 API를 요청하지 않는다.

| 조건 | 입력 | 목적 |
| --- | --- | --- |
| METADATA | 모든 문서의 이름·주소 | 후보 수만 300건 → 18,038건으로 늘렸을 때의 검색 난도 비교 |
| AVAILABLE_ORIGINAL | 이름·주소와 실제 확보된 기존 상세설명 363건 | 현재 확보한 정보의 검색 성능 |
| AVAILABLE_ENRICHED | AVAILABLE_ORIGINAL에서 기존 68개 평가 문서만 선별 보완 | 동일한 확장 후보군에서 기존 보완 효과 재확인 |

각 조건에서 예비·확장 후보군을 모두 평가한다. 총 10,822,800개의 검색어-문서 조합(600 × 18,038)이 확장 조건당 존재하며, 문서 벡터 및 중간 점수를 재사용한다. 서로 다른 검색어-문서 조합이나 반복 언어 평가를 독립 표본 수로 해석하지 않는다.

가용 상세설명을 반영한 조건은 평가셋 문서가 비평가셋 문서보다 훨씬 풍부하므로 절대 검색 성능이 낙관적일 수 있다. 300건과 18,038건의 검색 난도를 공정하게 비교하는 주요 조건은 METADATA이다. 설명 보완의 전후 비교는 동일한 18,038건의 후보군과 동일한 나머지 문서에서 평가한다. 언어별 검색 후보 수가 다르므로 단순한 언어별 순위 차이를 전부 번역 품질 차이로 해석하지 않는다.

### 9-2. 실행 방법

기존 5절 실험의 데이터와 저장된 질의·문서 벡터가 필요하다. 다음 명령은 신규 API 요청 없이 캐시된 BGE-M3 모델과 기존 설명을 사용한다. 이전 MPS 실험 벡터와 새 벡터를 섞는 오류를 방지하기 위해 샘플 벡터의 일치도를 확인하며, 장치 및 정밀도가 동일한 환경에서 실행한다.

```bash
python src/prepare_large_corpus.py --batch-size 64 --chunk-size 512
python src/evaluate_large_corpus.py
python src/evaluate_large_corpus_methods.py
```

첫 번째 명령은 18,038건의 메타데이터를 임베딩하고 기존 비평가셋 상세설명 63건도 임베딩한다. 진행한 벡터를 중간 파일에 저장하여 같은 데이터와 모델로 중단 후 재개할 수 있다. 두 번째 명령은 3개 입력 조건 × 2개 후보 범위 × 600개 질의 × 4개 검색 범위의 총 14,400개 평가 행과 지표, 대응 통계 분석을 생성한다. 세 번째 명령은 기존 선별 보완·전체 번역·혼합 보완 전략의 저장 벡터를 동일한 확장 검색 후보군에서 재평가한다. 모든 확장 결과는 `results/tourapi_large_corpus_eval/`에 별도 저장하므로 기존 연구 결과를 덮어쓰지 않는다.

### 9-3. 결과 파일

| 파일 | 내용 |
| --- | --- |
| `corpus_coverage.csv` | 18,038건의 언어별 자료 수, 주소·상세설명 보유 현황 |
| `corpus_index.csv` | 원본 레코드 ID, 연결 그룹, 평가셋 ID와 검색용 원본 텍스트 |
| `metrics.csv` | 입력 조건·후보 범위별 R@1/5/10/20/50, MRR, 평균·중앙 순위 |
| `language_metrics.csv` | 질의 언어와 문서 언어별 검색 지표 |
| `paired_summary.csv` | 후보 확대 및 정보 보완 전후 비교 |
| `rank_results.csv` | 개별 검색어의 정답 순위와 1위 문서 |
| `coverage_gap_comparison.csv` | 정보 충분 집단과 부족 집단의 검색 성공률 격차 |
| `statistical_tests.csv` | 확장 검색 조건의 보완 전후 대응 검정 |
| `method_strategy_metrics.csv` | 기존 번역 방식들의 확장 풀에서의 검색 성능과 번역 토큰 작업량 |
| `method_strategy_tests.csv` | 동일 엔진·동일 범위 번역 방식 간 탐색적 검정 |
| `experiment_metadata.json` | 확장 실험 설정, 원본 입력 해시 및 한계 |

본 확장 실험도 기존 평가셋에 대한 재검증이다. 새로운 독립 사용자 질의나 독립적인 인간 관련성 판단을 확보한 연구로 주장해서는 안 된다.

### 9-4. 실제 확장 실험 결과

18,038건 전부의 메타데이터 임베딩을 생성했다. BGE-M3의 기존 저장 벡터와 새로 생성한 표본 벡터의 최소 코사인 일치도는 0.99999803이었다. 메타데이터의 최대 입력 길이는 93토큰이었으며 잘린 문서는 없었다. 이전 예비 실험의 정답 순위 2,400개를 각 보완 조건에서 그대로 재현했다.

**세 언어 통합 검색의 1위 성공률(R@1), 검색어 600개:**

| 입력 조건 | 예비 실험: 300건 | 확장 실험: 18,038건 |
| --- | ---: | ---: |
| 이름·주소만 (METADATA) | 62.83% | 25.50% |
| 기존 가용 설명 반영 | 83.33% | 54.67% |
| 기존 설명 + 선별 보완 | 84.17% | 56.17% |

모든 후보의 정보 필드를 동일하게 제한한 메타데이터 조건에서 검색 후보가 확대될수록 검색 난도가 뚜렷하게 증가했다. 나머지 두 조건은 평가셋의 설명 보유율이 비평가셋보다 높으므로 절대 성공률 해석에 주의한다.

**전체 후보군에서 선별 정보 보완 전후 비교:**

| 평가 집단 | 보완 전 R@1 | 보완 후 R@1 | 변화 |
| --- | ---: | ---: | ---: |
| 전체 통합 검색, 600회 | 54.67% | 56.17% | +1.50%p |
| 정보 부족 집단, 219회 | 24.66% | 33.33% | +8.68%p |
| 교차언어 정보 부족 집단, 146회 | 13.70% | 21.23% | +7.53%p |

정보 부족 집단의 MRR은 0.315738 → 0.411499, 중앙 순위는 21위 → 7위로 변화했다. 정보 충분 집단과 부족 집단 간 R@1 격차는 14.37%p → 5.80%p로 감소했다. 정보 부족 집단 R@1의 95% 군집 부트스트랩 신뢰구간은 +4.23~+13.93%p이며 Holm 보정 p값은 0.00244이다. 이 결과는 고정 평가셋의 탐색적 통계 검증이다.

**전체 후보군에서 번역 방식 비교 (동일 NLLB 엔진, 동일 68개 보완 범위):**

| 방식 | 통합 검색 R@1 | 부족 집단 R@1 | 번역 출력 토큰 |
| --- | ---: | ---: | ---: |
| 원본 | 54.67% | 24.66% | 0 |
| 선별 번역 | 55.33% | 30.59% | 3,457 |
| 전체 번역 | 51.67% | 33.33% | 17,484 |
| 혼합 보완 | 54.33% | 31.51% | 7,003 |

동일 68개 문서에 대해 NLLB 선별 번역과 혼합 보완은 전체 번역보다 통합 검색 R@1이 높았으며, 보정 p값은 각각 0.00624와 0.02686이었다. 정보 부족 집단에서 방식 간 차이는 보정 후 유의하지 않았다. 별도의 출처 대조 20개 문서 표본에서도 혼합 방식은 전체 번역 대비 약 52% 적은 토큰으로 정보 부족 집단 R@1 44.93%를 기록했다(전체 번역 47.83%, 원본 24.64%). 이 차이는 통계적 동등성의 근거가 아니다.

확장 실험은 후보군 규모를 늘려도 선별 보완의 개선 효과가 유지될 수 있음을 보여준다. 다음 단계는 비평가셋의 실제 설명을 더 확보하여 정보 보유율의 불균형을 줄이고, 독립적인 평가 질의 및 신규 관광지로 외적 타당성을 검사하는 것이다.

## 10. 제한된 번역 작업량에 따른 보완 우선순위 실험

**연구 질문:** 동일한 번역 작업량으로 어떤 관광정보부터 보완해야 검색 가시성 개선 효과가 큰가?

기존 18,038건의 가용정보 검색 후보군, 평가 관광지 100곳, 검색어 600개를 사용한다. 보완 후보는 기존 NLLB 선별 번역이 준비된 외국어 문서 68개이며, 모든 전략에서 동일한 번역 문장과 임베딩을 사용한다. 각 문서의 추가 번역 출력 토큰을 작업량의 대리 지표로 사용한다(총 3,457토큰). 문서별 실제 인력·시간·가격은 측정하지 않았다.

### 10-1. 동일 작업량 제약과 세 전략

| 전략 | 문서 선정 방법 |
| --- | --- |
| A. 무작위 | 각 토큰 예산과 정확히 일치하는 가능한 문서 부분집합에서 균등 무작위 추출, 예산마다 500회 |
| B. 정보 부족도 | 문서별 누락 의도 수 × 2 + 부분 부족 의도 수로 중요도를 부여하고, 동일 토큰 제약에서 중요도 합을 최대화 |
| C. 제안 전략 | 한 문서씩 보완했을 때 기존 개발 검색어에서 정보 부족 집단의 MRR이 얼마나 변하는지 측정하고, 동일 토큰 제약에서 예상 개선량 합을 최대화 |

정수 토큰 예산은 691, 1,383, 2,074, 2,766(약 20%, 40%, 60%, 80%)이다. 각 예산에서 세 방식의 실제 선택 토큰 총량이 **정확하게 같도록** 설정했다. 0토큰(원본)과 3,457토큰(68개 모두 보완)도 비교 기준으로 평가한다.

전략 B와 C는 모든 문서를 대상으로 선택 여부를 나타내는 이진 변수 x_i를 정의하고, 합(c_i x_i) = B라는 토큰 예산 제약 아래 합(v_i x_i)를 최대화하는 0-1 배낭 최적화(0–1 knapsack)를 사용한다. v_i는 B에서는 정보 부족도, C에서는 해당 문서만 교체한 개발 검색의 MRR 증분이다. C의 개별 예측 개선량은 실제 공동 순위 효과의 근사치이며, 여러 문서를 동시에 변경할 때 동일한 효과가 단순 합산된다고 가정할 수는 없다. 따라서 선택 완료 후 전체 후보군에서 실제 검색 순위를 다시 산출한다.

### 10-2. 개발 실험 결과

정보 부족 집단의 기존 검색어 평가 219회에 대한 1위 성공률 및 MRR:

| 번역 작업량 | A 무작위 평균 R@1 / MRR | B 부족도 R@1 / MRR | C 개선 기대량 R@1 / MRR |
| --- | --- | --- | --- |
| 20% (691토큰) | 25.97% / 0.3324 | 27.40% / 0.3486 | 29.22% / 0.3731 |
| 40% (1,383토큰) | 27.05% / 0.3447 | 29.22% / 0.3673 | 30.59% / 0.3884 |
| 60% (2,074토큰) | 28.26% / 0.3577 | 29.22% / 0.3679 | 30.59% / 0.3903 |
| 80% (2,766토큰) | 29.30% / 0.3696 | 30.14% / 0.3770 | 30.59% / 0.3903 |

원본은 R@1 24.66%, MRR 0.3157이고, 68개 전체 선별 보완은 R@1 30.59%, MRR 0.3859이다. 개발 검색어에서는 제안 전략이 40% 토큰 예산으로 29개 문서를 선정하여 전체 68개 선별 보완과 동일한 R@1을 기록했다. 그러나 R@10은 전체 보완보다 낮았고, C의 문서 선정에 같은 검색어가 사용되었으므로 **독립 검증을 통한 효율성 입증이나 성능 동등성·실제 비용 절감 주장으로 해석하지 않는다.** 무작위 결과의 분포도 개발 데이터 내 비교 기준이지 독립적 가설 검정의 p값이 아니다.

### 10-3. 신규 검색어를 활용한 분리 평가 (작성 대기 중)

개발 데이터에서 산출한 우선순위와 각 예산별 문서 선택 목록을 `locked_allocations.json`에 고정했다. 아직 독립 검색어의 성능을 측정하지 않았다. 팀원은 기존 검색어, 보완 문장, 순위 및 전략별 결과를 보지 않은 상태에서 `holdout_query_template.csv`에 제시된 30개 관광지의 새로운 검색 의도를 작성한다.

- 관광지별로 이름을 그대로 복사하는 질문보다는 실제 사용자가 입력할 법한 짧은 특징 중심 검색어를 작성한다.
- 한국어·영어·일본어의 검색 의도가 일치하도록 하고 자연스러운 표현을 사용한다. AI 번역 사용 여부와 사람의 검수 여부를 `author_note`에 기록한다.
- 기존 검색어를 복사하지 않는다. `holdout_intent_id`와 `place_id`는 변경하지 않는다.
- 30개 관광지 중 20개는 보완 후보가 있는 관광지, 10개는 그 외 관광지에서 고정 시드로 선정했다. 작성용 양식에는 이 분류와 순위·보완 정보를 노출하지 않았다.
- 완료한 양식을 별도 복사본 `holdout_query_filled.csv`로 저장한다. 90개 신규 검색어로 같은 후보군과 고정 선택 목록을 평가한다. 새로운 의도·검색어 검증이며 새로운 관광지에 대한 검증은 아니다. 해당 신규 의도의 실제 누락 여부를 별도 검토하지 않는 한 성능을 정보 부족 집단의 성능으로 부르지 않는다.

### 10-4. 실행 및 결과

```bash
# 이미 개발 실험이 완료되었으므로 저장된 배분을 변경하지 않고 검증
python src/evaluate_budget_priority.py --verify-plan

# 팀원이 작성한 새로운 검색어가 있을 때만 실행 (경로는 실제 파일 위치로 변경)
python src/evaluate_budget_priority.py --holdout-file results/tourapi_budget_priority/holdout_query_filled.csv
```

개발 결과 및 산출물은 `results/tourapi_budget_priority/`에 저장했다.

| 파일 | 의미 |
| --- | --- |
| `budget_summary.csv` | 세 전략의 예산별 검색 성능 |
| `document_priority_scores.csv` | 각 문서의 보완 비용, 부족도, 개발 검색에서의 실제 개별 MRR 증분 |
| `allocation_documents.csv` | 제안·부족도 전략이 각 예산에서 선택한 문서 목록 |
| `random_trials.csv` | 4개 예산 × 500회 정확한 예산의 무작위 배분 결과와 검색 지표 |
| `deterministic_rank_results.csv` | 원본·전체·전략 B/C의 개별 질의 정답 순위 |
| `locked_allocations.json` | 고정 배분 목록, 입력 해시, 시드 및 선정 기준 |
| `holdout_query_template.csv` | 팀원에게 전달할 30개 관광지, 신규 검색어 90개 작성 양식 |

저장된 원본·전체 선별 보완과 동일한 검색 순위가 나오는지 확인하고, 최적화된 부분집합 순위 계산도 전체 점수 행렬로 직접 계산한 결과와 대조했다. 신규 검증을 수행하기 전까지 현재 결과는 개발 분석으로 한정한다.
