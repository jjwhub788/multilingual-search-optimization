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
