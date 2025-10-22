# 텍스트 마이닝 기반 기업-토픽 연관도 스코어링 시스템

## 🎯 프로젝트 개요

국내 상장기업 전체를 대상으로 한 6개월 리밸런싱 주기의 텍스트 마이닝 기반 연관도 스코어링 시스템입니다.

### 주요 기능
- **데이터 수집**: DeepSearch API를 통한 국내 상장기업 정보 및 리포트 수집
- **PDF 텍스트 추출**: 고품질 PDF 텍스트 추출 (PyPDF2, pdfplumber, OCR)
- **텍스트 전처리**: 한국어 형태소 분석, 불용어 제거, 키워드 추출
- **BM25 스코어링**: 키워드-문서 간 연관도 점수 계산
- **NMF 토픽 모델링**: Non-negative Matrix Factorization을 통한 토픽 추출
- **통합 스코어링**: BM25와 토픽 모델링 결과를 가중 결합
- **자동 리밸런싱**: 6개월 주기 자동 스코어링 및 리밸런싱
- **Excel 출력**: 결과를 Excel 파일로 저장 (여러 시트 포함)

## 📁 프로젝트 구조

```
theme_index/
├── main.py                 # 메인 실행 스크립트
├── example.py             # 테스트 및 예제 스크립트
├── test_simple.py         # 간단한 테스트 스크립트
├── test_excel.py          # Excel 기능 테스트 스크립트
├── requirements.txt       # 의존성 패키지
├── config.py             # 시스템 설정 파일
├── README.md             # 프로젝트 설명서
├── src/                  # 소스 코드
│   ├── api_client.py     # DeepSearch API 클라이언트
│   ├── pdf_processor.py  # PDF 텍스트 추출
│   ├── text_preprocessor.py # 텍스트 전처리
│   ├── bm25_scorer.py   # BM25 스코어링
│   ├── topic_modeler.py # NMF 토픽 모델링
│   ├── scoring_system.py # 통합 스코어링
│   ├── rebalancing_scheduler.py # 리밸런싱 스케줄러
│   └── period_execution_manager.py # 시점별 실행 관리
├── data/                 # 데이터 저장소
│   ├── stopwords.txt     # 불용어 리스트
│   └── keyword_patterns.txt # 키워드 패턴 매핑
├── logs/                 # 로그 파일
└── results/             # 결과 파일 (자동 생성)
```

## 🚀 성능 최적화

### Numba JIT 컴파일
- **BM25 스코어 계산**: 3-5배 속도 향상
- **TF-IDF 계산**: 2-3배 속도 향상  
- **행렬 정규화**: 2-4배 속도 향상

### 병렬 처리 (Joblib)
- **PDF 텍스트 추출**: CPU 코어 수만큼 병렬 처리
- **텍스트 전처리**: 배치 단위 병렬 처리
- **키워드 스코어 계산**: 키워드별 병렬 처리

### 예상 성능 향상
- **전체 실행 시간**: 50-70% 단축
- **메모리 사용량**: 최적화된 배치 처리로 효율성 증대
- **CPU 활용률**: 멀티코어 활용으로 처리량 증가

### 성능 최적화 활성화/비활성화
```python
# 최적화 활성화 (기본값)
bm25_scorer = BM25Scorer(use_optimization=True)
execution_manager = PeriodExecutionManager(..., use_optimization=True)

# 최적화 비활성화 (호환성 문제 시)
bm25_scorer = BM25Scorer(use_optimization=False)
execution_manager = PeriodExecutionManager(..., use_optimization=False)
```

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 실행 방법

#### 기본 실행
```bash
python main.py
```

#### 테스트 실행
```bash
# 텍스트 전처리 테스트
python example.py --test text

# 스코어링 시스템 테스트
python example.py --test scoring

# 전체 파이프라인 테스트
python example.py --test full

# 시점별 실행 테스트
python example.py --test period
```

#### 리밸런싱 스케줄러 실행
```bash
python example.py --run-scheduler
```

#### 시점별 실행 (2021-12-01 ~ 2024-12-01)
```bash
python example.py --run-period
```

#### Python 코드로 직접 실행
```python
from main import run_period_execution

# 시점별 실행
execution_manager = run_period_execution(
    execution_name="historical_analysis",
    start_date="2021-12-01",
    end_date="2024-12-01",
    rebalancing_months=6
)
```

## 📊 출력 결과

### 시점별 실행 결과 구조 (정확한 6개월 구간)

시점별 실행 시 다음과 같은 폴더 구조가 생성됩니다. **각 시점은 정확히 6개월 구간**으로 월말에 종료되어 더 정확하고 일관성 있는 기간 구분을 제공합니다:

```
results/
└── execution_historical_analysis_20200601_20250531/
    ├── period_20200601_20201130/          # 시점 1: 2020-06-01 ~ 2020-11-30 (정확한 6개월)
    │   ├── scoring_results_20241222_143022.json
    │   ├── scoring_results_companies_20241222_143022.csv
    │   ├── scoring_results_rankings_20241222_143022.csv
    │   └── scoring_results_20241222_143022.xlsx
    ├── period_20201201_20210531/          # 시점 2: 2020-12-01 ~ 2021-05-31 (정확한 6개월)
    │   ├── scoring_results_20241222_143045.json
    │   ├── scoring_results_companies_20241222_143045.csv
    │   ├── scoring_results_rankings_20241222_143045.csv
    │   └── scoring_results_20241222_143045.xlsx
    ├── period_20210601_20211130/          # 시점 3: 2021-06-01 ~ 2021-11-30 (정확한 6개월)
    │   └── ...
    ├── period_20211201_20220531/          # 시점 4: 2021-12-01 ~ 2022-05-31 (정확한 6개월)
    │   └── ...
    ├── period_20220601_20221130/          # 시점 5: 2022-06-01 ~ 2022-11-30 (정확한 6개월)
    │   └── ...
    ├── period_20221201_20230531/          # 시점 6: 2022-12-01 ~ 2023-05-31 (정확한 6개월)
    │   └── ...
    ├── period_20230601_20231130/          # 시점 7: 2023-06-01 ~ 2023-11-30 (정확한 6개월)
    │   └── ...
    ├── period_20231201_20240531/          # 시점 8: 2023-12-01 ~ 2024-05-31 (정확한 6개월)
    │   └── ...
    ├── period_20240601_20241130/          # 시점 9: 2024-06-01 ~ 2024-11-30 (정확한 6개월)
    │   └── ...
    ├── period_20241201_20250531/          # 시점 10: 2024-12-01 ~ 2025-05-31 (정확한 6개월)
    │   └── ...
    ├── consolidated_results.xlsx          # 통합 Excel 파일
    └── execution_summary.json             # 실행 요약
```

### 정확한 6개월 구간 생성의 장점

1. **정확한 기간 구분**: 각 시점이 정확히 6개월로 일관성 있음
2. **균등한 길이**: 모든 시점이 182-183일로 거의 동일한 길이
3. **분석 용이성**: 반기별 분석에 최적화된 기간 구분
4. **비즈니스 친화적**: 반기 보고 기간과 정확히 일치
5. **월말 종료**: 각 시점이 월말에 종료되어 직관적

### 통합 Excel 파일 구조 (`consolidated_results.xlsx`)

1. **시점별 기업 스코어 시트들**: 각 시점별로 하나의 시트
   - `20200601-20201130`: 첫 번째 시점의 기업별 키워드 스코어 (2020년 하반기)
   - `20201201-20210531`: 두 번째 시점의 기업별 키워드 스코어 (2020년 하반기~2021년 상반기)
   - `20210601-20211130`: 세 번째 시점의 기업별 키워드 스코어 (2021년 하반기)
   - `20211201-20220531`: 네 번째 시점의 기업별 키워드 스코어 (2021년 하반기~2022년 상반기)
   - `20220601-20221130`: 다섯 번째 시점의 기업별 키워드 스코어 (2022년 하반기)
   - `20221201-20230531`: 여섯 번째 시점의 기업별 키워드 스코어 (2022년 하반기~2023년 상반기)
   - `20230601-20231130`: 일곱 번째 시점의 기업별 키워드 스코어 (2023년 하반기)
   - `20231201-20240531`: 여덟 번째 시점의 기업별 키워드 스코어 (2023년 하반기~2024년 상반기)
   - `20240601-20241130`: 아홉 번째 시점의 기업별 키워드 스코어 (2024년 하반기)
   - `20241201-20250531`: 열 번째 시점의 기업별 키워드 스코어 (2024년 하반기~2025년 상반기)

2. **Execution_Summary 시트**: 실행 요약 정보
   ```
   period_name | start_date | end_date | total_companies | total_documents | execution_time
   20200601-20201130 | 2020-06-01 | 2020-11-30 | 2,500 | 15,000 | 2024-12-22T14:30:22
   20201201-20210531 | 2020-12-01 | 2021-05-31 | 2,500 | 18,000 | 2024-12-22T14:30:45
   20210601-20211130 | 2021-06-01 | 2021-11-30 | 2,500 | 16,000 | 2024-12-22T14:31:10
   ```

3. **키워드별 시점간 비교 시트들**: 각 키워드별로 하나의 시트
   - `Keyword_반도체`: 반도체 키워드의 시점별 상위 기업 변화
   - `Keyword_AI`: AI 키워드의 시점별 상위 기업 변화
   - `Keyword_배터리`: 배터리 키워드의 시점별 상위 기업 변화
   - ...

4. **Company_Trends 시트**: 기업별 시점간 스코어 변화
   ```
   company_symbol | 20200601-20201130_avg_score | 20200601-20201130_반도체 | 20200601-20201130_AI | ...
   005930        | 0.65                        | 0.85                     | 0.72                  | ...
   373220        | 0.58                        | 0.12                     | 0.18                  | ...
   ```

### 개별 시점 Excel 파일 구조 (`scoring_results_YYYYMMDD_HHMMSS.xlsx`)

1. **Company_Scores 시트**: 기업별 키워드 스코어
   ```
   company_symbol | 반도체 | AI | 배터리 | 전기차 | ...
   005930        | 0.85   | 0.72| 0.15   | 0.23   | ...
   373220        | 0.12   | 0.18| 0.92   | 0.88   | ...
   ```

2. **Keyword_Rankings 시트**: 키워드별 기업 순위
   ```
   keyword | rank | company_symbol | score
   반도체   | 1    | 005930        | 0.85
   반도체   | 2    | 000660        | 0.78
   ```

3. **Rebalancing_Recommendations 시트**: 리밸런싱 추천 기업
   ```
   company_symbol | average_score | max_score | keyword_count
   005930        | 0.65         | 0.85      | 8
   373220        | 0.58         | 0.92      | 6
   ```

4. **Metadata 시트**: 스코어링 메타데이터
   ```
   항목           | 값
   스코어링 날짜   | 2024-10-22T16:40:00
   데이터 기간     | 6개월
   총 기업 수     | 2,500
   총 문서 수     | 15,000
   ```

5. **Target_Keywords 시트**: 분석 대상 키워드
   ```
   keyword | rank
   반도체   | 1
   AI      | 2
   배터리   | 3
   ```

### CSV 파일
- `scoring_results_companies_YYYYMMDD_HHMMSS.csv`: 기업별 스코어
- `scoring_results_rankings_YYYYMMDD_HHMMSS.csv`: 키워드별 순위
- `scoring_results_YYYYMMDD_HHMMSS.json`: 전체 결과 (JSON)

## ⚙️ 설정 옵션

### 시스템 설정 (`main.py`의 `SystemConfig`)
```python
@dataclass
class SystemConfig:
    api_key: str = "ZGVlcHNlYXJjaDoyMWNkNTExODVkN2RlNjFjMGY0Yg=="
    rebalancing_months: int = 6
    target_keywords: List[str] = [
        "반도체", "AI", "인공지능", "배터리", "전기차", "EV", 
        "신재생에너지", "바이오", "의료", "핀테크", "금융", 
        "클라우드", "5G", "IoT", "블록체인", "메타버스"
    ]
    bm25_weight: float = 0.6
    topic_weight: float = 0.4
    min_documents_per_company: int = 5
```

### BM25 파라미터
- `k1`: BM25 파라미터 (기본값: 1.2)
- `b`: BM25 파라미터 (기본값: 0.75)

### NMF 파라미터
- `n_topics`: 토픽 수 (기본값: 20)
- `max_features`: 최대 특성 수 (기본값: 1000)
- `min_df`: 최소 문서 빈도 (기본값: 2)
- `max_df`: 최대 문서 빈도 (기본값: 0.95)

## 🔧 주요 컴포넌트

### 1. DeepSearch API 클라이언트 (`src/api_client.py`)
- 국내 상장기업 전체 목록 조회
- 기업별 리포트 및 문서 링크 수집
- Excel 및 CSV 형태로 데이터 저장

### 2. PDF 텍스트 추출 (`src/pdf_processor.py`)
- 다중 방법 텍스트 추출 (PyPDF2, pdfplumber, OCR)
- 텍스트 품질 신뢰도 평가
- 배치 처리 지원

### 3. 텍스트 전처리 (`src/text_preprocessor.py`)
- 한국어 형태소 분석 (KoNLPy)
- 불용어 제거 및 토큰화
- 키워드 패턴 매칭

### 4. BM25 스코어링 (`src/bm25_scorer.py`)
- BM25 알고리즘 구현
- 키워드-문서 연관도 점수 계산
- 기업별 스코어 집계

### 5. NMF 토픽 모델링 (`src/topic_modeler.py`)
- TF-IDF 행렬 구축
- NMF를 통한 토픽 추출
- 기업별 토픽 분포 계산

### 6. 통합 스코어링 (`src/scoring_system.py`)
- BM25와 토픽 모델링 결과 가중 결합
- 최종 기업-토픽 연관도 점수 산출
- Excel 및 CSV 형태로 결과 저장

### 7. 리밸런싱 스케줄러 (`src/rebalancing_scheduler.py`)
- 6개월 주기 자동 실행
- 전체 파이프라인 자동화
- 결과 저장 및 히스토리 관리

### 8. 시점별 실행 관리 (`src/period_execution_manager.py`)
- 여러 시점에 걸친 실행 관리
- 시점별 폴더 구조 자동 생성
- 통합 Excel 파일 생성 (시점별 시트 포함)
- 키워드별 시점간 비교 분석

## 📈 사용 시나리오

### 시나리오 1: 일회성 분석
```bash
# 전체 파이프라인 실행
python example.py --test full
```

### 시나리오 2: 지속적 모니터링
```bash
# 리밸런싱 스케줄러 실행
python example.py --run-scheduler
```

### 시나리오 3: 특정 컴포넌트 테스트
```bash
# 텍스트 전처리만 테스트
python example.py --test text

# 스코어링 시스템만 테스트
python example.py --test scoring
```

### 시나리오 4: 시점별 과거 데이터 분석
```bash
# 2021-12-01부터 2024-12-01까지 시점별 분석
python example.py --run-period

# 또는 Python 코드로 직접 실행
from main import run_period_execution
execution_manager = run_period_execution(
    execution_name="historical_analysis",
    start_date="2021-12-01",
    end_date="2024-12-01",
    rebalancing_months=6
)
```

## ⚠️ 주의사항

1. **API 제한**: DeepSearch API 호출 제한을 고려하여 적절한 지연 시간 설정
2. **메모리 사용량**: 대규모 데이터 처리 시 메모리 사용량 모니터링 필요
3. **PDF 품질**: 일부 PDF는 텍스트 추출이 어려울 수 있음
4. **한국어 처리**: KoNLPy 설치 시 Java 환경 필요
5. **Excel 파일**: openpyxl 패키지 필요

## 🐛 문제 해결

### 일반적인 오류
1. **API 연결 실패**: 네트워크 연결 및 API 키 확인
2. **PDF 추출 실패**: PDF 파일 접근 권한 및 형식 확인
3. **메모리 부족**: 배치 크기 조정 또는 서버 리소스 증설
4. **한국어 처리 오류**: KoNLPy 및 Java 환경 재설치
5. **Excel 저장 실패**: openpyxl 패키지 재설치

### 성능 최적화
1. **병렬 처리**: PDF 추출 및 텍스트 전처리 병렬화
2. **캐싱**: 중간 결과 캐싱으로 재계산 방지
3. **배치 처리**: 대용량 데이터 배치 단위 처리

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

프로젝트 개선을 위한 기여를 환영합니다. 이슈 리포트나 풀 리퀘스트를 통해 참여해주세요.