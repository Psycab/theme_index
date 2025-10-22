# 시스템 설정 파일
# 텍스트 마이닝 기반 기업-토픽 연관도 스코어링 시스템

# API 설정
API_KEY = "ZGVlcHNlYXJjaDoyMWNkNTExODVkN2RlNjFjMGY0Yg=="
BASE_URL = "https://api-v2.deepsearch.com"
COUNTRY_CODE = "kr"
PAGE_SIZE = 100

# 리밸런싱 설정
REBALANCING_MONTHS = 6
BM25_WEIGHT = 0.6
TOPIC_WEIGHT = 0.4
MIN_DOCUMENTS_PER_COMPANY = 5

# 타겟 키워드 설정
TARGET_KEYWORDS = [
    "반도체", "AI", "인공지능", "배터리", "전기차", "EV", 
    "신재생에너지", "바이오", "의료", "핀테크", "금융", 
    "클라우드", "5G", "IoT", "블록체인", "메타버스"
]

# BM25 파라미터
BM25_K1 = 1.2
BM25_B = 0.75

# NMF 파라미터
NMF_N_TOPICS = 20
NMF_MAX_FEATURES = 1000
NMF_MIN_DF = 2
NMF_MAX_DF = 0.95

# PDF 처리 설정
PDF_EXTRACTION_TIMEOUT = 30
PDF_BATCH_SIZE = 10
PDF_CONFIDENCE_THRESHOLD = 0.3

# 텍스트 전처리 설정
MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 10000
STOPWORDS_FILE = "data/stopwords.txt"
KEYWORD_PATTERNS_FILE = "data/keyword_patterns.txt"

# 로깅 설정
LOG_LEVEL = "INFO"
LOG_FILE = "logs/scoring_system.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 결과 저장 설정
RESULTS_DIR = "results"
DATA_DIR = "data"
LOGS_DIR = "logs"

# 스케줄러 설정
SCHEDULER_CHECK_INTERVAL = 60  # 초
DAILY_STATUS_CHECK_TIME = "06:00"
REBALANCING_CHECK_DAY = 1  # 매월 1일

# 성능 설정
MAX_WORKERS = 5
MEMORY_LIMIT_GB = 8
CACHE_SIZE_MB = 1000
