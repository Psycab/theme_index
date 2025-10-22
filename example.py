"""
실행 예제 및 테스트 스크립트
"""

import logging
import sys
import os
from datetime import datetime
import argparse

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api_client import DeepSearchClient
from src.pdf_processor import PDFProcessor
from src.text_preprocessor import TextPreprocessor
from src.bm25_scorer import BM25Scorer
from src.topic_modeler import TopicModeler
from src.scoring_system import ScoringSystem
from src.rebalancing_scheduler import RebalancingScheduler
from src.period_execution_manager import PeriodExecutionManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/example.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_api_client():
    """API 클라이언트 테스트"""
    logger.info("=== API 클라이언트 테스트 ===")
    
    from main import SystemConfig
    config = SystemConfig()
    
    api_client = DeepSearchClient(config)
    
    # 기업 목록 조회 테스트 (소규모)
    logger.info("기업 목록 조회 테스트")
    companies = api_client.get_all_companies()
    logger.info(f"조회된 기업 수: {len(companies)}")
    
    if companies:
        # 상위 5개 기업의 문서 조회 테스트
        test_symbols = [c.symbol for c in companies[:5]]
        logger.info(f"테스트 기업 심볼: {test_symbols}")
        
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 최근 30일
        
        documents = api_client.get_company_documents(test_symbols, start_date, end_date)
        logger.info(f"조회된 문서 수: {len(documents)}")
        
        return companies, documents
    
    return [], []

def test_pdf_processing():
    """PDF 처리 테스트"""
    logger.info("=== PDF 처리 테스트 ===")
    
    pdf_processor = PDFProcessor()
    
    # 테스트용 PDF URL (실제 존재하는 PDF)
    test_url = "https://example.com/test.pdf"  # 실제 테스트 시 유효한 URL 사용
    
    logger.info(f"PDF 텍스트 추출 테스트: {test_url}")
    # extracted = pdf_processor.extract_text_from_pdf(test_url)
    # logger.info(f"추출 결과: {extracted.extraction_method}, 신뢰도: {extracted.confidence_score}")
    
    logger.info("PDF 처리 테스트 완료")

def test_text_preprocessing():
    """텍스트 전처리 테스트"""
    logger.info("=== 텍스트 전처리 테스트 ===")
    
    preprocessor = TextPreprocessor()
    
    # 테스트 텍스트
    test_text = """
    삼성전자는 반도체와 AI 기술에 집중하고 있습니다. 
    최근 메모리 반도체 시장에서 강세를 보이고 있으며, 
    인공지능 칩 개발에도 투자하고 있습니다.
    """
    
    logger.info(f"테스트 텍스트: {test_text}")
    
    # 전처리 실행
    result = preprocessor.preprocess_document(test_text)
    
    logger.info(f"정리된 텍스트: {result['cleaned_text']}")
    logger.info(f"토큰 수: {result['word_count']}")
    logger.info(f"키워드: {result['keywords']}")
    
    logger.info("텍스트 전처리 테스트 완료")

def test_scoring_system():
    """스코어링 시스템 테스트"""
    logger.info("=== 스코어링 시스템 테스트 ===")
    
    from main import SystemConfig
    config = SystemConfig()
    
    # 테스트용 문서 데이터
    test_documents = [
        {
            'document_id': 'test_1',
            'title': '삼성전자 반도체 사업 분석',
            'company_symbol': '005930',
            'filtered_tokens': ['삼성전자', '반도체', '메모리', 'AI', '인공지능', '칩', '시장', '성장'],
            'publish_date': '2024-01-01',
            'document_type': 'research'
        },
        {
            'document_id': 'test_2',
            'title': 'LG에너지솔루션 배터리 기술',
            'company_symbol': '373220',
            'filtered_tokens': ['LG에너지솔루션', '배터리', '리튬', '전기차', 'EV', '에너지저장'],
            'publish_date': '2024-01-02',
            'document_type': 'research'
        },
        {
            'document_id': 'test_3',
            'title': '네이버 AI 플랫폼 전략',
            'company_symbol': '035420',
            'filtered_tokens': ['네이버', 'AI', '인공지능', '플랫폼', '클라우드', '데이터'],
            'publish_date': '2024-01-03',
            'document_type': 'research'
        }
    ]
    
    # BM25 스코어러 테스트
    bm25_scorer = BM25Scorer()
    if bm25_scorer.build_index(test_documents):
        logger.info("BM25 인덱스 구축 성공")
        
        keyword_scores = bm25_scorer.calculate_keyword_scores(['반도체', 'AI', '배터리'])
        logger.info(f"키워드 스코어: {keyword_scores}")
    
    # 토픽 모델러 테스트
    topic_modeler = TopicModeler(n_topics=5)
    if topic_modeler.fit_model(test_documents):
        logger.info("토픽 모델 학습 성공")
        topic_modeler.print_topic_summary()
    
    # 통합 스코어링 시스템 테스트
    scoring_system = ScoringSystem(config, bm25_scorer, topic_modeler)
    results = scoring_system.generate_scoring_results(test_documents)
    
    if results:
        logger.info("스코어링 결과 생성 성공")
        logger.info(f"총 기업 수: {results['total_companies']}")
        logger.info(f"총 문서 수: {results['total_documents']}")
        
        # 상위 기업 출력
        for keyword, companies in results['top_companies_per_keyword'].items():
            logger.info(f"{keyword} 상위 기업: {companies[:3]}")
    
    logger.info("스코어링 시스템 테스트 완료")

def run_full_pipeline():
    """전체 파이프라인 실행"""
    logger.info("=== 전체 파이프라인 실행 ===")
    
    from main import SystemConfig
    config = SystemConfig()
    
    try:
        # 1. 데이터 수집
        logger.info("1단계: 데이터 수집")
        api_client = DeepSearchClient(config)
        companies, documents = api_client.get_recent_documents(months=1)  # 테스트용 1개월
        
        if not companies or not documents:
            logger.error("데이터 수집 실패")
            return
        
        logger.info(f"수집된 기업: {len(companies)}개, 문서: {len(documents)}개")
        
        # 2. PDF 텍스트 추출 (소규모 테스트)
        logger.info("2단계: PDF 텍스트 추출")
        pdf_processor = PDFProcessor()
        
        # 상위 10개 문서만 테스트
        test_documents = documents[:10]
        doc_dicts = []
        for doc in test_documents:
            doc_dicts.append({
                'document_id': doc.document_id,
                'title': doc.title,
                'url': doc.url,
                'company_symbol': doc.company_symbol,
                'publish_date': doc.publish_date.isoformat(),
                'document_type': doc.document_type
            })
        
        extracted_texts = pdf_processor.batch_extract_texts(doc_dicts)
        logger.info(f"텍스트 추출 완료: {len(extracted_texts)}개 문서")
        
        # 3. 텍스트 전처리
        logger.info("3단계: 텍스트 전처리")
        text_preprocessor = TextPreprocessor()
        processed_documents = text_preprocessor.batch_preprocess(extracted_texts)
        logger.info(f"전처리 완료: {len(processed_documents)}개 문서")
        
        # 4. 스코어링
        logger.info("4단계: 스코어링")
        bm25_scorer = BM25Scorer()
        topic_modeler = TopicModeler(n_topics=10)
        scoring_system = ScoringSystem(config, bm25_scorer, topic_modeler)
        
        results = scoring_system.generate_scoring_results(processed_documents)
        
        if results:
            logger.info("스코어링 완료")
            scoring_system.save_results(results, "test_results")
            
            # 리밸런싱 추천
            recommendations = scoring_system.get_rebalancing_recommendations(top_n=20)
            logger.info(f"리밸런싱 추천: {len(recommendations)}개 기업")
            
            for i, rec in enumerate(recommendations[:5]):
                logger.info(f"{i+1}. {rec['company_symbol']}: {rec['average_score']:.3f}")
        
        logger.info("전체 파이프라인 실행 완료")
        
    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {str(e)}")
        raise

def test_period_execution():
    """시점별 실행 테스트"""
    logger.info("=== 시점별 실행 테스트 ===")
    
    from main import SystemConfig
    config = SystemConfig()
    
    # 테스트용 시점별 실행 (짧은 기간)
    execution_name = "test_execution"
    start_date = "2024-01-01"
    end_date = "2024-03-01"  # 2개월 테스트
    rebalancing_months = 1   # 1개월 주기로 테스트
    
    logger.info(f"테스트 실행: {execution_name}")
    logger.info(f"기간: {start_date} ~ {end_date}")
    logger.info(f"리밸런싱 주기: {rebalancing_months}개월")
    
    try:
        # 시점별 실행 관리자 초기화
        execution_manager = PeriodExecutionManager(
            execution_name, start_date, end_date, rebalancing_months
        )
        
        # 시점 생성 확인
        periods = execution_manager.generate_periods()
        logger.info(f"생성된 시점 수: {len(periods)}")
        
        for i, (period_name, start, end) in enumerate(periods):
            logger.info(f"  시점 {i+1}: {period_name} ({start} ~ {end})")
        
        logger.info("시점별 실행 테스트 완료")
        
    except Exception as e:
        logger.error(f"시점별 실행 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()

def run_period_execution_example():
    """시점별 실행 예제"""
    logger.info("=== 시점별 실행 예제 ===")
    
    from main import run_period_execution
    
    # 실제 실행 예제 (2021-12-01 ~ 2024-12-01)
    execution_name = "historical_analysis"
    start_date = "2021-12-01"
    end_date = "2024-12-01"
    rebalancing_months = 6
    
    logger.info(f"시점별 실행 시작: {execution_name}")
    logger.info(f"기간: {start_date} ~ {end_date}")
    logger.info(f"리밸런싱 주기: {rebalancing_months}개월")
    
    try:
        execution_manager = run_period_execution(
            execution_name, start_date, end_date, rebalancing_months
        )
        
        status = execution_manager.get_execution_status()
        logger.info(f"실행 완료: {status}")
        
    except Exception as e:
        logger.error(f"시점별 실행 실패: {str(e)}")
        import traceback
        traceback.print_exc()

def run_custom_period_execution(execution_name: str, start_date: str, end_date: str, 
                               rebalancing_months: int = 6):
    """사용자 지정 기간으로 시점별 실행"""
    logger.info("=== 사용자 지정 기간 시점별 실행 ===")
    
    from main import run_period_execution
    
    logger.info(f"실행명: {execution_name}")
    logger.info(f"시작일: {start_date}")
    logger.info(f"종료일: {end_date}")
    logger.info(f"리밸런싱 주기: {rebalancing_months}개월")
    
    try:
        execution_manager = run_period_execution(
            execution_name, start_date, end_date, rebalancing_months
        )
        
        status = execution_manager.get_execution_status()
        logger.info(f"실행 완료: {status}")
        
        return execution_manager
        
    except Exception as e:
        logger.error(f"시점별 실행 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='텍스트 마이닝 기반 기업-토픽 연관도 스코어링 시스템')
    parser.add_argument('--test', choices=['api', 'pdf', 'text', 'scoring', 'full', 'period'], 
                       help='테스트 모드 선택')
    parser.add_argument('--run-scheduler', action='store_true', 
                       help='리밸런싱 스케줄러 실행')
    parser.add_argument('--run-period', action='store_true',
                       help='시점별 실행 예제 실행')
    
    # 기간 지정 옵션 추가
    parser.add_argument('--execution-name', type=str, default='custom_execution',
                       help='실행명 (기본값: custom_execution)')
    parser.add_argument('--start-date', type=str, 
                       help='시작일 (YYYY-MM-DD 형식, 예: 2021-12-01)')
    parser.add_argument('--end-date', type=str,
                       help='종료일 (YYYY-MM-DD 형식, 예: 2024-12-01)')
    parser.add_argument('--rebalancing-months', type=int, default=6,
                       help='리밸런싱 주기 (개월, 기본값: 6)')
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.test:
        if args.test == 'api':
            test_api_client()
        elif args.test == 'pdf':
            test_pdf_processing()
        elif args.test == 'text':
            test_text_preprocessing()
        elif args.test == 'scoring':
            test_scoring_system()
        elif args.test == 'full':
            run_full_pipeline()
        elif args.test == 'period':
            test_period_execution()
    
    elif args.run_scheduler:
        logger.info("리밸런싱 스케줄러 시작")
        from main import SystemConfig
        config = SystemConfig()
        
        bm25_scorer = BM25Scorer()
        topic_modeler = TopicModeler()
        scoring_system = ScoringSystem(config, bm25_scorer, topic_modeler)
        scheduler = RebalancingScheduler(config, scoring_system)
        
        try:
            scheduler.start()
            logger.info("스케줄러가 실행 중입니다. Ctrl+C로 중지하세요.")
            
            # 무한 대기
            import time
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("스케줄러 중지 중...")
            scheduler.stop()
            logger.info("스케줄러가 중지되었습니다.")
    
    elif args.run_period:
        run_period_execution_example()
    
    # 기간이 지정된 경우 사용자 지정 실행
    elif args.start_date and args.end_date:
        logger.info("사용자 지정 기간으로 실행")
        execution_manager = run_custom_period_execution(
            execution_name=args.execution_name,
            start_date=args.start_date,
            end_date=args.end_date,
            rebalancing_months=args.rebalancing_months
        )
        
        if execution_manager:
            logger.info("사용자 지정 기간 실행 완료")
        else:
            logger.error("사용자 지정 기간 실행 실패")
    
    else:
        # 기본 실행
        logger.info("기본 테스트 실행")
        test_text_preprocessing()
        test_scoring_system()

if __name__ == "__main__":
    main()
