# 텍스트 마이닝 기반 기업-토픽 연관도 스코어링 시스템
# 6개월 리밸런싱 주기로 국내 상장기업 전체 대상 분석

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

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
        logging.FileHandler('logs/scoring_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """시스템 설정 클래스"""
    api_key: str = "ZGVlcHNlYXJjaDoyMWNkNTExODVkN2RlNjFjMGY0Yg=="
    base_url: str = "https://api-v2.deepsearch.com"
    country_code: str = "kr"
    page_size: int = 100
    rebalancing_months: int = 6
    target_keywords: List[str] = None
    bm25_weight: float = 0.6
    topic_weight: float = 0.4
    min_documents_per_company: int = 5
    
    def __post_init__(self):
        if self.target_keywords is None:
            self.target_keywords = [
                "반도체", "AI", "인공지능", "배터리", "전기차", "EV", 
                "신재생에너지", "바이오", "의료", "핀테크", "금융", 
                "클라우드", "5G", "IoT", "블록체인", "메타버스"
            ]

def run_period_execution(execution_name: str, start_date: str, end_date: str, 
                        rebalancing_months: int = 6):
    """시점별 실행 함수"""
    try:
        logger.info(f"시점별 실행 시작: {execution_name}")
        logger.info(f"실행 기간: {start_date} ~ {end_date}")
        
        # 설정 초기화
        config = SystemConfig()
        config.rebalancing_months = rebalancing_months
        
        # 디렉토리 생성
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # 시스템 컴포넌트 초기화
        api_client = DeepSearchClient(config)
        pdf_processor = PDFProcessor()
        text_preprocessor = TextPreprocessor()
        bm25_scorer = BM25Scorer()
        topic_modeler = TopicModeler()
        scoring_system = ScoringSystem(config, bm25_scorer, topic_modeler)
        
        # 시점별 실행 관리자 초기화
        execution_manager = PeriodExecutionManager(
            execution_name, start_date, end_date, rebalancing_months
        )
        
        # 모든 시점 실행
        execution_manager.execute_all_periods(
            scoring_system, api_client, pdf_processor, text_preprocessor
        )
        
        # 실행 상태 출력
        status = execution_manager.get_execution_status()
        logger.info(f"시점별 실행 완료: {status}")
        
        return execution_manager
        
    except Exception as e:
        logger.error(f"시점별 실행 중 오류 발생: {str(e)}")
        raise

def main():
    """메인 실행 함수"""
    try:
        # 설정 초기화
        config = SystemConfig()
        
        # 디렉토리 생성
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        logger.info("텍스트 마이닝 기반 기업-토픽 연관도 스코어링 시스템 시작")
        
        # 시스템 컴포넌트 초기화
        api_client = DeepSearchClient(config)
        pdf_processor = PDFProcessor()
        text_preprocessor = TextPreprocessor()
        bm25_scorer = BM25Scorer()
        topic_modeler = TopicModeler()
        scoring_system = ScoringSystem(config, bm25_scorer, topic_modeler)
        scheduler = RebalancingScheduler(config, scoring_system)
        
        # 리밸런싱 스케줄러 시작
        scheduler.start()
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
