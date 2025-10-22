"""
리밸런싱 스케줄러
6개월 주기로 자동 리밸런싱을 수행하는 스케줄러
"""

import logging
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RebalancingResult:
    """리밸런싱 결과 데이터 클래스"""
    rebalancing_date: datetime
    data_period: str
    total_companies: int
    total_documents: int
    recommendations: List[Dict]
    scoring_results: Dict
    execution_time: float
    status: str
    error_message: Optional[str] = None

class RebalancingScheduler:
    """리밸런싱 스케줄러"""
    
    def __init__(self, config, scoring_system):
        self.config = config
        self.scoring_system = scoring_system
        self.api_client = None
        self.pdf_processor = None
        self.text_preprocessor = None
        
        self.rebalancing_months = config.rebalancing_months
        self.is_running = False
        self.scheduler_thread = None
        self.last_rebalancing = None
        self.rebalancing_history = []
        
        # 스케줄 설정
        self._setup_schedule()
        
    def _setup_schedule(self):
        """스케줄 설정"""
        # 매월 1일 오전 9시에 리밸런싱 체크
        schedule.every().month.do(self._check_rebalancing_needed)
        
        # 매일 오전 6시에 상태 체크
        schedule.every().day.at("06:00").do(self._daily_status_check)
        
        logger.info("리밸런싱 스케줄 설정 완료")
    
    def _check_rebalancing_needed(self):
        """리밸런싱 필요 여부 확인"""
        logger.info("리밸런싱 필요 여부 확인")
        
        if self.last_rebalancing is None:
            logger.info("첫 번째 리밸런싱 실행")
            self._execute_rebalancing()
            return
        
        # 마지막 리밸런싱으로부터 경과 시간 확인
        time_since_last = datetime.now() - self.last_rebalancing
        months_since_last = time_since_last.days / 30
        
        if months_since_last >= self.rebalancing_months:
            logger.info(f"리밸런싱 실행 필요: {months_since_last:.1f}개월 경과")
            self._execute_rebalancing()
        else:
            remaining_months = self.rebalancing_months - months_since_last
            logger.info(f"리밸런싱 대기 중: {remaining_months:.1f}개월 남음")
    
    def _daily_status_check(self):
        """일일 상태 체크"""
        logger.info("일일 상태 체크")
        
        # 시스템 상태 확인
        status = {
            'timestamp': datetime.now().isoformat(),
            'is_running': self.is_running,
            'last_rebalancing': self.last_rebalancing.isoformat() if self.last_rebalancing else None,
            'next_rebalancing': self._get_next_rebalancing_date(),
            'total_rebalancings': len(self.rebalancing_history)
        }
        
        logger.info(f"시스템 상태: {status}")
        
        # 상태 파일 저장
        self._save_status(status)
    
    def _get_next_rebalancing_date(self) -> Optional[str]:
        """다음 리밸런싱 예정일 계산"""
        if self.last_rebalancing is None:
            return datetime.now().isoformat()
        
        next_date = self.last_rebalancing + timedelta(days=self.rebalancing_months * 30)
        return next_date.isoformat()
    
    def _execute_rebalancing(self):
        """리밸런싱 실행"""
        logger.info("리밸런싱 실행 시작")
        
        start_time = time.time()
        result = RebalancingResult(
            rebalancing_date=datetime.now(),
            data_period=f"{self.rebalancing_months}개월",
            total_companies=0,
            total_documents=0,
            recommendations=[],
            scoring_results={},
            execution_time=0.0,
            status="running"
        )
        
        try:
            # 1. 데이터 수집
            logger.info("1단계: 데이터 수집")
            companies, documents = self._collect_data()
            
            if not companies or not documents:
                raise Exception("데이터 수집 실패")
            
            result.total_companies = len(companies)
            result.total_documents = len(documents)
            
            # 2. PDF 텍스트 추출
            logger.info("2단계: PDF 텍스트 추출")
            extracted_texts = self._extract_texts(documents)
            
            if not extracted_texts:
                raise Exception("텍스트 추출 실패")
            
            # 3. 텍스트 전처리
            logger.info("3단계: 텍스트 전처리")
            processed_documents = self._preprocess_texts(extracted_texts)
            
            if not processed_documents:
                raise Exception("텍스트 전처리 실패")
            
            # 4. 스코어링 실행
            logger.info("4단계: 스코어링 실행")
            scoring_results = self.scoring_system.generate_scoring_results(processed_documents)
            
            if not scoring_results:
                raise Exception("스코어링 실패")
            
            result.scoring_results = scoring_results
            
            # 5. 리밸런싱 추천 생성
            logger.info("5단계: 리밸런싱 추천 생성")
            recommendations = self.scoring_system.get_rebalancing_recommendations(top_n=50)
            
            result.recommendations = recommendations
            
            # 6. 결과 저장
            logger.info("6단계: 결과 저장")
            self._save_rebalancing_results(result)
            
            # 성공 상태 업데이트
            result.status = "completed"
            result.execution_time = time.time() - start_time
            
            # 마지막 리밸런싱 시간 업데이트
            self.last_rebalancing = result.rebalancing_date
            
            # 히스토리에 추가
            self.rebalancing_history.append(result)
            
            logger.info(f"리밸런싱 완료: {result.execution_time:.2f}초 소요")
            
        except Exception as e:
            logger.error(f"리밸런싱 실행 실패: {str(e)}")
            result.status = "failed"
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
            
            # 실패한 리밸런싱도 히스토리에 추가
            self.rebalancing_history.append(result)
    
    def _collect_data(self) -> tuple[List, List]:
        """데이터 수집"""
        if not self.api_client:
            from src.api_client import DeepSearchClient
            self.api_client = DeepSearchClient(self.config)
        
        companies, documents = self.api_client.get_recent_documents(self.rebalancing_months)
        
        # 데이터 저장
        if companies and documents:
            self.api_client.save_data_to_csv(companies, documents, "data/rebalancing_data")
        
        return companies, documents
    
    def _extract_texts(self, documents: List[Dict]) -> List[Dict]:
        """PDF 텍스트 추출"""
        if not self.pdf_processor:
            from src.pdf_processor import PDFProcessor
            self.pdf_processor = PDFProcessor()
        
        # 문서를 딕셔너리 형태로 변환
        doc_dicts = []
        for doc in documents:
            doc_dicts.append({
                'document_id': doc.document_id,
                'title': doc.title,
                'url': doc.url,
                'company_symbol': doc.company_symbol,
                'publish_date': doc.publish_date.isoformat(),
                'document_type': doc.document_type
            })
        
        extracted_texts = self.pdf_processor.batch_extract_texts(doc_dicts)
        
        return extracted_texts
    
    def _preprocess_texts(self, extracted_texts: List[Dict]) -> List[Dict]:
        """텍스트 전처리"""
        if not self.text_preprocessor:
            from src.text_preprocessor import TextPreprocessor
            self.text_preprocessor = TextPreprocessor()
        
        processed_documents = self.text_preprocessor.batch_preprocess(extracted_texts)
        
        return processed_documents
    
    def _save_rebalancing_results(self, result: RebalancingResult):
        """리밸런싱 결과 저장"""
        timestamp = result.rebalancing_date.strftime('%Y%m%d_%H%M%S')
        
        # 전체 결과 저장
        result_filename = f"results/rebalancing_result_{timestamp}.json"
        os.makedirs('results', exist_ok=True)
        
        try:
            with open(result_filename, 'w', encoding='utf-8') as f:
                json.dump(result.__dict__, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"리밸런싱 결과 저장 완료: {result_filename}")
        except Exception as e:
            logger.error(f"결과 저장 실패: {str(e)}")
        
        # 스코어링 시스템 결과 저장
        if result.scoring_results:
            self.scoring_system.save_results(
                result.scoring_results, 
                f"results/scoring_{timestamp}"
            )
    
    def _save_status(self, status: Dict):
        """상태 정보 저장"""
        try:
            with open('results/system_status.json', 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"상태 저장 실패: {str(e)}")
    
    def start(self):
        """스케줄러 시작"""
        if self.is_running:
            logger.warning("스케줄러가 이미 실행 중입니다")
            return
        
        logger.info("리밸런싱 스케줄러 시작")
        self.is_running = True
        
        # 백그라운드 스레드에서 스케줄러 실행
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        # 즉시 첫 번째 리밸런싱 체크
        self._check_rebalancing_needed()
    
    def stop(self):
        """스케줄러 중지"""
        logger.info("리밸런싱 스케줄러 중지")
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
    
    def _run_scheduler(self):
        """스케줄러 실행 루프"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크
            except Exception as e:
                logger.error(f"스케줄러 실행 오류: {str(e)}")
                time.sleep(60)
    
    def force_rebalancing(self):
        """강제 리밸런싱 실행"""
        logger.info("강제 리밸런싱 실행")
        self._execute_rebalancing()
    
    def get_rebalancing_history(self) -> List[Dict]:
        """리밸런싱 히스토리 반환"""
        history = []
        for result in self.rebalancing_history:
            history.append({
                'rebalancing_date': result.rebalancing_date.isoformat(),
                'status': result.status,
                'total_companies': result.total_companies,
                'total_documents': result.total_documents,
                'execution_time': result.execution_time,
                'error_message': result.error_message
            })
        
        return history
    
    def get_next_rebalancing_date(self) -> Optional[str]:
        """다음 리밸런싱 예정일 반환"""
        return self._get_next_rebalancing_date()
    
    def get_system_status(self) -> Dict:
        """시스템 상태 반환"""
        return {
            'is_running': self.is_running,
            'last_rebalancing': self.last_rebalancing.isoformat() if self.last_rebalancing else None,
            'next_rebalancing': self.get_next_rebalancing_date(),
            'total_rebalancings': len(self.rebalancing_history),
            'rebalancing_months': self.rebalancing_months,
            'target_keywords': self.config.target_keywords
        }
