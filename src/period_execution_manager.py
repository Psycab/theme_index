"""
시점별 실행 관리 시스템
여러 시점에 걸친 실행 결과를 체계적으로 관리하고 통합 Excel 파일 생성
"""

import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict
import time
from tqdm import tqdm
from src.performance_optimizer import PerformanceOptimizer, performance_timer

logger = logging.getLogger(__name__)

class PeriodExecutionManager:
    """시점별 실행 관리 클래스"""
    
    def __init__(self, execution_name: str, start_date: str, end_date: str, 
                 rebalancing_months: int = 6, use_optimization: bool = True):
        self.execution_name = execution_name
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.rebalancing_months = rebalancing_months
        
        # 실행 폴더 생성
        self.execution_folder = f"execution_{execution_name}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}"
        self.results_base_path = f"results/{self.execution_folder}"
        os.makedirs(self.results_base_path, exist_ok=True)
        
        # 시점별 실행 정보 저장
        self.period_executions = []
        self.all_results = {}
        
        # 성능 최적화 초기화
        self.use_optimization = use_optimization
        if self.use_optimization:
            self.optimizer = PerformanceOptimizer(use_numba=True)
            logger.info("시점별 실행 관리자 성능 최적화 활성화")
        else:
            self.optimizer = None
            logger.info("시점별 실행 관리자 기본 모드")
        
        # 실행 통계 초기화
        self.execution_stats = {
            'total_periods': 0,
            'completed_periods': 0,
            'failed_periods': 0,
            'start_time': None,
            'end_time': None,
            'total_execution_time': 0,
            'period_times': {},  # 각 시점별 실행 시간
            'step_times': {}     # 각 단계별 실행 시간
        }
        
        logger.info(f"시점별 실행 관리자 초기화: {self.execution_name}")
        logger.info(f"실행 기간: {start_date} ~ {end_date}")
        logger.info(f"리밸런싱 주기: {rebalancing_months}개월")
    
    def generate_periods(self) -> List[Tuple[str, str, str]]:
        """실행 기간을 리밸런싱 주기로 나누어 시점 생성 (정확한 6개월 구간)"""
        periods = []
        current_start = self.start_date
        
        while current_start < self.end_date:
            # 시작일로부터 정확히 N개월 후의 마지막 영업일 계산
            # 예: 2020-06-01 시작이면 2020-11-30 종료 (6개월)
            
            # 목표 월 계산 (시작 월 + N개월 - 1)
            target_month = current_start.month + self.rebalancing_months - 1
            target_year = current_start.year
            
            # 연도 조정
            while target_month > 12:
                target_month -= 12
                target_year += 1
            
            # 해당 월의 마지막 날 계산
            if target_month == 12:
                next_month_first = datetime(target_year + 1, 1, 1)
            else:
                next_month_first = datetime(target_year, target_month + 1, 1)
            
            current_end = next_month_first - timedelta(days=1)
            
            # 종료일이 전체 종료일을 넘지 않도록 제한
            current_end = min(current_end, self.end_date)
            
            # 시점 폴더명 생성
            period_name = f"period_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}"
            
            periods.append((
                period_name,
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d')
            ))
            
            # 다음 시점의 시작일은 현재 종료일의 다음 날
            current_start = current_end + timedelta(days=1)
            
            # 시작일이 전체 종료일을 넘으면 중단
            if current_start >= self.end_date:
                break
        
        logger.info(f"총 {len(periods)}개 시점 생성 (정확한 {self.rebalancing_months}개월 구간)")
        for i, (period_name, start, end) in enumerate(periods):
            logger.info(f"  시점 {i+1}: {period_name} ({start} ~ {end})")
        
        return periods
    
    @performance_timer
    def execute_period(self, period_name: str, start_date: str, end_date: str, 
                      scoring_system, api_client, pdf_processor, text_preprocessor) -> Dict:
        """특정 시점 실행 (성능 최적화 + 단계별 시간 측정)"""
        logger.info(f"🔄 시점 실행 시작: {period_name} ({start_date} ~ {end_date})")
        
        # 단계별 시간 측정을 위한 딕셔너리
        step_times = {}
        period_start_time = time.time()
        
        try:
            # 1. 데이터 수집
            step_start = time.time()
            logger.info("📊 1단계: 데이터 수집")
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # 기업 목록 조회 (전체 기간에서 한 번만)
            if not hasattr(self, '_companies'):
                companies = api_client.get_all_companies()
                self._companies = companies
                logger.info(f"기업 목록 조회 완료: {len(companies)}개")
            else:
                companies = self._companies
            
            # 해당 기간의 문서 조회
            symbols = [c.symbol for c in companies if c.symbol]
            documents = api_client.get_company_documents(symbols, start_dt, end_dt)
            logger.info(f"문서 조회 완료: {len(documents)}개")
            
            if not documents:
                logger.warning(f"시점 {period_name}에 문서가 없습니다")
                return None
            
            # 2. PDF 텍스트 추출 (성능 최적화)
            logger.info("2단계: PDF 텍스트 추출 (최적화)")
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
            
            # 고성능 PDF 추출 사용
            if hasattr(pdf_processor, 'high_performance_batch_extract'):
                # 새로운 고성능 메서드 사용
                extracted_texts = pdf_processor.high_performance_batch_extract(
                    doc_dicts, max_workers=None, batch_size=50
                )
            elif self.use_optimization and self.optimizer:
                # 기존 최적화 메서드 사용
                extracted_texts = self.optimizer.parallel_extract_texts(
                    doc_dicts, pdf_processor, batch_size=20
                )
            else:
                # 기본 메서드 사용
                extracted_texts = pdf_processor.batch_extract_texts(doc_dicts)
            
            logger.info(f"텍스트 추출 완료: {len(extracted_texts)}개")
            
            # 3. 텍스트 전처리 (성능 최적화)
            logger.info("3단계: 텍스트 전처리 (최적화)")
            
            # 성능 최적화 사용 여부에 따라 다른 방법 선택
            if self.use_optimization and self.optimizer:
                processed_documents = self.optimizer.parallel_preprocess_texts(
                    extracted_texts, text_preprocessor, batch_size=100
                )
            else:
                processed_documents = text_preprocessor.batch_preprocess(extracted_texts)
            
            logger.info(f"전처리 완료: {len(processed_documents)}개")
            
            # 4. 스코어링 실행
            logger.info("4단계: 스코어링 실행")
            
            # 기업 정보를 스코어링 시스템에 설정
            scoring_system.set_company_info(companies)
            
            results = scoring_system.generate_scoring_results(processed_documents)
            
            if not results:
                logger.error(f"시점 {period_name} 스코어링 실패")
                return None
            
            # 5. 결과 저장
            logger.info("5단계: 결과 저장")
            saved_files = scoring_system.save_results(
                results, 
                "scoring_results",
                self.execution_folder,
                period_name
            )
            
            # 실행 정보 저장
            execution_info = {
                'period_name': period_name,
                'start_date': start_date,
                'end_date': end_date,
                'total_companies': results['total_companies'],
                'total_documents': results['total_documents'],
                'execution_time': datetime.now().isoformat(),
                'saved_files': saved_files,
                'use_optimization': self.use_optimization
            }
            
            self.period_executions.append(execution_info)
            self.all_results[period_name] = results
            
            logger.info(f"시점 {period_name} 실행 완료")
            return execution_info
            
        except Exception as e:
            logger.error(f"시점 {period_name} 실행 실패: {str(e)}")
            return None
    
    def execute_all_periods(self, scoring_system, api_client, pdf_processor, text_preprocessor):
        """모든 시점 실행 (진행률 표시 및 시간 측정 포함)"""
        logger.info("🚀 전체 시점 실행 시작")
        
        # 실행 시작 시간 기록
        self.execution_stats['start_time'] = time.time()
        
        periods = self.generate_periods()
        self.execution_stats['total_periods'] = len(periods)
        
        print(f"\n📊 실행 계획:")
        print(f"   • 총 시점 수: {len(periods)}개")
        print(f"   • 실행 기간: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        print(f"   • 리밸런싱 주기: {self.rebalancing_months}개월")
        print(f"   • 예상 실행 시간: {len(periods) * 15}~{len(periods) * 25}분")
        print()
        
        # 진행률 표시기로 시점별 실행
        with tqdm(total=len(periods), desc="시점별 실행", unit="시점", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for i, (period_name, start_date, end_date) in enumerate(periods, 1):
                period_start_time = time.time()
                
                # 현재 시점 정보 표시
                pbar.set_description(f"시점 {i}/{len(periods)}: {period_name}")
                
                execution_info = self.execute_period(
                    period_name, start_date, end_date,
                    scoring_system, api_client, pdf_processor, text_preprocessor
                )
                
                period_end_time = time.time()
                period_duration = period_end_time - period_start_time
                
                # 시점별 실행 시간 기록
                self.execution_stats['period_times'][period_name] = period_duration
                
                if execution_info:
                    self.execution_stats['completed_periods'] += 1
                    pbar.set_postfix({
                        '상태': '완료',
                        '소요시간': f"{period_duration:.1f}초",
                        '완료율': f"{self.execution_stats['completed_periods']}/{self.execution_stats['total_periods']}"
                    })
                    logger.info(f"✅ 시점 {period_name} 완료 ({period_duration:.1f}초)")
                else:
                    self.execution_stats['failed_periods'] += 1
                    pbar.set_postfix({
                        '상태': '실패',
                        '소요시간': f"{period_duration:.1f}초",
                        '완료율': f"{self.execution_stats['completed_periods']}/{self.execution_stats['total_periods']}"
                    })
                    logger.error(f"❌ 시점 {period_name} 실패 ({period_duration:.1f}초)")
                
                pbar.update(1)
        
        # 실행 종료 시간 기록
        self.execution_stats['end_time'] = time.time()
        self.execution_stats['total_execution_time'] = self.execution_stats['end_time'] - self.execution_stats['start_time']
        
        # 실행 결과 요약 출력
        self._print_execution_summary()
        
        # 통합 Excel 파일 생성
        logger.info("📊 통합 Excel 파일 생성 중...")
        excel_start_time = time.time()
        self.create_consolidated_excel()
        excel_duration = time.time() - excel_start_time
        logger.info(f"✅ 통합 Excel 파일 생성 완료 ({excel_duration:.1f}초)")
        
        # 실행 요약 저장
        logger.info("💾 실행 요약 저장 중...")
        self.save_execution_summary()
        
        logger.info("🎉 전체 시점 실행 완료")
    
    def _print_execution_summary(self):
        """실행 결과 요약 출력"""
        total_time = self.execution_stats['total_execution_time']
        completed = self.execution_stats['completed_periods']
        failed = self.execution_stats['failed_periods']
        total = self.execution_stats['total_periods']
        
        print(f"\n{'='*60}")
        print(f"📊 실행 결과 요약")
        print(f"{'='*60}")
        print(f"총 실행 시간: {total_time/60:.1f}분 ({total_time:.1f}초)")
        print(f"완료된 시점: {completed}/{total} ({completed/total*100:.1f}%)")
        print(f"실패한 시점: {failed}/{total} ({failed/total*100:.1f}%)")
        
        if self.execution_stats['period_times']:
            avg_time = sum(self.execution_stats['period_times'].values()) / len(self.execution_stats['period_times'])
            print(f"평균 시점 실행 시간: {avg_time:.1f}초")
            
            # 가장 빠른/느린 시점
            fastest_period = min(self.execution_stats['period_times'].items(), key=lambda x: x[1])
            slowest_period = max(self.execution_stats['period_times'].items(), key=lambda x: x[1])
            print(f"가장 빠른 시점: {fastest_period[0]} ({fastest_period[1]:.1f}초)")
            print(f"가장 느린 시점: {slowest_period[0]} ({slowest_period[1]:.1f}초)")
        
        print(f"{'='*60}")
    
    def create_consolidated_excel(self):
        """시점별 통합 Excel 파일 생성"""
        logger.info("통합 Excel 파일 생성 시작")
        
        # 키워드 정보를 파일명에 포함
        keyword_info = self._create_keyword_filename_suffix()
        excel_file = f"{self.results_base_path}/consolidated_results_{keyword_info}.xlsx"
        
        try:
            # openpyxl 엔진 사용 가능 여부 확인
            try:
                import openpyxl
                engine = 'openpyxl'
                logger.info("openpyxl 엔진 사용")
            except ImportError:
                logger.warning("openpyxl 사용 불가, 기본 엔진 사용")
                engine = None
            
            with pd.ExcelWriter(excel_file, engine=engine) as writer:
                # 1. 시점별 기업 스코어 시트들
                for period_name, results in self.all_results.items():
                    if 'company_scores' in results:
                        company_scores_data = []
                        for company, scores in results['company_scores'].items():
                            # 기업명 조회 (기업 정보가 있는 경우)
                            company_name = self._get_company_name(company)
                            row = {
                                'company_symbol': company,
                                'company_name': company_name,
                                **scores
                            }
                            company_scores_data.append(row)
                        
                        company_scores_df = pd.DataFrame(company_scores_data)
                        
                        # 시트명 생성 (Excel 시트명 제한 고려)
                        sheet_name = period_name.replace('period_', '').replace('_', '-')[:31]
                        company_scores_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 2. 시점별 요약 시트
                summary_data = []
                for execution in self.period_executions:
                    summary_data.append({
                        'period_name': execution['period_name'],
                        'start_date': execution['start_date'],
                        'end_date': execution['end_date'],
                        'total_companies': execution['total_companies'],
                        'total_documents': execution['total_documents'],
                        'execution_time': execution['execution_time']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Execution_Summary', index=False)
                
                # 3. 키워드별 시점간 비교 시트
                self._create_keyword_comparison_sheet(writer)
                
                # 4. 기업별 시점간 스코어 변화 시트
                self._create_company_trend_sheet(writer)
            
            logger.info(f"통합 Excel 파일 생성 완료: {excel_file}")
            
        except Exception as e:
            logger.error(f"통합 Excel 파일 생성 실패: {str(e)}")
    
    def _create_keyword_comparison_sheet(self, writer):
        """키워드별 시점간 비교 시트 생성"""
        try:
            # 모든 시점의 키워드별 상위 기업 수집
            keyword_comparison = defaultdict(list)
            
            for period_name, results in self.all_results.items():
                if 'top_companies_per_keyword' in results:
                    for keyword, companies in results['top_companies_per_keyword'].items():
                        for company in companies[:10]:  # 상위 10개만
                            keyword_comparison[keyword].append({
                                'period': period_name.replace('period_', ''),
                                'company_symbol': company['company_symbol'],
                                'score': company['score'],
                                'rank': companies.index(company) + 1
                            })
            
            # 각 키워드별로 시트 생성
            for keyword, companies in keyword_comparison.items():
                if companies:
                    df = pd.DataFrame(companies)
                    sheet_name = f"Keyword_{keyword}"[:31]  # Excel 시트명 제한
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        except Exception as e:
            logger.error(f"키워드 비교 시트 생성 실패: {str(e)}")
    
    def _create_company_trend_sheet(self, writer):
        """기업별 시점간 스코어 변화 시트 생성"""
        try:
            # 모든 기업 심볼 수집
            all_companies = set()
            for results in self.all_results.values():
                if 'company_scores' in results:
                    all_companies.update(results['company_scores'].keys())
            
            # 기업별 시점간 스코어 변화 데이터 생성
            trend_data = []
            for company in all_companies:
                company_trend = {'company_symbol': company}
                
                for period_name, results in self.all_results.items():
                    if 'company_scores' in results and company in results['company_scores']:
                        scores = results['company_scores'][company]
                        # 평균 스코어 계산
                        avg_score = sum(scores.values()) / len(scores) if scores else 0
                        company_trend[f"{period_name}_avg_score"] = avg_score
                        
                        # 각 키워드별 스코어 추가
                        for keyword, score in scores.items():
                            company_trend[f"{period_name}_{keyword}"] = score
                    else:
                        company_trend[f"{period_name}_avg_score"] = 0
                
                trend_data.append(company_trend)
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                trend_df.to_excel(writer, sheet_name='Company_Trends', index=False)
        
        except Exception as e:
            logger.error(f"기업 트렌드 시트 생성 실패: {str(e)}")
    
    def _create_keyword_filename_suffix(self) -> str:
        """키워드 정보를 파일명 접미사로 생성"""
        try:
            # 첫 번째 실행에서 키워드 정보 가져오기
            if self.period_executions:
                first_execution = self.period_executions[0]
                keywords = first_execution.get('target_keywords', [])
                
                if not keywords:
                    return "no_keywords"
                
                # 키워드 개수에 따라 접미사 생성
                if len(keywords) == 1:
                    # 단일 키워드인 경우
                    keyword = keywords[0]
                    return f"KW_{keyword}"
                elif len(keywords) <= 3:
                    # 3개 이하인 경우 모든 키워드 포함
                    keywords_str = "_".join(keywords)
                    return f"KW_{keywords_str}"
                else:
                    # 3개 초과인 경우 첫 3개만 포함하고 개수 표시
                    first_three = "_".join(keywords[:3])
                    return f"KW_{first_three}_and_{len(keywords)-3}more"
            else:
                return "no_executions"
                
        except Exception as e:
            logger.error(f"키워드 파일명 접미사 생성 실패: {str(e)}")
            return "keywords_error"
    
    def _get_company_name(self, symbol: str) -> str:
        """기업 심볼로부터 한글명 조회"""
        if hasattr(self, '_companies') and self._companies:
            for company in self._companies:
                if hasattr(company, 'symbol') and company.symbol == symbol:
                    return getattr(company, 'name', symbol)
                elif isinstance(company, dict) and company.get('symbol') == symbol:
                    return company.get('name', symbol)
        return symbol
    
    def save_execution_summary(self):
        """실행 요약 저장"""
        summary = {
            'execution_name': self.execution_name,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'rebalancing_months': self.rebalancing_months,
            'total_periods': len(self.period_executions),
            'successful_periods': len([e for e in self.period_executions if e]),
            'period_executions': self.period_executions,
            'created_at': datetime.now().isoformat()
        }
        
        summary_file = f"{self.results_base_path}/execution_summary.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"실행 요약 저장 완료: {summary_file}")
        except Exception as e:
            logger.error(f"실행 요약 저장 실패: {str(e)}")
    
    def get_execution_status(self) -> Dict:
        """실행 상태 반환"""
        return {
            'execution_name': self.execution_name,
            'total_periods': len(self.period_executions),
            'completed_periods': len([e for e in self.period_executions if e]),
            'execution_folder': self.execution_folder,
            'results_path': self.results_base_path
        }
