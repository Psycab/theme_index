"""
통합 스코어링 시스템
BM25와 NMF 토픽 모델링 결과를 결합하여 최종 기업-토픽 연관도 점수 계산
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

class ScoringSystem:
    """통합 스코어링 시스템"""
    
    def __init__(self, config, bm25_scorer, topic_modeler):
        self.config = config
        self.bm25_scorer = bm25_scorer
        self.topic_modeler = topic_modeler
        
        self.target_keywords = config.target_keywords
        self.bm25_weight = config.bm25_weight
        self.topic_weight = config.topic_weight
        self.min_documents_per_company = config.min_documents_per_company
        
        self.final_scores = {}
        self.scoring_metadata = {}
        self.company_info = {}  # 기업 정보 저장 (symbol -> name 매핑)
        
    def set_company_info(self, companies: List[Dict]):
        """기업 정보 설정"""
        self.company_info = {}
        for company in companies:
            if isinstance(company, dict):
                symbol = company.get('symbol', '')
                name = company.get('name', '')
                if symbol:
                    self.company_info[symbol] = {
                        'name': name,
                        'sector': company.get('sector', ''),
                        'industry': company.get('industry', ''),
                        'market_cap': company.get('market_cap', None)
                    }
            else:
                # Company 객체인 경우
                symbol = getattr(company, 'symbol', '')
                name = getattr(company, 'name', '')
                if symbol:
                    self.company_info[symbol] = {
                        'name': name,
                        'sector': getattr(company, 'sector', ''),
                        'industry': getattr(company, 'industry', ''),
                        'market_cap': getattr(company, 'market_cap', None)
                    }
        
        logger.info(f"기업 정보 설정 완료: {len(self.company_info)}개 기업")
        
    def get_company_name(self, symbol: str) -> str:
        """기업 심볼로부터 한글명 조회"""
        return self.company_info.get(symbol, {}).get('name', symbol)
        
    def calculate_bm25_scores(self, documents: List[Dict]) -> Dict[str, Dict[str, float]]:
        """BM25 스코어 계산"""
        logger.info("BM25 스코어 계산 시작")
        
        # BM25 인덱스 구축
        if not self.bm25_scorer.build_index(documents):
            logger.error("BM25 인덱스 구축 실패")
            return {}
        
        # 키워드별 스코어 계산
        bm25_scores = self.bm25_scorer.get_company_keyword_scores(self.target_keywords)
        
        logger.info(f"BM25 스코어 계산 완료: {len(bm25_scores)}개 기업")
        return bm25_scores
    
    def calculate_topic_scores(self, documents: List[Dict]) -> Dict[str, Dict[str, float]]:
        """토픽 모델링 스코어 계산"""
        logger.info("토픽 모델링 스코어 계산 시작")
        
        # 토픽 모델 학습
        if not self.topic_modeler.fit_model(documents):
            logger.error("토픽 모델 학습 실패")
            return {}
        
        # 기업별 토픽 분포 계산
        company_topics = self.topic_modeler.get_company_topic_distribution()
        
        # 타겟 키워드와 토픽 매핑
        topic_scores = self._map_keywords_to_topics(company_topics)
        
        logger.info(f"토픽 스코어 계산 완료: {len(topic_scores)}개 기업")
        return topic_scores
    
    def _map_keywords_to_topics(self, company_topics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """키워드와 토픽 매핑"""
        keyword_topic_mapping = self._create_keyword_topic_mapping()
        
        mapped_scores = defaultdict(lambda: defaultdict(float))
        
        for company_symbol, topics in company_topics.items():
            for keyword in self.target_keywords:
                if keyword in keyword_topic_mapping:
                    mapped_topics = keyword_topic_mapping[keyword]
                    
                    # 매핑된 토픽들의 평균 스코어 계산
                    topic_scores = []
                    for topic_name in mapped_topics:
                        if topic_name in topics:
                            topic_scores.append(topics[topic_name])
                    
                    if topic_scores:
                        mapped_scores[company_symbol][keyword] = np.mean(topic_scores)
        
        return dict(mapped_scores)
    
    def _create_keyword_topic_mapping(self) -> Dict[str, List[str]]:
        """키워드-토픽 매핑 생성"""
        mapping = {}
        
        # 토픽 이름에서 키워드 매칭
        topic_names = self.topic_modeler.topic_names
        
        for keyword in self.target_keywords:
            matched_topics = []
            
            for topic_name in topic_names:
                # 키워드가 토픽 이름에 포함되는지 확인
                if keyword.lower() in topic_name.lower():
                    matched_topics.append(topic_name)
                
                # 관련 키워드 매칭
                related_keywords = self._get_related_keywords(keyword)
                for related in related_keywords:
                    if related.lower() in topic_name.lower():
                        matched_topics.append(topic_name)
            
            # 중복 제거
            mapping[keyword] = list(set(matched_topics))
        
        return mapping
    
    def _get_related_keywords(self, keyword: str) -> List[str]:
        """관련 키워드 반환"""
        related_map = {
            '반도체': ['칩', '메모리', 'DRAM', 'NAND', '플래시'],
            'AI': ['인공지능', '머신러닝', '딥러닝', '신경망'],
            '배터리': ['리튬', '전지', '에너지저장', 'ESS'],
            '전기차': ['EV', '하이브리드', '충전'],
            '신재생에너지': ['태양광', '풍력', '재생에너지'],
            '바이오': ['생명공학', '의료기기', '제약'],
            '핀테크': ['금융기술', '모바일결제', '디지털뱅킹'],
            '클라우드': ['클라우드컴퓨팅', 'SaaS', 'PaaS'],
            '5G': ['무선통신', '모바일통신', '네트워크'],
            'IoT': ['사물인터넷', '스마트', '연결'],
            '블록체인': ['암호화폐', '비트코인', '이더리움'],
            '메타버스': ['가상현실', 'VR', 'AR', 'MR']
        }
        
        return related_map.get(keyword, [])
    
    def calculate_final_scores(self, bm25_scores: Dict[str, Dict[str, float]], 
                             topic_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """최종 통합 스코어 계산"""
        logger.info("최종 통합 스코어 계산 시작")
        
        final_scores = defaultdict(lambda: defaultdict(float))
        
        # 모든 기업 심볼 수집
        all_companies = set()
        all_companies.update(bm25_scores.keys())
        all_companies.update(topic_scores.keys())
        
        for company_symbol in all_companies:
            for keyword in self.target_keywords:
                # BM25 스코어
                bm25_score = bm25_scores.get(company_symbol, {}).get(keyword, 0.0)
                
                # 토픽 스코어
                topic_score = topic_scores.get(company_symbol, {}).get(keyword, 0.0)
                
                # 가중 평균 계산
                final_score = (self.bm25_weight * bm25_score + 
                             self.topic_weight * topic_score)
                
                final_scores[company_symbol][keyword] = final_score
        
        # 정규화 (0-1 범위)
        final_scores = self._normalize_scores(final_scores)
        
        logger.info(f"최종 통합 스코어 계산 완료: {len(final_scores)}개 기업")
        return dict(final_scores)
    
    def _normalize_scores(self, scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """스코어 정규화"""
        normalized_scores = {}
        
        for keyword in self.target_keywords:
            # 키워드별 최대값 찾기
            keyword_scores = [scores.get(company, {}).get(keyword, 0.0) 
                            for company in scores.keys()]
            max_score = max(keyword_scores) if keyword_scores else 1.0
            
            if max_score > 0:
                for company_symbol in scores:
                    if company_symbol not in normalized_scores:
                        normalized_scores[company_symbol] = {}
                    
                    original_score = scores[company_symbol].get(keyword, 0.0)
                    normalized_scores[company_symbol][keyword] = original_score / max_score
        
        return normalized_scores
    
    def filter_companies_by_document_count(self, documents: List[Dict]) -> Dict[str, int]:
        """문서 수 기준 기업 필터링"""
        company_doc_counts = defaultdict(int)
        
        for doc in documents:
            company_symbol = doc.get('company_symbol', '')
            if company_symbol:
                company_doc_counts[company_symbol] += 1
        
        # 최소 문서 수 이상인 기업만 필터링
        filtered_companies = {
            company: count for company, count in company_doc_counts.items()
            if count >= self.min_documents_per_company
        }
        
        logger.info(f"문서 수 기준 필터링: {len(filtered_companies)}개 기업 (최소 {self.min_documents_per_company}개 문서)")
        return filtered_companies
    
    def generate_scoring_results(self, documents: List[Dict]) -> Dict:
        """전체 스코어링 결과 생성 (Outer Join 방식)"""
        logger.info("스코어링 결과 생성 시작")
        
        # 모든 기업 목록 가져오기 (Outer Join을 위해)
        all_companies = set(self.company_info.keys())
        
        # 문서 수 기준 기업 필터링 (참고용)
        valid_companies = self.filter_companies_by_document_count(documents)
        
        if not documents:
            logger.warning("문서가 없습니다. 모든 기업의 점수를 0으로 설정합니다.")
            # 모든 기업의 점수를 0으로 설정
            zero_scores = {}
            for company_symbol in all_companies:
                zero_scores[company_symbol] = {keyword: 0.0 for keyword in self.target_keywords}
            
            return self._create_results_with_all_companies(zero_scores, valid_companies, documents)
        
        # BM25 스코어 계산
        bm25_scores = self.calculate_bm25_scores(documents)
        
        # 토픽 스코어 계산
        topic_scores = self.calculate_topic_scores(documents)
        
        # 최종 통합 스코어 계산
        final_scores = self.calculate_final_scores(bm25_scores, topic_scores)
        
        # Outer Join: 모든 기업을 포함하되, 데이터가 없는 기업은 0점으로 설정
        all_company_scores = {}
        for company_symbol in all_companies:
            if company_symbol in final_scores:
                # 데이터가 있는 기업
                all_company_scores[company_symbol] = final_scores[company_symbol]
            else:
                # 데이터가 없는 기업 (0점으로 설정)
                all_company_scores[company_symbol] = {
                    keyword: 0.0 for keyword in self.target_keywords
                }
        
        # 결과 구성
        results = self._create_results_with_all_companies(all_company_scores, valid_companies, documents)
        
        self.final_scores = all_company_scores
        self.scoring_metadata = results
        
        logger.info("스코어링 결과 생성 완료")
        return results
    
    def _create_results_with_all_companies(self, all_company_scores: Dict, valid_companies: Dict, documents: List[Dict]) -> Dict:
        """모든 기업을 포함한 결과 생성"""
        return {
            'scoring_date': datetime.now().isoformat(),
            'data_period': f"{self.config.rebalancing_months}개월",
            'total_companies': len(all_company_scores),
            'companies_with_data': len(valid_companies),
            'companies_without_data': len(all_company_scores) - len(valid_companies),
            'total_documents': len(documents),
            'target_keywords': self.target_keywords,
            'scoring_weights': {
                'bm25_weight': self.bm25_weight,
                'topic_weight': self.topic_weight
            },
            'company_scores': all_company_scores,
            'company_document_counts': valid_companies,
            'top_companies_per_keyword': self._get_top_companies_per_keyword(all_company_scores),
            'data_availability': {
                'companies_with_data': list(valid_companies.keys()),
                'companies_without_data': [
                    company for company in all_company_scores.keys() 
                    if company not in valid_companies
                ]
            },
            'model_statistics': {
                'bm25_stats': self.bm25_scorer.get_statistics() if hasattr(self.bm25_scorer, 'get_statistics') else {},
                'topic_stats': self.topic_modeler.get_model_statistics() if hasattr(self.topic_modeler, 'get_model_statistics') else {}
            }
        }
    
    def _get_top_companies_per_keyword(self, scores: Dict[str, Dict[str, float]], 
                                     top_k: int = 10) -> Dict[str, List[Dict]]:
        """키워드별 상위 기업 반환"""
        top_companies = {}
        
        for keyword in self.target_keywords:
            company_scores = []
            for company_symbol, keyword_scores in scores.items():
                if keyword in keyword_scores:
                    company_scores.append({
                        'company_symbol': company_symbol,
                        'score': keyword_scores[keyword]
                    })
            
            # 스코어 기준 정렬
            company_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # 상위 k개 선택
            top_companies[keyword] = company_scores[:top_k]
        
        return top_companies
    
    def save_results(self, results: Dict, filename_prefix: str = "scoring_results", 
                     execution_folder: str = None, period_folder: str = None):
        """결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 키워드 정보를 파일명에 포함
        keyword_info = self._create_keyword_filename_suffix()
        
        # 실행 폴더와 기간 폴더 설정
        if execution_folder and period_folder:
            base_path = f"results/{execution_folder}/{period_folder}"
            os.makedirs(base_path, exist_ok=True)
            file_prefix = f"{base_path}/{filename_prefix}_{keyword_info}"
        else:
            file_prefix = f"{filename_prefix}_{keyword_info}"
        
        # JSON 형태로 전체 결과 저장
        json_filename = f"{file_prefix}_{timestamp}.json"
        try:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"결과 JSON 저장 완료: {json_filename}")
        except Exception as e:
            logger.error(f"JSON 저장 실패: {str(e)}")
        
        # CSV 형태로 기업별 스코어 저장
        csv_filename = f"{file_prefix}_companies_{timestamp}.csv"
        try:
            self._save_company_scores_csv(results['company_scores'], csv_filename)
            logger.info(f"기업 스코어 CSV 저장 완료: {csv_filename}")
        except Exception as e:
            logger.error(f"CSV 저장 실패: {str(e)}")
        
        # 키워드별 상위 기업 저장
        ranking_filename = f"{file_prefix}_rankings_{timestamp}.csv"
        try:
            self._save_rankings_csv(results['top_companies_per_keyword'], ranking_filename)
            logger.info(f"순위 CSV 저장 완료: {ranking_filename}")
        except Exception as e:
            logger.error(f"순위 CSV 저장 실패: {str(e)}")
        
        # Excel 형태로 통합 결과 저장
        excel_filename = f"{file_prefix}_{timestamp}.xlsx"
        try:
            self._save_results_excel(results, excel_filename)
            logger.info(f"Excel 결과 저장 완료: {excel_filename}")
        except Exception as e:
            logger.error(f"Excel 저장 실패: {str(e)}")
        
        return {
            'json_file': json_filename,
            'csv_file': csv_filename,
            'ranking_file': ranking_filename,
            'excel_file': excel_filename
        }
    
    def _save_company_scores_csv(self, company_scores: Dict[str, Dict[str, float]], filename: str):
        """기업별 스코어를 CSV로 저장"""
        rows = []
        for company_symbol, scores in company_scores.items():
            row = {
                'company_symbol': company_symbol,
                'company_name': self.get_company_name(company_symbol)
            }
            row.update(scores)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False, encoding='utf-8')
    
    def _save_rankings_csv(self, rankings: Dict[str, List[Dict]], filename: str):
        """키워드별 순위를 CSV로 저장"""
        rows = []
        for keyword, companies in rankings.items():
            for rank, company in enumerate(companies, 1):
                rows.append({
                    'keyword': keyword,
                    'rank': rank,
                    'company_symbol': company['company_symbol'],
                    'company_name': self.get_company_name(company['company_symbol']),
                    'score': company['score']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False, encoding='utf-8')
    
    def _save_results_excel(self, results: Dict, filename: str):
        """결과를 Excel 파일로 저장"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 1. 기업별 스코어 시트
                if 'company_scores' in results:
                    company_scores_data = []
                    for company, scores in results['company_scores'].items():
                        row = {
                            'company_symbol': company,
                            'company_name': self.get_company_name(company),
                            **scores
                        }
                        company_scores_data.append(row)
                    
                    company_scores_df = pd.DataFrame(company_scores_data)
                    company_scores_df.to_excel(writer, sheet_name='Company_Scores', index=False)
                
                # 2. 키워드별 순위 시트
                if 'top_companies_per_keyword' in results:
                    rankings_rows = []
                    for keyword, companies in results['top_companies_per_keyword'].items():
                        for rank, company in enumerate(companies, 1):
                            rankings_rows.append({
                                'keyword': keyword,
                                'rank': rank,
                                'company_symbol': company['company_symbol'],
                                'company_name': self.get_company_name(company['company_symbol']),
                                'score': company['score']
                            })
                    
                    rankings_df = pd.DataFrame(rankings_rows)
                    rankings_df.to_excel(writer, sheet_name='Keyword_Rankings', index=False)
                
                # 3. 리밸런싱 추천 시트
                recommendations = self.get_rebalancing_recommendations(top_n=50)
                if recommendations:
                    rec_df = pd.DataFrame(recommendations)
                    rec_df.to_excel(writer, sheet_name='Rebalancing_Recommendations', index=False)
                
                # 4. 메타데이터 시트
                metadata_rows = [
                    {'항목': '스코어링 날짜', '값': results.get('scoring_date', '')},
                    {'항목': '데이터 기간', '값': results.get('data_period', '')},
                    {'항목': '총 기업 수', '값': results.get('total_companies', 0)},
                    {'항목': '데이터 보유 기업 수', '값': results.get('companies_with_data', 0)},
                    {'항목': '데이터 미보유 기업 수', '값': results.get('companies_without_data', 0)},
                    {'항목': '총 문서 수', '값': results.get('total_documents', 0)},
                    {'항목': 'BM25 가중치', '값': results.get('scoring_weights', {}).get('bm25_weight', 0)},
                    {'항목': '토픽 가중치', '값': results.get('scoring_weights', {}).get('topic_weight', 0)},
                ]
                
                metadata_df = pd.DataFrame(metadata_rows)
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                # 5. 타겟 키워드 시트
                if 'target_keywords' in results:
                    keywords_df = pd.DataFrame({
                        'keyword': results['target_keywords'],
                        'rank': range(1, len(results['target_keywords']) + 1)
                    })
                    keywords_df.to_excel(writer, sheet_name='Target_Keywords', index=False)
                
                # 6. 데이터 가용성 시트
                self._create_data_availability_sheet(writer, results)
                
                # 7. 키워드별 파생 키워드 분석 시트
                self._create_keyword_analysis_sheet(writer, results)
            
            logger.info(f"Excel 파일 저장 완료: {filename}")
            
        except Exception as e:
            logger.error(f"Excel 저장 중 오류: {str(e)}")
            raise
    
    def _create_data_availability_sheet(self, writer, results: Dict):
        """데이터 가용성 시트 생성"""
        try:
            logger.info("데이터 가용성 시트 생성 시작")
            
            # 데이터 보유 기업 목록
            companies_with_data = results.get('data_availability', {}).get('companies_with_data', [])
            companies_without_data = results.get('data_availability', {}).get('companies_without_data', [])
            
            # 데이터 보유 기업 정보
            with_data_rows = []
            for company_symbol in companies_with_data:
                doc_count = results.get('company_document_counts', {}).get(company_symbol, 0)
                with_data_rows.append({
                    'company_symbol': company_symbol,
                    'company_name': self.get_company_name(company_symbol),
                    'document_count': doc_count,
                    'data_status': '보유'
                })
            
            # 데이터 미보유 기업 정보
            without_data_rows = []
            for company_symbol in companies_without_data:
                without_data_rows.append({
                    'company_symbol': company_symbol,
                    'company_name': self.get_company_name(company_symbol),
                    'document_count': 0,
                    'data_status': '미보유'
                })
            
            # 통합 데이터
            all_data_rows = with_data_rows + without_data_rows
            
            if all_data_rows:
                availability_df = pd.DataFrame(all_data_rows)
                availability_df.to_excel(writer, sheet_name='Data_Availability', index=False)
                logger.info(f"데이터 가용성 시트 생성 완료: {len(all_data_rows)}개 기업")
            
        except Exception as e:
            logger.error(f"데이터 가용성 시트 생성 실패: {str(e)}")
    
    def _create_keyword_analysis_sheet(self, writer, results: Dict):
        """키워드별 파생 키워드 분석 시트 생성"""
        try:
            logger.info("키워드별 파생 키워드 분석 시트 생성 시작")
            
            # 키워드별 파생 키워드 정보 생성
            keyword_analysis_data = []
            
            for keyword in self.target_keywords:
                # 파생 키워드 가져오기
                related_keywords = self._get_related_keywords(keyword)
                
                # 키워드 패턴 파일에서 더 상세한 파생 키워드 가져오기
                detailed_keywords = self._get_detailed_keywords(keyword)
                
                # 토픽 매핑 정보 가져오기
                topic_mapping = self._create_keyword_topic_mapping()
                mapped_topics = topic_mapping.get(keyword, [])
                
                # 키워드별 상위 기업 정보
                top_companies = results.get('top_companies_per_keyword', {}).get(keyword, [])
                top_company_count = len(top_companies)
                avg_score = np.mean([c['score'] for c in top_companies]) if top_companies else 0.0
                
                keyword_analysis_data.append({
                    'main_keyword': keyword,
                    'related_keywords_count': len(related_keywords),
                    'detailed_keywords_count': len(detailed_keywords),
                    'mapped_topics_count': len(mapped_topics),
                    'top_companies_count': top_company_count,
                    'average_score': round(avg_score, 3),
                    'related_keywords': ', '.join(related_keywords[:10]),  # 처음 10개만 표시
                    'detailed_keywords': ', '.join(detailed_keywords[:10]),  # 처음 10개만 표시
                    'mapped_topics': ', '.join(mapped_topics[:5])  # 처음 5개만 표시
                })
            
            # DataFrame 생성 및 저장
            analysis_df = pd.DataFrame(keyword_analysis_data)
            analysis_df.to_excel(writer, sheet_name='Keyword_Analysis', index=False)
            
            # 키워드별 상세 파생 키워드 시트들 생성
            for keyword in self.target_keywords:
                self._create_detailed_keyword_sheet(writer, keyword, results)
            
            logger.info("키워드별 파생 키워드 분석 시트 생성 완료")
            
        except Exception as e:
            logger.error(f"키워드 분석 시트 생성 실패: {str(e)}")
    
    def _get_detailed_keywords(self, keyword: str) -> List[str]:
        """키워드 패턴 파일에서 상세한 파생 키워드 가져오기"""
        try:
            # 키워드 패턴 파일 읽기
            pattern_file = "data/keyword_patterns.txt"
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split(':')
                        if len(parts) == 2 and parts[0] == keyword:
                            return [k.strip() for k in parts[1].split(',') if k.strip()]
            
            # 파일이 없거나 키워드가 없으면 기본 관련 키워드 반환
            return self._get_related_keywords(keyword)
            
        except Exception as e:
            logger.error(f"상세 키워드 가져오기 실패: {str(e)}")
            return self._get_related_keywords(keyword)
    
    def _create_detailed_keyword_sheet(self, writer, keyword: str, results: Dict):
        """키워드별 상세 파생 키워드 시트 생성"""
        try:
            # 파생 키워드들 가져오기
            related_keywords = self._get_related_keywords(keyword)
            detailed_keywords = self._get_detailed_keywords(keyword)
            
            # 토픽 매핑 정보
            topic_mapping = self._create_keyword_topic_mapping()
            mapped_topics = topic_mapping.get(keyword, [])
            
            # 키워드별 상위 기업 정보
            top_companies = results.get('top_companies_per_keyword', {}).get(keyword, [])
            
            # 상세 정보 데이터 생성
            detailed_data = []
            
            # 파생 키워드 정보
            for i, related_kw in enumerate(related_keywords):
                detailed_data.append({
                    'category': 'Related Keywords',
                    'keyword': related_kw,
                    'rank': i + 1,
                    'description': f'{keyword} 관련 키워드',
                    'type': 'Derived'
                })
            
            # 상세 파생 키워드 정보 (중복 제거)
            for i, detailed_kw in enumerate(detailed_keywords):
                if detailed_kw not in related_keywords:  # 중복 제거
                    detailed_data.append({
                        'category': 'Detailed Keywords',
                        'keyword': detailed_kw,
                        'rank': len(related_keywords) + i + 1,
                        'description': f'{keyword} 상세 관련 키워드',
                        'type': 'Detailed'
                    })
            
            # 토픽 매핑 정보
            for i, topic in enumerate(mapped_topics):
                detailed_data.append({
                    'category': 'Mapped Topics',
                    'keyword': topic,
                    'rank': i + 1,
                    'description': f'{keyword}와 매핑된 토픽',
                    'type': 'Topic'
                })
            
            # 상위 기업 정보
            for i, company in enumerate(top_companies[:10]):  # 상위 10개만
                detailed_data.append({
                    'category': 'Top Companies',
                    'keyword': company['company_symbol'],
                    'company_name': self.get_company_name(company['company_symbol']),
                    'rank': i + 1,
                    'description': f'{keyword} 관련 상위 기업',
                    'type': 'Company',
                    'score': company['score']
                })
            
            # DataFrame 생성 및 저장
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                # 시트명 생성 (Excel 시트명 제한 고려)
                sheet_name = f"KW_{keyword}"[:31]
                detailed_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
        except Exception as e:
            logger.error(f"키워드 {keyword} 상세 시트 생성 실패: {str(e)}")
    
    def _create_keyword_filename_suffix(self) -> str:
        """키워드 정보를 파일명 접미사로 생성"""
        try:
            if not self.target_keywords:
                return "no_keywords"
            
            # 키워드 개수에 따라 접미사 생성
            if len(self.target_keywords) == 1:
                # 단일 키워드인 경우
                keyword = self.target_keywords[0]
                return f"KW_{keyword}"
            elif len(self.target_keywords) <= 3:
                # 3개 이하인 경우 모든 키워드 포함
                keywords_str = "_".join(self.target_keywords)
                return f"KW_{keywords_str}"
            else:
                # 3개 초과인 경우 첫 3개만 포함하고 개수 표시
                first_three = "_".join(self.target_keywords[:3])
                return f"KW_{first_three}_and_{len(self.target_keywords)-3}more"
                
        except Exception as e:
            logger.error(f"키워드 파일명 접미사 생성 실패: {str(e)}")
            return "keywords_error"
    
    def get_rebalancing_recommendations(self, top_n: int = 50) -> List[Dict]:
        """리밸런싱 추천 기업 반환"""
        if not self.final_scores:
            logger.warning("스코어링 결과가 없습니다")
            return []
        
        logger.info(f"리밸런싱 추천 기업 생성: 상위 {top_n}개")
        
        # 기업별 평균 스코어 계산
        company_avg_scores = []
        for company_symbol, scores in self.final_scores.items():
            avg_score = np.mean(list(scores.values()))
            company_avg_scores.append({
                'company_symbol': company_symbol,
                'company_name': self.get_company_name(company_symbol),
                'average_score': avg_score,
                'max_score': max(scores.values()),
                'keyword_count': len([s for s in scores.values() if s > 0.1])
            })
        
        # 평균 스코어 기준 정렬
        company_avg_scores.sort(key=lambda x: x['average_score'], reverse=True)
        
        # 상위 n개 반환
        recommendations = company_avg_scores[:top_n]
        
        logger.info(f"리밸런싱 추천 완료: {len(recommendations)}개 기업")
        return recommendations
