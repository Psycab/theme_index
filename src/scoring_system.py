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
        """전체 스코어링 결과 생성"""
        logger.info("스코어링 결과 생성 시작")
        
        # 문서 수 기준 기업 필터링
        valid_companies = self.filter_companies_by_document_count(documents)
        
        if not valid_companies:
            logger.error("유효한 기업이 없습니다")
            return {}
        
        # BM25 스코어 계산
        bm25_scores = self.calculate_bm25_scores(documents)
        
        # 토픽 스코어 계산
        topic_scores = self.calculate_topic_scores(documents)
        
        # 최종 통합 스코어 계산
        final_scores = self.calculate_final_scores(bm25_scores, topic_scores)
        
        # 유효한 기업만 필터링
        filtered_final_scores = {
            company: scores for company, scores in final_scores.items()
            if company in valid_companies
        }
        
        # 결과 구성
        results = {
            'scoring_date': datetime.now().isoformat(),
            'data_period': f"{self.config.rebalancing_months}개월",
            'total_companies': len(valid_companies),
            'total_documents': len(documents),
            'target_keywords': self.target_keywords,
            'scoring_weights': {
                'bm25_weight': self.bm25_weight,
                'topic_weight': self.topic_weight
            },
            'company_scores': filtered_final_scores,
            'bm25_scores': bm25_scores,
            'topic_scores': topic_scores,
            'company_document_counts': valid_companies,
            'top_companies_per_keyword': self._get_top_companies_per_keyword(filtered_final_scores),
            'model_statistics': {
                'bm25_stats': self.bm25_scorer.get_statistics(),
                'topic_stats': self.topic_modeler.get_model_statistics()
            }
        }
        
        self.final_scores = filtered_final_scores
        self.scoring_metadata = results
        
        logger.info("스코어링 결과 생성 완료")
        return results
    
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
        
        # 실행 폴더와 기간 폴더 설정
        if execution_folder and period_folder:
            base_path = f"results/{execution_folder}/{period_folder}"
            os.makedirs(base_path, exist_ok=True)
            file_prefix = f"{base_path}/{filename_prefix}"
        else:
            file_prefix = filename_prefix
        
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
            row = {'company_symbol': company_symbol}
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
                    company_scores_df = pd.DataFrame([
                        {'company_symbol': company, **scores}
                        for company, scores in results['company_scores'].items()
                    ])
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
            
            logger.info(f"Excel 파일 저장 완료: {filename}")
            
        except Exception as e:
            logger.error(f"Excel 저장 중 오류: {str(e)}")
            raise
    
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
