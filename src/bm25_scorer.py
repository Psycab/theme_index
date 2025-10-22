"""
BM25 스코어링 시스템
키워드와 문서 간의 연관도 점수 계산
"""

import logging
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
import pandas as pd
from collections import defaultdict
import pickle
import os
from src.performance_optimizer import PerformanceOptimizer, performance_timer

logger = logging.getLogger(__name__)

class BM25Scorer:
    """BM25 스코어링 클래스"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, use_optimization: bool = True):
        self.k1 = k1  # BM25 파라미터
        self.b = b    # BM25 파라미터
        self.bm25_index = None
        self.document_ids = []
        self.company_symbols = []
        self.keyword_scores = {}
        
        # 성능 최적화 초기화
        self.use_optimization = use_optimization
        if self.use_optimization:
            self.optimizer = PerformanceOptimizer(use_numba=True)
            logger.info("BM25 스코어러 성능 최적화 활성화")
        else:
            self.optimizer = None
            logger.info("BM25 스코어러 기본 모드")
        
    def build_index(self, documents: List[Dict]) -> bool:
        """BM25 인덱스 구축"""
        logger.info(f"BM25 인덱스 구축 시작: {len(documents)}개 문서")
        
        if not documents:
            logger.error("문서가 없습니다")
            return False
        
        # 문서 텍스트 추출
        corpus = []
        doc_ids = []
        company_symbols = []
        
        for doc in documents:
            # 전처리된 텍스트 사용
            text = doc.get('filtered_tokens', [])
            if isinstance(text, str):
                text = text.split()
            
            if text and len(text) > 0:
                corpus.append(text)
                doc_ids.append(doc.get('document_id', ''))
                company_symbols.append(doc.get('company_symbol', ''))
        
        if not corpus:
            logger.error("유효한 텍스트가 없습니다")
            return False
        
        # BM25 인덱스 구축
        try:
            self.bm25_index = BM25Okapi(corpus, k1=self.k1, b=self.b)
            self.document_ids = doc_ids
            self.company_symbols = company_symbols
            
            logger.info(f"BM25 인덱스 구축 완료: {len(corpus)}개 문서")
            return True
            
        except Exception as e:
            logger.error(f"BM25 인덱스 구축 실패: {str(e)}")
            return False
    
    @performance_timer
    def calculate_keyword_scores(self, keywords: List[str]) -> Dict[str, List[float]]:
        """키워드별 문서 스코어 계산 (성능 최적화)"""
        if not self.bm25_index or not keywords:
            return {}
        
        logger.info(f"키워드 스코어 계산 시작: {len(keywords)}개 키워드")
        
        # 성능 최적화 사용 여부 확인
        if self.use_optimization and self.optimizer:
            return self._calculate_keyword_scores_optimized(keywords)
        else:
            return self._calculate_keyword_scores_basic(keywords)
    
    def _calculate_keyword_scores_optimized(self, keywords: List[str]) -> Dict[str, List[float]]:
        """최적화된 키워드 스코어 계산"""
        try:
            # 병렬 처리로 키워드별 스코어 계산
            keyword_scores = {}
            
            def calculate_single_keyword(keyword):
                try:
                    # 키워드를 토큰 리스트로 변환
                    if isinstance(keyword, str):
                        query_tokens = keyword.split()
                    else:
                        query_tokens = keyword
                    
                    # BM25 스코어 계산
                    scores = self.bm25_index.get_scores(query_tokens)
                    
                    # 정규화 (0-1 범위)
                    if len(scores) > 0:
                        max_score = np.max(scores)
                        if max_score > 0:
                            scores = scores / max_score
                    
                    return keyword, scores.tolist()
                    
                except Exception as e:
                    logger.error(f"키워드 '{keyword}' 스코어 계산 실패: {str(e)}")
                    return keyword, []
            
            # 병렬 처리
            results = self.optimizer.parallel_process_documents(
                keywords, 
                lambda batch: [calculate_single_keyword(kw) for kw in batch],
                batch_size=10
            )
            
            # 결과 정리
            for keyword, scores in results:
                keyword_scores[keyword] = scores
            
            self.keyword_scores = keyword_scores
            logger.info("최적화된 키워드 스코어 계산 완료")
            
            return keyword_scores
            
        except Exception as e:
            logger.error(f"최적화된 계산 실패, 기본 방법 사용: {str(e)}")
            return self._calculate_keyword_scores_basic(keywords)
    
    def _calculate_keyword_scores_basic(self, keywords: List[str]) -> Dict[str, List[float]]:
        """기본 키워드 스코어 계산"""
        keyword_scores = {}
        
        for keyword in keywords:
            try:
                # 키워드를 토큰 리스트로 변환
                if isinstance(keyword, str):
                    query_tokens = keyword.split()
                else:
                    query_tokens = keyword
                
                # BM25 스코어 계산
                scores = self.bm25_index.get_scores(query_tokens)
                
                # 정규화 (0-1 범위)
                if len(scores) > 0:
                    max_score = np.max(scores)
                    if max_score > 0:
                        scores = scores / max_score
                
                keyword_scores[keyword] = scores.tolist()
                
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 스코어 계산 실패: {str(e)}")
                keyword_scores[keyword] = []
        
        self.keyword_scores = keyword_scores
        logger.info("기본 키워드 스코어 계산 완료")
        
        return keyword_scores
    
    def get_company_keyword_scores(self, keywords: List[str]) -> Dict[str, Dict[str, float]]:
        """기업별 키워드 스코어 계산"""
        if not self.keyword_scores:
            self.calculate_keyword_scores(keywords)
        
        logger.info("기업별 키워드 스코어 계산 시작")
        
        company_scores = defaultdict(lambda: defaultdict(float))
        
        for keyword, scores in self.keyword_scores.items():
            if not scores:
                continue
            
            for i, score in enumerate(scores):
                if i < len(self.company_symbols):
                    company_symbol = self.company_symbols[i]
                    if company_symbol:
                        # 기업별 최대 스코어 유지 (여러 문서가 있는 경우)
                        company_scores[company_symbol][keyword] = max(
                            company_scores[company_symbol][keyword], 
                            score
                        )
        
        logger.info(f"기업별 키워드 스코어 계산 완료: {len(company_scores)}개 기업")
        return dict(company_scores)
    
    def get_top_documents(self, keyword: str, top_k: int = 10) -> List[Dict]:
        """특정 키워드에 대한 상위 문서 반환"""
        if not self.bm25_index or keyword not in self.keyword_scores:
            return []
        
        scores = self.keyword_scores[keyword]
        if not scores:
            return []
        
        # 스코어 기준으로 정렬
        doc_score_pairs = list(zip(self.document_ids, self.company_symbols, scores))
        doc_score_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 상위 k개 반환
        top_docs = []
        for i, (doc_id, company_symbol, score) in enumerate(doc_score_pairs[:top_k]):
            top_docs.append({
                'document_id': doc_id,
                'company_symbol': company_symbol,
                'score': score,
                'rank': i + 1
            })
        
        return top_docs
    
    def get_company_rankings(self, keyword: str) -> List[Dict]:
        """특정 키워드에 대한 기업 순위 반환"""
        company_scores = self.get_company_keyword_scores([keyword])
        
        if keyword not in company_scores:
            return []
        
        # 기업별 스코어 추출
        rankings = []
        for company_symbol, scores in company_scores.items():
            if keyword in scores:
                rankings.append({
                    'company_symbol': company_symbol,
                    'score': scores[keyword],
                    'keyword': keyword
                })
        
        # 스코어 기준으로 정렬
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        # 순위 추가
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def calculate_weighted_scores(self, keywords: List[str], 
                                weights: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """가중치를 적용한 통합 스코어 계산"""
        if not keywords:
            return {}
        
        # 기본 가중치 설정
        if weights is None:
            weights = {keyword: 1.0 for keyword in keywords}
        
        logger.info("가중치 적용 통합 스코어 계산 시작")
        
        # 키워드별 스코어 계산
        keyword_scores = self.calculate_keyword_scores(keywords)
        
        # 기업별 통합 스코어 계산
        company_scores = defaultdict(lambda: defaultdict(float))
        
        for keyword, scores in keyword_scores.items():
            weight = weights.get(keyword, 1.0)
            
            for i, score in enumerate(scores):
                if i < len(self.company_symbols):
                    company_symbol = self.company_symbols[i]
                    if company_symbol:
                        # 가중치 적용
                        weighted_score = score * weight
                        company_scores[company_symbol][keyword] = max(
                            company_scores[company_symbol][keyword], 
                            weighted_score
                        )
        
        # 정규화
        for company_symbol in company_scores:
            total_weight = sum(weights.get(k, 1.0) for k in keywords)
            if total_weight > 0:
                for keyword in keywords:
                    company_scores[company_symbol][keyword] /= total_weight
        
        logger.info("가중치 적용 통합 스코어 계산 완료")
        return dict(company_scores)
    
    def save_model(self, filepath: str):
        """BM25 모델 저장"""
        try:
            model_data = {
                'bm25_index': self.bm25_index,
                'document_ids': self.document_ids,
                'company_symbols': self.company_symbols,
                'keyword_scores': self.keyword_scores,
                'k1': self.k1,
                'b': self.b
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"BM25 모델 저장 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"BM25 모델 저장 실패: {str(e)}")
    
    def load_model(self, filepath: str) -> bool:
        """BM25 모델 로드"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"모델 파일이 존재하지 않습니다: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.bm25_index = model_data['bm25_index']
            self.document_ids = model_data['document_ids']
            self.company_symbols = model_data['company_symbols']
            self.keyword_scores = model_data['keyword_scores']
            self.k1 = model_data.get('k1', 1.2)
            self.b = model_data.get('b', 0.75)
            
            logger.info(f"BM25 모델 로드 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"BM25 모델 로드 실패: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict:
        """BM25 모델 통계 정보 반환"""
        if not self.bm25_index:
            return {}
        
        stats = {
            'total_documents': len(self.document_ids),
            'unique_companies': len(set(self.company_symbols)),
            'total_keywords': len(self.keyword_scores),
            'k1_parameter': self.k1,
            'b_parameter': self.b,
            'avg_document_length': np.mean([len(doc) for doc in self.bm25_index.corpus]) if self.bm25_index.corpus else 0
        }
        
        return stats
