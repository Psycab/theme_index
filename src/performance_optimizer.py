"""
성능 최적화 모듈
numba, joblib, multiprocessing을 사용한 실행 속도 향상
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
import time
from functools import wraps
import numba
from numba import jit, prange
import warnings

logger = logging.getLogger(__name__)

# numba 경고 무시
warnings.filterwarnings('ignore', category=numba.errors.NumbaDeprecationWarning)

class PerformanceOptimizer:
    """성능 최적화 클래스"""
    
    def __init__(self, n_jobs: int = -1, use_numba: bool = True):
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.use_numba = use_numba
        
        logger.info(f"성능 최적화 초기화: CPU 코어 {self.n_jobs}개 사용")
        if self.use_numba:
            logger.info("Numba JIT 컴파일 활성화")
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def calculate_bm25_scores_numba(doc_lengths: np.ndarray, 
                                 term_freqs: np.ndarray,
                                 doc_freqs: np.ndarray,
                                 k1: float = 1.2, 
                                 b: float = 0.75) -> np.ndarray:
        """Numba로 최적화된 BM25 스코어 계산"""
        n_docs = len(doc_lengths)
        n_terms = len(doc_freqs)
        avg_doc_length = np.mean(doc_lengths)
        
        scores = np.zeros((n_docs, n_terms))
        
        for i in prange(n_docs):
            doc_length = doc_lengths[i]
            for j in prange(n_terms):
                if term_freqs[i, j] > 0:
                    idf = np.log((n_docs - doc_freqs[j] + 0.5) / (doc_freqs[j] + 0.5))
                    tf = term_freqs[i, j]
                    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                    scores[i, j] = score
        
        return scores
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def calculate_tfidf_scores_numba(term_freqs: np.ndarray, 
                                  doc_freqs: np.ndarray) -> np.ndarray:
        """Numba로 최적화된 TF-IDF 스코어 계산"""
        n_docs, n_terms = term_freqs.shape
        tfidf_scores = np.zeros((n_docs, n_terms))
        
        for i in prange(n_docs):
            for j in prange(n_terms):
                if term_freqs[i, j] > 0:
                    tf = term_freqs[i, j]
                    idf = np.log(n_docs / doc_freqs[j])
                    tfidf_scores[i, j] = tf * idf
        
        return tfidf_scores
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def normalize_matrix_numba(matrix: np.ndarray) -> np.ndarray:
        """Numba로 최적화된 행렬 정규화"""
        n_rows, n_cols = matrix.shape
        normalized = np.zeros_like(matrix)
        
        for i in prange(n_rows):
            row_sum = np.sum(matrix[i, :])
            if row_sum > 0:
                normalized[i, :] = matrix[i, :] / row_sum
        
        return normalized
    
    def parallel_process_documents(self, documents: List[Dict], 
                                 process_func: Callable,
                                 batch_size: int = 100) -> List:
        """문서들을 병렬로 처리"""
        logger.info(f"병렬 문서 처리 시작: {len(documents)}개 문서, 배치 크기: {batch_size}")
        
        # 문서를 배치로 나누기
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        # 병렬 처리
        results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(process_func)(batch) for batch in batches
        )
        
        # 결과 합치기
        final_results = []
        for batch_result in results:
            final_results.extend(batch_result)
        
        logger.info(f"병렬 문서 처리 완료: {len(final_results)}개 결과")
        return final_results
    
    def parallel_extract_texts(self, documents: List[Dict], 
                             pdf_processor, batch_size: int = 50) -> List:
        """PDF 텍스트 추출을 병렬로 처리"""
        logger.info(f"병렬 PDF 텍스트 추출 시작: {len(documents)}개 문서")
        
        def extract_batch(batch_docs):
            results = []
            for doc in batch_docs:
                try:
                    extracted = pdf_processor.extract_text_from_pdf(doc['url'])
                    extracted.document_id = doc.get('document_id', '')
                    extracted.title = doc.get('title', '')
                    results.append(extracted)
                except Exception as e:
                    logger.warning(f"PDF 추출 실패: {doc.get('url', '')}, 오류: {str(e)}")
                    continue
            return results
        
        return self.parallel_process_documents(documents, extract_batch, batch_size)
    
    def parallel_preprocess_texts(self, extracted_texts: List, 
                                text_preprocessor, batch_size: int = 200) -> List:
        """텍스트 전처리를 병렬로 처리"""
        logger.info(f"병렬 텍스트 전처리 시작: {len(extracted_texts)}개 텍스트")
        
        def preprocess_batch(batch_texts):
            results = []
            for text_obj in batch_texts:
                try:
                    processed = text_preprocessor.preprocess_document(text_obj.raw_text)
                    processed['document_id'] = text_obj.document_id
                    processed['title'] = text_obj.title
                    processed['company_symbol'] = getattr(text_obj, 'company_symbol', '')
                    processed['publish_date'] = getattr(text_obj, 'publish_date', '')
                    processed['document_type'] = getattr(text_obj, 'document_type', '')
                    results.append(processed)
                except Exception as e:
                    logger.warning(f"전처리 실패: {text_obj.document_id}, 오류: {str(e)}")
                    continue
            return results
        
        return self.parallel_process_documents(extracted_texts, preprocess_batch, batch_size)
    
    def optimize_bm25_calculation(self, documents: List[Dict], 
                                target_keywords: List[str]) -> Dict:
        """BM25 계산 최적화"""
        logger.info("BM25 계산 최적화 시작")
        
        if not self.use_numba:
            logger.info("Numba 비활성화, 기본 계산 사용")
            return self._calculate_bm25_basic(documents, target_keywords)
        
        try:
            # 문서 길이 계산
            doc_lengths = np.array([len(doc.get('filtered_tokens', [])) for doc in documents])
            
            # 용어 빈도 행렬 생성
            all_tokens = set()
            for doc in documents:
                tokens = doc.get('filtered_tokens', [])
                all_tokens.update(tokens)
            
            token_list = list(all_tokens)
            n_docs = len(documents)
            n_terms = len(token_list)
            
            # 용어 빈도 행렬
            term_freqs = np.zeros((n_docs, n_terms), dtype=np.float32)
            doc_freqs = np.zeros(n_terms, dtype=np.float32)
            
            for i, doc in enumerate(documents):
                tokens = doc.get('filtered_tokens', [])
                token_counts = {}
                for token in tokens:
                    if token in token_list:
                        idx = token_list.index(token)
                        token_counts[idx] = token_counts.get(idx, 0) + 1
                
                for idx, count in token_counts.items():
                    term_freqs[i, idx] = count
                    doc_freqs[idx] += 1
            
            # Numba로 최적화된 BM25 계산
            scores = self.calculate_bm25_scores_numba(doc_lengths, term_freqs, doc_freqs)
            
            # 결과 변환
            keyword_scores = {}
            for keyword in target_keywords:
                if keyword in token_list:
                    idx = token_list.index(keyword)
                    keyword_scores[keyword] = scores[:, idx].tolist()
                else:
                    keyword_scores[keyword] = [0.0] * n_docs
            
            logger.info("BM25 계산 최적화 완료")
            return keyword_scores
            
        except Exception as e:
            logger.error(f"BM25 최적화 실패, 기본 방법 사용: {str(e)}")
            return self._calculate_bm25_basic(documents, target_keywords)
    
    def _calculate_bm25_basic(self, documents: List[Dict], 
                            target_keywords: List[str]) -> Dict:
        """기본 BM25 계산 (폴백)"""
        keyword_scores = {}
        
        for keyword in target_keywords:
            scores = []
            for doc in documents:
                tokens = doc.get('filtered_tokens', [])
                score = tokens.count(keyword) / len(tokens) if tokens else 0
                scores.append(score)
            keyword_scores[keyword] = scores
        
        return keyword_scores
    
    def optimize_tfidf_calculation(self, documents: List[Dict]) -> np.ndarray:
        """TF-IDF 계산 최적화"""
        logger.info("TF-IDF 계산 최적화 시작")
        
        if not self.use_numba:
            logger.info("Numba 비활성화, 기본 계산 사용")
            return self._calculate_tfidf_basic(documents)
        
        try:
            # 모든 토큰 수집
            all_tokens = set()
            doc_tokens = []
            
            for doc in documents:
                tokens = doc.get('filtered_tokens', [])
                doc_tokens.append(tokens)
                all_tokens.update(tokens)
            
            token_list = list(all_tokens)
            n_docs = len(documents)
            n_terms = len(token_list)
            
            # 용어 빈도 행렬 생성
            term_freqs = np.zeros((n_docs, n_terms), dtype=np.float32)
            doc_freqs = np.zeros(n_terms, dtype=np.float32)
            
            for i, tokens in enumerate(doc_tokens):
                token_counts = {}
                for token in tokens:
                    if token in token_list:
                        idx = token_list.index(token)
                        token_counts[idx] = token_counts.get(idx, 0) + 1
                
                for idx, count in token_counts.items():
                    term_freqs[i, idx] = count
                    doc_freqs[idx] += 1
            
            # Numba로 최적화된 TF-IDF 계산
            tfidf_scores = self.calculate_tfidf_scores_numba(term_freqs, doc_freqs)
            
            logger.info("TF-IDF 계산 최적화 완료")
            return tfidf_scores
            
        except Exception as e:
            logger.error(f"TF-IDF 최적화 실패, 기본 방법 사용: {str(e)}")
            return self._calculate_tfidf_basic(documents)
    
    def _calculate_tfidf_basic(self, documents: List[Dict]) -> np.ndarray:
        """기본 TF-IDF 계산 (폴백)"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        corpus = []
        for doc in documents:
            tokens = doc.get('filtered_tokens', [])
            corpus.append(' '.join(tokens))
        
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        return tfidf_matrix.toarray()
    
    def optimize_nmf_calculation(self, tfidf_matrix: np.ndarray, 
                               n_topics: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """NMF 계산 최적화"""
        logger.info("NMF 계산 최적화 시작")
        
        try:
            from sklearn.decomposition import NMF
            
            # NMF 모델 생성
            nmf = NMF(n_components=n_topics, 
                     init='random', 
                     random_state=42,
                     max_iter=1000,
                     alpha=0.1,
                     l1_ratio=0.5)
            
            # 문서-토픽 행렬 학습
            doc_topic_matrix = nmf.fit_transform(tfidf_matrix)
            
            # 토픽-단어 행렬 추출
            topic_word_matrix = nmf.components_
            
            # Numba로 정규화 최적화
            if self.use_numba:
                doc_topic_matrix = self.normalize_matrix_numba(doc_topic_matrix)
                topic_word_matrix = self.normalize_matrix_numba(topic_word_matrix)
            else:
                from sklearn.preprocessing import normalize
                doc_topic_matrix = normalize(doc_topic_matrix, norm='l1', axis=1)
                topic_word_matrix = normalize(topic_word_matrix, norm='l1', axis=1)
            
            logger.info("NMF 계산 최적화 완료")
            return doc_topic_matrix, topic_word_matrix
            
        except Exception as e:
            logger.error(f"NMF 최적화 실패: {str(e)}")
            raise
    
    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        return {
            'n_jobs': self.n_jobs,
            'use_numba': self.use_numba,
            'cpu_count': mp.cpu_count(),
            'numba_version': numba.__version__ if self.use_numba else None
        }

def performance_timer(func):
    """성능 측정 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result
    
    return wrapper
