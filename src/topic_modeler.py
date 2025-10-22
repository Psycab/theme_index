"""
NMF 토픽 모델링 시스템
Non-negative Matrix Factorization을 사용한 토픽 추출 및 기업별 토픽 분포 계산
"""

import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple, Optional
import pickle
import os
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class TopicModeler:
    """NMF 토픽 모델링 클래스"""
    
    def __init__(self, n_topics: int = 20, max_features: int = 1000, 
                 min_df: int = 2, max_df: float = 0.95):
        self.n_topics = n_topics
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = None
        self.nmf_model = None
        self.topic_word_matrix = None
        self.document_topic_matrix = None
        self.topic_names = []
        self.document_ids = []
        self.company_symbols = []
        self.feature_names = []
        
    def prepare_corpus(self, documents: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
        """문서 코퍼스 준비"""
        logger.info(f"코퍼스 준비 시작: {len(documents)}개 문서")
        
        corpus = []
        doc_ids = []
        company_symbols = []
        
        for doc in documents:
            # 전처리된 텍스트 사용
            text = doc.get('filtered_tokens', [])
            if isinstance(text, str):
                text = text.split()
            
            if text and len(text) > 0:
                # 토큰 리스트를 문자열로 변환
                text_str = ' '.join(text)
                corpus.append(text_str)
                doc_ids.append(doc.get('document_id', ''))
                company_symbols.append(doc.get('company_symbol', ''))
        
        logger.info(f"코퍼스 준비 완료: {len(corpus)}개 문서")
        return corpus, doc_ids, company_symbols
    
    def build_tfidf_matrix(self, corpus: List[str]) -> np.ndarray:
        """TF-IDF 행렬 구축"""
        logger.info("TF-IDF 행렬 구축 시작")
        
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=(1, 2),  # 단어와 바이그램 사용
                stop_words=None,  # 이미 전처리에서 불용어 제거됨
                lowercase=False  # 한국어 대소문자 구분
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            logger.info(f"TF-IDF 행렬 구축 완료: {tfidf_matrix.shape}")
            return tfidf_matrix
            
        except Exception as e:
            logger.error(f"TF-IDF 행렬 구축 실패: {str(e)}")
            return None
    
    def fit_nmf_model(self, tfidf_matrix: np.ndarray) -> bool:
        """NMF 모델 학습"""
        logger.info(f"NMF 모델 학습 시작: {self.n_topics}개 토픽")
        
        try:
            self.nmf_model = NMF(
                n_components=self.n_topics,
                init='random',
                random_state=42,
                max_iter=1000,
                alpha=0.1,
                l1_ratio=0.5
            )
            
            # 문서-토픽 행렬 학습
            self.document_topic_matrix = self.nmf_model.fit_transform(tfidf_matrix)
            
            # 토픽-단어 행렬 추출
            self.topic_word_matrix = self.nmf_model.components_
            
            # 정규화
            self.document_topic_matrix = normalize(self.document_topic_matrix, norm='l1', axis=1)
            self.topic_word_matrix = normalize(self.topic_word_matrix, norm='l1', axis=1)
            
            logger.info("NMF 모델 학습 완료")
            return True
            
        except Exception as e:
            logger.error(f"NMF 모델 학습 실패: {str(e)}")
            return False
    
    def extract_topic_names(self) -> List[str]:
        """토픽 이름 추출"""
        if self.topic_word_matrix is None or not self.feature_names:
            return []
        
        logger.info("토픽 이름 추출 시작")
        
        topic_names = []
        
        for topic_idx in range(self.n_topics):
            # 각 토픽의 상위 단어들 추출
            topic_words = self.topic_word_matrix[topic_idx]
            top_word_indices = np.argsort(topic_words)[-10:][::-1]  # 상위 10개 단어
            
            top_words = [self.feature_names[idx] for idx in top_word_indices]
            
            # 토픽 이름 생성 (상위 3개 단어 조합)
            topic_name = '_'.join(top_words[:3])
            topic_names.append(topic_name)
        
        self.topic_names = topic_names
        logger.info(f"토픽 이름 추출 완료: {len(topic_names)}개 토픽")
        
        return topic_names
    
    def get_topic_keywords(self, topic_idx: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """특정 토픽의 상위 키워드 반환"""
        if (self.topic_word_matrix is None or 
            topic_idx >= self.n_topics or 
            not self.feature_names):
            return []
        
        topic_words = self.topic_word_matrix[topic_idx]
        top_word_indices = np.argsort(topic_words)[-top_k:][::-1]
        
        keywords = []
        for idx in top_word_indices:
            word = self.feature_names[idx]
            score = topic_words[idx]
            keywords.append((word, score))
        
        return keywords
    
    def get_document_topics(self, doc_idx: int) -> List[Tuple[int, float]]:
        """특정 문서의 토픽 분포 반환"""
        if (self.document_topic_matrix is None or 
            doc_idx >= len(self.document_topic_matrix)):
            return []
        
        doc_topics = self.document_topic_matrix[doc_idx]
        topic_scores = [(i, score) for i, score in enumerate(doc_topics) if score > 0.01]
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        
        return topic_scores
    
    def get_company_topic_distribution(self) -> Dict[str, Dict[str, float]]:
        """기업별 토픽 분포 계산"""
        if (self.document_topic_matrix is None or 
            not self.company_symbols):
            return {}
        
        logger.info("기업별 토픽 분포 계산 시작")
        
        company_topics = defaultdict(lambda: defaultdict(float))
        
        for doc_idx, company_symbol in enumerate(self.company_symbols):
            if doc_idx < len(self.document_topic_matrix):
                doc_topics = self.document_topic_matrix[doc_idx]
                
                for topic_idx, score in enumerate(doc_topics):
                    if score > 0.01:  # 임계값 이상만 고려
                        topic_name = self.topic_names[topic_idx] if topic_idx < len(self.topic_names) else f"topic_{topic_idx}"
                        company_topics[company_symbol][topic_name] = max(
                            company_topics[company_symbol][topic_name], 
                            score
                        )
        
        logger.info(f"기업별 토픽 분포 계산 완료: {len(company_topics)}개 기업")
        return dict(company_topics)
    
    def get_top_companies_per_topic(self, top_k: int = 10) -> Dict[str, List[Dict]]:
        """토픽별 상위 기업 반환"""
        company_topics = self.get_company_topic_distribution()
        
        topic_companies = defaultdict(list)
        
        for company_symbol, topics in company_topics.items():
            for topic_name, score in topics.items():
                topic_companies[topic_name].append({
                    'company_symbol': company_symbol,
                    'score': score
                })
        
        # 각 토픽별로 상위 k개 기업 선택
        for topic_name in topic_companies:
            topic_companies[topic_name].sort(key=lambda x: x['score'], reverse=True)
            topic_companies[topic_name] = topic_companies[topic_name][:top_k]
        
        return dict(topic_companies)
    
    def fit_model(self, documents: List[Dict]) -> bool:
        """전체 토픽 모델링 파이프라인 실행"""
        logger.info("토픽 모델링 파이프라인 시작")
        
        # 코퍼스 준비
        corpus, doc_ids, company_symbols = self.prepare_corpus(documents)
        
        if not corpus:
            logger.error("유효한 코퍼스가 없습니다")
            return False
        
        self.document_ids = doc_ids
        self.company_symbols = company_symbols
        
        # TF-IDF 행렬 구축
        tfidf_matrix = self.build_tfidf_matrix(corpus)
        if tfidf_matrix is None:
            return False
        
        # NMF 모델 학습
        if not self.fit_nmf_model(tfidf_matrix):
            return False
        
        # 토픽 이름 추출
        self.extract_topic_names()
        
        logger.info("토픽 모델링 파이프라인 완료")
        return True
    
    def save_model(self, filepath: str):
        """토픽 모델 저장"""
        try:
            model_data = {
                'nmf_model': self.nmf_model,
                'vectorizer': self.vectorizer,
                'topic_word_matrix': self.topic_word_matrix,
                'document_topic_matrix': self.document_topic_matrix,
                'topic_names': self.topic_names,
                'document_ids': self.document_ids,
                'company_symbols': self.company_symbols,
                'feature_names': self.feature_names,
                'n_topics': self.n_topics,
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"토픽 모델 저장 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"토픽 모델 저장 실패: {str(e)}")
    
    def load_model(self, filepath: str) -> bool:
        """토픽 모델 로드"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"모델 파일이 존재하지 않습니다: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.nmf_model = model_data['nmf_model']
            self.vectorizer = model_data['vectorizer']
            self.topic_word_matrix = model_data['topic_word_matrix']
            self.document_topic_matrix = model_data['document_topic_matrix']
            self.topic_names = model_data['topic_names']
            self.document_ids = model_data['document_ids']
            self.company_symbols = model_data['company_symbols']
            self.feature_names = model_data['feature_names']
            self.n_topics = model_data.get('n_topics', 20)
            self.max_features = model_data.get('max_features', 1000)
            self.min_df = model_data.get('min_df', 2)
            self.max_df = model_data.get('max_df', 0.95)
            
            logger.info(f"토픽 모델 로드 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"토픽 모델 로드 실패: {str(e)}")
            return False
    
    def get_model_statistics(self) -> Dict:
        """모델 통계 정보 반환"""
        if not self.nmf_model:
            return {}
        
        stats = {
            'n_topics': self.n_topics,
            'n_documents': len(self.document_ids),
            'n_companies': len(set(self.company_symbols)),
            'n_features': len(self.feature_names),
            'topic_names': self.topic_names,
            'avg_topic_coherence': 0.0,  # TODO: 일관성 점수 계산
            'model_perplexity': 0.0  # TODO: 복잡도 계산
        }
        
        return stats
    
    def print_topic_summary(self):
        """토픽 요약 정보 출력"""
        if not self.topic_names:
            logger.warning("토픽 이름이 없습니다")
            return
        
        logger.info("=== 토픽 요약 ===")
        for i, topic_name in enumerate(self.topic_names):
            keywords = self.get_topic_keywords(i, top_k=5)
            keyword_str = ', '.join([f"{word}({score:.3f})" for word, score in keywords])
            logger.info(f"토픽 {i+1}: {topic_name}")
            logger.info(f"  키워드: {keyword_str}")
        
        logger.info("================")
