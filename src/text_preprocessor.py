"""
텍스트 전처리 모듈
추출된 텍스트의 정제, 토큰화, 불용어 제거 등
"""

import logging
import re
import nltk
from typing import List, Dict, Set, Tuple
import pandas as pd
from collections import Counter
import pickle
import os

# 한국어 처리 라이브러리 (선택적)
try:
    from konlpy.tag import Okt, Kkma
    KONLPY_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("KoNLPy 사용 가능")
except ImportError as e:
    Okt = None
    Kkma = None
    KONLPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"KoNLPy 사용 불가: {str(e)}. 기본 토큰화 사용")

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """텍스트 전처리 클래스"""
    
    def __init__(self):
        # 한국어 처리기 초기화 (사용 가능한 경우에만)
        if KONLPY_AVAILABLE:
            try:
                self.okt = Okt()
                self.kkma = Kkma()
                logger.info("한국어 처리기 초기화 완료")
            except Exception as e:
                logger.warning(f"한국어 처리기 초기화 실패: {str(e)}")
                self.okt = None
                self.kkma = None
        else:
            self.okt = None
            self.kkma = None
        
        # 불용어 리스트 초기화
        self.stopwords = self._load_stopwords()
        
        # 키워드 패턴 초기화
        self.keyword_patterns = self._load_keyword_patterns()
        
        # NLTK 데이터 다운로드 (필요시)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def _load_stopwords(self) -> Set[str]:
        """불용어 리스트 로드"""
        # 기본 불용어
        stopwords = {
            '이', '가', '을', '를', '에', '의', '로', '으로', '와', '과', '도', '는', '은',
            '에서', '부터', '까지', '에게', '한테', '께', '보다', '처럼', '같이', '만큼',
            '하고', '하며', '하면서', '하지만', '그러나', '그런데', '따라서', '그래서',
            '그리고', '또한', '또는', '혹은', '즉', '예를', '들면', '같은', '다른',
            '모든', '각', '각각', '여러', '많은', '적은', '큰', '작은', '좋은', '나쁜',
            '새로운', '오래된', '최근', '과거', '미래', '현재', '지금', '오늘', '어제',
            '내일', '년', '월', '일', '시', '분', '초', '원', '달러', '엔', '위안',
            '회사', '기업', '법인', '주식', '주가', '시장', '경제', '금융', '투자',
            '매출', '수익', '이익', '손실', '자산', '부채', '자본', '현금', '유동',
            '고정', '단기', '장기', '분기', '연간', '월간', '일간', '주간', '년도',
            '상반기', '하반기', '전년', '전년동기', '전분기', '전월', '전주', '전일',
            '증가', '감소', '상승', '하락', '성장', '축소', '확대', '축소', '개선',
            '악화', '향상', '저하', '발전', '퇴보', '진보', '혁신', '보수', '안정',
            '불안', '위험', '기회', '도전', '과제', '문제', '해결', '방안', '정책',
            '전략', '계획', '목표', '비전', '미션', '가치', '문화', '철학', '이념'
        }
        
        # 파일에서 추가 불용어 로드
        stopwords_file = 'data/stopwords.txt'
        if os.path.exists(stopwords_file):
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    additional_stopwords = set(line.strip() for line in f if line.strip())
                    stopwords.update(additional_stopwords)
                logger.info(f"추가 불용어 {len(additional_stopwords)}개 로드")
            except Exception as e:
                logger.warning(f"불용어 파일 로드 실패: {str(e)}")
        
        return stopwords
    
    def _load_keyword_patterns(self) -> Dict[str, List[str]]:
        """키워드 패턴 로드"""
        patterns = {
            '반도체': ['반도체', '칩', '메모리', 'DRAM', 'NAND', '플래시', 'CPU', 'GPU', 'AP', 'SoC'],
            'AI': ['AI', '인공지능', '머신러닝', '딥러닝', '신경망', '알고리즘', '데이터사이언스'],
            '배터리': ['배터리', '리튬', '전지', '에너지저장', 'ESS', '양극재', '음극재'],
            '전기차': ['전기차', 'EV', '하이브리드', '충전', '충전소', '배터리팩'],
            '신재생에너지': ['태양광', '풍력', '신재생', '재생에너지', '에너지전환'],
            '바이오': ['바이오', '생명공학', '의료기기', '제약', '신약', '임상'],
            '핀테크': ['핀테크', '금융기술', '모바일결제', '디지털뱅킹', '암호화폐'],
            '클라우드': ['클라우드', '클라우드컴퓨팅', 'SaaS', 'PaaS', 'IaaS'],
            '5G': ['5G', '무선통신', '모바일통신', '네트워크', '통신'],
            'IoT': ['IoT', '사물인터넷', '스마트', '연결', '센서'],
            '블록체인': ['블록체인', '암호화폐', '비트코인', '이더리움', 'NFT'],
            '메타버스': ['메타버스', '가상현실', 'VR', 'AR', 'MR', '증강현실']
        }
        
        # 파일에서 추가 패턴 로드
        patterns_file = 'data/keyword_patterns.txt'
        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            category, keywords = line.strip().split(':', 1)
                            patterns[category] = keywords.split(',')
                logger.info(f"키워드 패턴 로드 완료: {len(patterns)}개 카테고리")
            except Exception as e:
                logger.warning(f"키워드 패턴 파일 로드 실패: {str(e)}")
        
        return patterns
    
    def clean_text(self, text: str) -> str:
        """텍스트 기본 정리"""
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL 제거
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 이메일 제거
        text = re.sub(r'\S+@\S+', '', text)
        
        # 전화번호 제거
        text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '', text)
        
        # 특수 문자 정리
        text = re.sub(r'[^\w\s가-힣.,!?;:()\[\]{}""''-]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 구두점 제거
        text = re.sub(r'[.,!?;:]{2,}', '.', text)
        
        return text.strip()
    
    def tokenize_korean(self, text: str) -> List[str]:
        """한국어 텍스트 토큰화"""
        if not text:
            return []
        
        # KoNLPy 사용 가능한 경우
        if self.okt is not None:
            try:
                # 형태소 분석
                morphs = self.okt.morphs(text, stem=True)
                return morphs
            except Exception as e:
                logger.error(f"한국어 토큰화 실패: {str(e)}")
                # 폴백: 간단한 공백 분리
                return text.split()
        else:
            # KoNLPy 사용 불가능한 경우 기본 토큰화
            logger.warning("KoNLPy 사용 불가, 기본 토큰화 사용")
            return self._basic_tokenize(text)
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """기본 토큰화 (KoNLPy 없이)"""
        if not text:
            return []
        
        # 간단한 정규식 기반 토큰화
        # 한글, 영문, 숫자만 추출
        tokens = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """불용어 제거"""
        if not tokens:
            return []
        
        return [token for token in tokens if token not in self.stopwords and len(token) > 1]
    
    def extract_keywords(self, text: str) -> Dict[str, int]:
        """키워드 추출 및 빈도 계산"""
        if not text:
            return {}
        
        # 텍스트 정리
        cleaned_text = self.clean_text(text)
        
        # 토큰화
        tokens = self.tokenize_korean(cleaned_text)
        
        # 불용어 제거
        filtered_tokens = self.remove_stopwords(tokens)
        
        # 키워드 카테고리별 매칭
        keyword_counts = {}
        
        for category, patterns in self.keyword_patterns.items():
            count = 0
            for pattern in patterns:
                # 정확한 매칭
                count += sum(1 for token in filtered_tokens if pattern.lower() in token.lower())
                
                # 부분 매칭 (긴 키워드의 경우)
                if len(pattern) > 3:
                    count += sum(1 for token in filtered_tokens if pattern in token)
            
            if count > 0:
                keyword_counts[category] = count
        
        return keyword_counts
    
    def preprocess_document(self, text: str) -> Dict:
        """문서 전체 전처리"""
        if not text:
            return {
                'cleaned_text': '',
                'tokens': [],
                'filtered_tokens': [],
                'keywords': {},
                'word_count': 0,
                'keyword_count': 0
            }
        
        # 텍스트 정리
        cleaned_text = self.clean_text(text)
        
        # 토큰화
        tokens = self.tokenize_korean(cleaned_text)
        
        # 불용어 제거
        filtered_tokens = self.remove_stopwords(tokens)
        
        # 키워드 추출
        keywords = self.extract_keywords(text)
        
        return {
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'keywords': keywords,
            'word_count': len(filtered_tokens),
            'keyword_count': sum(keywords.values())
        }
    
    def batch_preprocess(self, documents: List[Dict]) -> List[Dict]:
        """여러 문서 배치 전처리"""
        logger.info(f"{len(documents)}개 문서 전처리 시작")
        
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                text = doc.get('raw_text', '')
                processed = self.preprocess_document(text)
                
                # 원본 문서 정보 유지
                processed_doc = {
                    'document_id': doc.get('document_id', ''),
                    'title': doc.get('title', ''),
                    'company_symbol': doc.get('company_symbol', ''),
                    'publish_date': doc.get('publish_date', ''),
                    'document_type': doc.get('document_type', ''),
                    **processed
                }
                
                processed_docs.append(processed_doc)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"전처리 진행: {i + 1}/{len(documents)}")
                
            except Exception as e:
                logger.error(f"문서 {i} 전처리 실패: {str(e)}")
                continue
        
        logger.info(f"전처리 완료: {len(processed_docs)}개 문서")
        return processed_docs
    
    def save_preprocessed_data(self, processed_docs: List[Dict], filename: str):
        """전처리된 데이터 저장"""
        try:
            df = pd.DataFrame(processed_docs)
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"전처리 데이터 저장 완료: {filename}")
        except Exception as e:
            logger.error(f"데이터 저장 실패: {str(e)}")
    
    def load_preprocessed_data(self, filename: str) -> List[Dict]:
        """전처리된 데이터 로드"""
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"데이터 로드 실패: {str(e)}")
            return []
