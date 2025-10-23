"""
PDF 텍스트 추출 및 전처리 모듈 (GPU 가속 최적화 버전)
PDF 문서에서 고품질 텍스트 추출 및 전처리
"""

import logging
import requests
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import io
import re
from typing import List, Optional, Dict
from dataclasses import dataclass
import time
from urllib.parse import urlparse
import hashlib
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache
import threading
from queue import Queue
import json

# GPU OCR 라이브러리 (선택적)
try:
    import easyocr
    import torch
    GPU_OCR_AVAILABLE = torch.cuda.is_available()
    logger = logging.getLogger(__name__)
    if GPU_OCR_AVAILABLE:
        logger.info("GPU OCR 지원 가능 (CUDA 사용 가능)")
    else:
        logger.info("GPU OCR 지원 불가 (CUDA 사용 불가)")
except ImportError as e:
    easyocr = None
    torch = None
    GPU_OCR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"EasyOCR 또는 PyTorch가 설치되지 않음: {str(e)}. GPU OCR 비활성화")

logger = logging.getLogger(__name__)

@dataclass
class ExtractedText:
    """추출된 텍스트 데이터 클래스"""
    document_id: str
    title: str
    raw_text: str
    processed_text: str
    extraction_method: str
    confidence_score: float
    page_count: int
    error_message: Optional[str] = None

class PDFProcessor:
    """PDF 텍스트 추출 프로세서 (GPU 가속 최적화 버전)"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3, cache_dir: str = "cache/pdf_cache", 
                 use_gpu: bool = True, gpu_ocr_method: str = 'easyocr'):
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu and GPU_OCR_AVAILABLE
        self.gpu_ocr_method = gpu_ocr_method
        
        # 캐시 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)
        
        # 세션 재사용으로 성능 향상
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # GPU OCR 초기화
        self.gpu_ocr_reader = None
        if self.use_gpu:
            self._init_gpu_ocr()
        
        # 성능 통계
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'extraction_failures': 0,
            'gpu_ocr_used': 0,
            'cpu_ocr_used': 0,
            'total_time': 0.0
        }
        
        # 스레드 안전성을 위한 락
        self.stats_lock = threading.Lock()
        
        logger.info(f"GPU 가속 PDF 프로세서 초기화 완료 (캐시: {cache_dir}, GPU: {self.use_gpu})")
    
    def _init_gpu_ocr(self):
        """GPU OCR 초기화"""
        try:
            if self.gpu_ocr_method == 'easyocr' and easyocr:
                logger.info("EasyOCR GPU 초기화 중...")
                self.gpu_ocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)
                logger.info("EasyOCR GPU 초기화 완료")
            else:
                logger.warning(f"지원하지 않는 GPU OCR 방법: {self.gpu_ocr_method}")
                self.use_gpu = False
        except Exception as e:
            logger.error(f"GPU OCR 초기화 실패: {str(e)}")
            self.use_gpu = False
            self.gpu_ocr_reader = None
    
    def _get_cache_key(self, url: str) -> str:
        """URL을 기반으로 캐시 키 생성"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, url: str) -> str:
        """캐시 파일 경로 반환"""
        cache_key = self._get_cache_key(url)
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_from_cache(self, url: str) -> Optional[ExtractedText]:
        """캐시에서 데이터 로드"""
        try:
            cache_path = self._get_cache_path(url)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # 캐시 만료 확인 (24시간)
                if time.time() - cached_data.get('timestamp', 0) < 86400:
                    with self.stats_lock:
                        self.stats['cache_hits'] += 1
                    logger.debug(f"캐시 히트: {url}")
                    return cached_data['extracted_text']
                else:
                    # 만료된 캐시 삭제
                    os.remove(cache_path)
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {url}, 오류: {str(e)}")
        
        with self.stats_lock:
            self.stats['cache_misses'] += 1
        return None
    
    def _save_to_cache(self, url: str, extracted_text: ExtractedText):
        """캐시에 데이터 저장"""
        try:
            cache_path = self._get_cache_path(url)
            cache_data = {
                'extracted_text': extracted_text,
                'timestamp': time.time(),
                'url': url
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug(f"캐시 저장: {url}")
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {url}, 오류: {str(e)}")
    
    def download_pdf(self, url: str) -> Optional[bytes]:
        """PDF 파일 다운로드"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Content-Type 확인
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type:
                logger.warning(f"PDF가 아닌 파일: {url}, Content-Type: {content_type}")
                return None
                
            return response.content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"PDF 다운로드 실패: {url}, 오류: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"예상치 못한 오류: {str(e)}")
            return None
    
    def extract_text_pypdf2(self, pdf_content: bytes) -> tuple[str, int]:
        """PyPDF2를 사용한 텍스트 추출"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            page_count = len(pdf_reader.pages)
            
            for page_num in range(page_count):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"페이지 {page_num} 텍스트 추출 실패: {str(e)}")
                    continue
            
            return text.strip(), page_count
            
        except Exception as e:
            logger.error(f"PyPDF2 텍스트 추출 실패: {str(e)}")
            return "", 0
    
    def extract_text_pdfplumber(self, pdf_content: bytes) -> tuple[str, int]:
        """pdfplumber를 사용한 텍스트 추출"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                page_count = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"페이지 {page_num} 텍스트 추출 실패: {str(e)}")
                        continue
                
                return text.strip(), page_count
                
        except Exception as e:
            logger.error(f"pdfplumber 텍스트 추출 실패: {str(e)}")
            return "", 0
    
    def extract_text_ocr(self, pdf_content: bytes) -> tuple[str, int]:
        """OCR을 사용한 텍스트 추출 (이미지 기반 PDF용)"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            page_count = len(pdf_reader.pages)
            
            for page_num in range(page_count):
                try:
                    page = pdf_reader.pages[page_num]
                    
                    # 페이지를 이미지로 변환
                    page_obj = page
                    if hasattr(page_obj, 'get_images'):
                        # 이미지 추출 시도
                        images = page_obj.get_images()
                        if images:
                            # 첫 번째 이미지 처리
                            img_obj = images[0]
                            img_data = pdf_reader.get_object(img_obj[0])
                            img_bytes = img_data.get_data()
                            
                            # PIL Image로 변환
                            img = Image.open(io.BytesIO(img_bytes))
                            
                            # OCR 수행
                            page_text = pytesseract.image_to_string(img, lang='kor+eng')
                            if page_text:
                                text += page_text + "\n"
                    
                except Exception as e:
                    logger.warning(f"페이지 {page_num} OCR 실패: {str(e)}")
                    continue
            
            return text.strip(), page_count
            
        except Exception as e:
            logger.error(f"OCR 텍스트 추출 실패: {str(e)}")
            return "", 0
    
    def extract_text_easyocr_gpu(self, pdf_content: bytes) -> tuple[str, int]:
        """EasyOCR GPU를 사용한 텍스트 추출"""
        try:
            if not self.gpu_ocr_reader:
                logger.warning("GPU OCR 리더가 초기화되지 않음")
                return "", 0
            
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            page_count = len(pdf_reader.pages)
            
            for page_num in range(page_count):
                try:
                    page = pdf_reader.pages[page_num]
                    
                    # 페이지를 이미지로 변환
                    page_obj = page
                    if hasattr(page_obj, 'get_images'):
                        images = page_obj.get_images()
                        if images:
                            # 첫 번째 이미지 처리
                            img_obj = images[0]
                            img_data = pdf_reader.get_object(img_obj[0])
                            img_bytes = img_data.get_data()
                            
                            # PIL Image로 변환
                            img = Image.open(io.BytesIO(img_bytes))
                            
                            # EasyOCR GPU로 텍스트 추출
                            results = self.gpu_ocr_reader.readtext(img)
                            page_text = ' '.join([result[1] for result in results])
                            if page_text:
                                text += page_text + "\n"
                    
                except Exception as e:
                    logger.warning(f"페이지 {page_num} GPU OCR 실패: {str(e)}")
                    continue
            
            return text.strip(), page_count
            
        except Exception as e:
            logger.error(f"EasyOCR GPU 텍스트 추출 실패: {str(e)}")
            return "", 0
    
    def extract_text_from_pdf(self, url: str) -> ExtractedText:
        """PDF에서 텍스트 추출 (캐싱 + 다중 방법 시도)"""
        start_time = time.time()
        
        # 캐시에서 먼저 확인
        cached_result = self._load_from_cache(url)
        if cached_result:
            logger.debug(f"캐시에서 반환: {url}")
            return cached_result
        
        logger.info(f"PDF 텍스트 추출 시작: {url}")
        
        # PDF 다운로드
        pdf_content = self.download_pdf(url)
        if not pdf_content:
            result = ExtractedText(
                document_id="",
                title="",
                raw_text="",
                processed_text="",
                extraction_method="download_failed",
                confidence_score=0.0,
                page_count=0,
                error_message="PDF 다운로드 실패"
            )
            with self.stats_lock:
                self.stats['extraction_failures'] += 1
            return result
        
        # 다양한 방법으로 텍스트 추출 시도 (하이브리드 접근법)
        extraction_methods = [
            ("pdfplumber", self.extract_text_pdfplumber),
            ("pypdf2", self.extract_text_pypdf2),
            ("ocr", self.extract_text_ocr)
        ]
        
        # GPU OCR을 추출 방법에 추가 (사용 가능한 경우)
        if self.use_gpu and self.gpu_ocr_reader:
            extraction_methods.append(("easyocr_gpu", self.extract_text_easyocr_gpu))
        
        best_text = ""
        best_method = ""
        best_page_count = 0
        best_confidence = 0.0
        
        for method_name, extract_func in extraction_methods:
            try:
                text, page_count = extract_func(pdf_content)
                
                if text and len(text.strip()) > 50:  # 최소 텍스트 길이 확인
                    # 텍스트 품질 평가 (간단한 휴리스틱)
                    confidence = self._calculate_text_confidence(text)
                    
                    if confidence > best_confidence:
                        best_text = text
                        best_method = method_name
                        best_page_count = page_count
                        best_confidence = confidence
                        
                    logger.info(f"{method_name} 성공: {len(text)}자, 신뢰도: {confidence:.2f}")
                    
                    # GPU OCR 사용 통계 업데이트
                    if method_name == "easyocr_gpu":
                        with self.stats_lock:
                            self.stats['gpu_ocr_used'] += 1
                    elif method_name == "ocr":
                        with self.stats_lock:
                            self.stats['cpu_ocr_used'] += 1
                else:
                    logger.warning(f"{method_name} 결과 부족: {len(text)}자")
                    
            except Exception as e:
                logger.error(f"{method_name} 추출 실패: {str(e)}")
                continue
        
        if not best_text:
            return ExtractedText(
                document_id="",
                title="",
                raw_text="",
                processed_text="",
                extraction_method="all_failed",
                confidence_score=0.0,
                page_count=0,
                error_message="모든 추출 방법 실패"
            )
        
        # 텍스트 전처리
        processed_text = self._preprocess_text(best_text)
        
        result = ExtractedText(
            document_id="",
            title="",
            raw_text=best_text,
            processed_text=processed_text,
            extraction_method=best_method,
            confidence_score=best_confidence,
            page_count=best_page_count
        )
        
        # 캐시에 저장
        self._save_to_cache(url, result)
        
        # 통계 업데이트
        processing_time = time.time() - start_time
        with self.stats_lock:
            self.stats['total_processed'] += 1
            self.stats['total_time'] += processing_time
        
        logger.info(f"PDF 추출 완료: {url} ({best_method}, {processing_time:.2f}초)")
        return result
    
    def _calculate_text_confidence(self, text: str) -> float:
        """텍스트 품질 신뢰도 계산"""
        if not text:
            return 0.0
        
        confidence = 0.0
        
        # 텍스트 길이 점수 (0-0.3)
        text_length = len(text.strip())
        if text_length > 1000:
            confidence += 0.3
        elif text_length > 500:
            confidence += 0.2
        elif text_length > 100:
            confidence += 0.1
        
        # 한글 비율 점수 (0-0.3)
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.findall(r'[가-힣a-zA-Z0-9]', text))
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            confidence += korean_ratio * 0.3
        
        # 문장 구조 점수 (0-0.2)
        sentences = re.split(r'[.!?]', text)
        avg_sentence_length = sum(len(s.strip()) for s in sentences) / len(sentences) if sentences else 0
        if 10 <= avg_sentence_length <= 100:
            confidence += 0.2
        elif 5 <= avg_sentence_length <= 150:
            confidence += 0.1
        
        # 특수 문자 비율 점수 (0-0.2)
        special_chars = len(re.findall(r'[^\w\s가-힣]', text))
        if total_chars > 0:
            special_ratio = special_chars / total_chars
            if special_ratio < 0.1:  # 특수문자가 적을수록 좋음
                confidence += 0.2
            elif special_ratio < 0.2:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return ""
        
        # 기본 정리
        text = text.strip()
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 줄바꿈 제거
        text = re.sub(r'\n+', '\n', text)
        
        # 특수 문자 정리 (일부 유지)
        text = re.sub(r'[^\w\s가-힣.,!?;:()\[\]{}""''-]', ' ', text)
        
        # 연속된 특수 문자 제거
        text = re.sub(r'[.,!?;:]{2,}', '.', text)
        
        return text.strip()
    
    def batch_extract_texts(self, documents: List[Dict], 
                           max_workers: int = 5) -> List[ExtractedText]:
        """여러 문서의 텍스트를 배치로 추출"""
        logger.info(f"{len(documents)}개 문서 텍스트 추출 시작")
        
        extracted_texts = []
        
        for i, doc in enumerate(documents):
            try:
                url = doc.get('url', '')
                if not url:
                    logger.warning(f"문서 {i} URL 없음")
                    continue
                
                extracted = self.extract_text_from_pdf(url)
                extracted.document_id = doc.get('document_id', '')
                extracted.title = doc.get('title', '')
                
                extracted_texts.append(extracted)
                
                logger.info(f"문서 {i+1}/{len(documents)} 처리 완료: {extracted.extraction_method}")
                
                # API 제한 방지
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"문서 {i} 처리 실패: {str(e)}")
                continue
        
        logger.info(f"텍스트 추출 완료: {len(extracted_texts)}개 문서")
        return extracted_texts
    
    def high_performance_batch_extract(self, documents: List[Dict], 
                                     max_workers: int = None, 
                                     batch_size: int = 50) -> List[ExtractedText]:
        """고성능 병렬 PDF 텍스트 추출"""
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) * 4)  # CPU 코어의 4배
        
        logger.info(f"고성능 병렬 PDF 추출 시작: {len(documents)}개 문서, {max_workers}개 워커")
        start_time = time.time()
        
        extracted_texts = []
        
        # 배치로 나누어 처리
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 각 배치를 병렬로 처리
            future_to_batch = {
                executor.submit(self._process_batch, batch, batch_idx): batch_idx 
                for batch_idx, batch in enumerate(batches)
            }
            
            # 진행률 표시기로 결과 수집
            from tqdm import tqdm
            with tqdm(total=len(batches), desc="PDF 추출", unit="배치", 
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        extracted_texts.extend(batch_results)
                        pbar.set_postfix({
                            '문서수': len(batch_results),
                            '총문서': len(extracted_texts)
                        })
                        logger.info(f"📄 배치 {batch_idx + 1}/{len(batches)} 완료: {len(batch_results)}개 문서")
                    except Exception as e:
                        logger.error(f"❌ 배치 {batch_idx + 1} 처리 실패: {str(e)}")
                    
                    pbar.update(1)
        
        total_time = time.time() - start_time
        logger.info(f"고성능 병렬 추출 완료: {len(extracted_texts)}개 문서, {total_time:.2f}초")
        
        # 성능 통계 출력
        self._log_performance_stats()
        
        return extracted_texts
    
    def _process_batch(self, batch: List[Dict], batch_idx: int) -> List[ExtractedText]:
        """배치 단위로 PDF 처리"""
        batch_results = []
        
        for i, doc in enumerate(batch):
            try:
                url = doc.get('url', '')
                if not url:
                    logger.warning(f"배치 {batch_idx} 문서 {i} URL 없음")
                    continue
                
                extracted = self.extract_text_from_pdf(url)
                extracted.document_id = doc.get('document_id', '')
                extracted.title = doc.get('title', '')
                
                batch_results.append(extracted)
                
                # 적응적 지연 (API 제한 방지)
                if i < len(batch) - 1:  # 마지막 문서가 아닌 경우
                    time.sleep(0.05)  # 기존 0.1초에서 0.05초로 단축
                
            except Exception as e:
                logger.error(f"배치 {batch_idx} 문서 {i} 처리 실패: {str(e)}")
                continue
        
        return batch_results
    
    def _log_performance_stats(self):
        """성능 통계 로깅"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        if stats['total_processed'] > 0:
            avg_time = stats['total_time'] / stats['total_processed']
            cache_hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100
            
            logger.info("=== PDF 처리 성능 통계 ===")
            logger.info(f"총 처리 문서: {stats['total_processed']}개")
            logger.info(f"평균 처리 시간: {avg_time:.2f}초/문서")
            logger.info(f"캐시 히트율: {cache_hit_rate:.1f}% ({stats['cache_hits']}/{stats['cache_hits'] + stats['cache_misses']})")
            logger.info(f"추출 실패: {stats['extraction_failures']}개")
            logger.info(f"GPU OCR 사용: {stats['gpu_ocr_used']}개")
            logger.info(f"CPU OCR 사용: {stats['cpu_ocr_used']}개")
            if stats['gpu_ocr_used'] + stats['cpu_ocr_used'] > 0:
                gpu_ocr_ratio = stats['gpu_ocr_used'] / (stats['gpu_ocr_used'] + stats['cpu_ocr_used']) * 100
                logger.info(f"GPU OCR 비율: {gpu_ocr_ratio:.1f}%")
            logger.info("========================")
    
    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        if stats['total_processed'] > 0:
            stats['average_time'] = stats['total_time'] / stats['total_processed']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100
        else:
            stats['average_time'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def clear_cache(self):
        """캐시 삭제"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("PDF 캐시 삭제 완료")
        except Exception as e:
            logger.error(f"캐시 삭제 실패: {str(e)}")
