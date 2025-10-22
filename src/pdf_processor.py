"""
PDF 텍스트 추출 및 전처리 모듈
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
    """PDF 텍스트 추출 프로세서"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
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
    
    def extract_text_from_pdf(self, url: str) -> ExtractedText:
        """PDF에서 텍스트 추출 (다중 방법 시도)"""
        logger.info(f"PDF 텍스트 추출 시작: {url}")
        
        # PDF 다운로드
        pdf_content = self.download_pdf(url)
        if not pdf_content:
            return ExtractedText(
                document_id="",
                title="",
                raw_text="",
                processed_text="",
                extraction_method="download_failed",
                confidence_score=0.0,
                page_count=0,
                error_message="PDF 다운로드 실패"
            )
        
        # 다양한 방법으로 텍스트 추출 시도
        extraction_methods = [
            ("pdfplumber", self.extract_text_pdfplumber),
            ("pypdf2", self.extract_text_pypdf2),
            ("ocr", self.extract_text_ocr)
        ]
        
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
        
        return ExtractedText(
            document_id="",
            title="",
            raw_text=best_text,
            processed_text=processed_text,
            extraction_method=best_method,
            confidence_score=best_confidence,
            page_count=best_page_count
        )
    
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
