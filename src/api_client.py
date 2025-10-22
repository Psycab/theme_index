"""
DeepSearch API 클라이언트 모듈
국내 상장기업 정보 및 리포트 데이터 수집
"""

import requests
import logging
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Company:
    """기업 정보 데이터 클래스"""
    symbol: str
    name: str
    sector: str
    market_cap: Optional[float] = None
    industry: Optional[str] = None

@dataclass
class Document:
    """문서 정보 데이터 클래스"""
    document_id: str
    title: str
    url: str
    company_symbol: str
    publish_date: datetime
    document_type: str
    content: Optional[str] = None

class DeepSearchClient:
    """DeepSearch API 클라이언트"""
    
    def __init__(self, config):
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.country_code = config.country_code
        self.page_size = config.page_size
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Company-Topic-Scoring-System/1.0',
            'Accept': 'application/json'
        })
        
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """API 요청 실행"""
        try:
            url = f"{self.base_url}{endpoint}"
            params['api_key'] = self.api_key
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패: {endpoint}, 오류: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"예상치 못한 오류: {str(e)}")
            return None
    
    def get_all_companies(self) -> List[Company]:
        """국내 상장기업 전체 목록 조회"""
        logger.info("국내 상장기업 목록 조회 시작")
        
        companies = []
        page = 1
        
        while True:
            params = {
                'country_code': self.country_code,
                'page_size': self.page_size,
                'page': page
            }
            
            data = self._make_request('/v2/companies', params)
            if not data or 'companies' not in data:
                break
                
            company_list = data['companies']
            if not company_list:
                break
                
            for company_data in company_list:
                try:
                    company = Company(
                        symbol=company_data.get('symbol', ''),
                        name=company_data.get('name', ''),
                        sector=company_data.get('sector', ''),
                        market_cap=company_data.get('market_cap'),
                        industry=company_data.get('industry', '')
                    )
                    companies.append(company)
                except Exception as e:
                    logger.warning(f"기업 데이터 파싱 실패: {company_data}, 오류: {str(e)}")
            
            logger.info(f"페이지 {page} 처리 완료: {len(company_list)}개 기업")
            page += 1
            
            # API 호출 제한 방지
            time.sleep(0.1)
            
            # 최대 페이지 수 제한 (안전장치)
            if page > 1000:
                logger.warning("최대 페이지 수에 도달하여 조회 중단")
                break
        
        logger.info(f"총 {len(companies)}개 기업 조회 완료")
        return companies
    
    def get_company_documents(self, symbols: List[str], 
                            date_from: datetime, 
                            date_to: datetime) -> List[Document]:
        """특정 기업들의 문서 조회"""
        logger.info(f"{len(symbols)}개 기업의 문서 조회 시작: {date_from} ~ {date_to}")
        
        documents = []
        
        # 심볼을 배치로 나누어 처리 (API 제한 고려)
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            
            # 심볼 형식 변환 (KRX: prefix 추가)
            formatted_symbols = [f"KRX:{symbol}" for symbol in batch_symbols]
            symbols_str = ','.join(formatted_symbols)
            
            params = {
                'symbols': symbols_str,
                'date_from': date_from.strftime('%Y-%m-%d'),
                'date_to': date_to.strftime('%Y-%m-%d')
            }
            
            data = self._make_request('/v1/articles/documents/research', params)
            if not data or 'documents' not in data:
                logger.warning(f"배치 {i//batch_size + 1} 문서 조회 실패")
                continue
            
            document_list = data['documents']
            for doc_data in document_list:
                try:
                    # 심볼에서 KRX: prefix 제거
                    symbol = doc_data.get('symbol', '').replace('KRX:', '')
                    
                    document = Document(
                        document_id=doc_data.get('id', ''),
                        title=doc_data.get('title', ''),
                        url=doc_data.get('url', ''),
                        company_symbol=symbol,
                        publish_date=datetime.fromisoformat(
                            doc_data.get('publish_date', '').replace('Z', '+00:00')
                        ),
                        document_type=doc_data.get('type', 'research')
                    )
                    documents.append(document)
                    
                except Exception as e:
                    logger.warning(f"문서 데이터 파싱 실패: {doc_data}, 오류: {str(e)}")
            
            logger.info(f"배치 {i//batch_size + 1} 처리 완료: {len(document_list)}개 문서")
            
            # API 호출 제한 방지
            time.sleep(0.2)
        
        logger.info(f"총 {len(documents)}개 문서 조회 완료")
        return documents
    
    def get_recent_documents(self, months: int = 6) -> Tuple[List[Company], List[Document]]:
        """최근 N개월간의 기업 및 문서 데이터 조회"""
        logger.info(f"최근 {months}개월 데이터 수집 시작")
        
        # 기업 목록 조회
        companies = self.get_all_companies()
        if not companies:
            logger.error("기업 목록 조회 실패")
            return [], []
        
        # 날짜 범위 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        # 기업 심볼 추출
        symbols = [company.symbol for company in companies if company.symbol]
        
        # 문서 조회
        documents = self.get_company_documents(symbols, start_date, end_date)
        
        logger.info(f"데이터 수집 완료: {len(companies)}개 기업, {len(documents)}개 문서")
        return companies, documents
    
    def save_data_to_csv(self, companies: List[Company], 
                        documents: List[Document], 
                        filename_prefix: str = "data"):
        """데이터를 CSV 파일로 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 기업 데이터 저장
        if companies:
            companies_df = pd.DataFrame([
                {
                    'symbol': c.symbol,
                    'name': c.name,
                    'sector': c.sector,
                    'market_cap': c.market_cap,
                    'industry': c.industry
                }
                for c in companies
            ])
            companies_file = f"{filename_prefix}_companies_{timestamp}.csv"
            companies_df.to_csv(companies_file, index=False, encoding='utf-8')
            logger.info(f"기업 데이터 저장 완료: {companies_file}")
        
        # 문서 데이터 저장
        if documents:
            documents_df = pd.DataFrame([
                {
                    'document_id': d.document_id,
                    'title': d.title,
                    'url': d.url,
                    'company_symbol': d.company_symbol,
                    'publish_date': d.publish_date.isoformat(),
                    'document_type': d.document_type
                }
                for d in documents
            ])
            documents_file = f"{filename_prefix}_documents_{timestamp}.csv"
            documents_df.to_csv(documents_file, index=False, encoding='utf-8')
            logger.info(f"문서 데이터 저장 완료: {documents_file}")
    
    def save_data_to_excel(self, companies: List[Company], 
                          documents: List[Document], 
                          filename_prefix: str = "data"):
        """데이터를 Excel 파일로 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_file = f"{filename_prefix}_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 기업 데이터 저장
                if companies:
                    companies_df = pd.DataFrame([
                        {
                            'symbol': c.symbol,
                            'name': c.name,
                            'sector': c.sector,
                            'market_cap': c.market_cap,
                            'industry': c.industry
                        }
                        for c in companies
                    ])
                    companies_df.to_excel(writer, sheet_name='Companies', index=False)
                
                # 문서 데이터 저장
                if documents:
                    documents_df = pd.DataFrame([
                        {
                            'document_id': d.document_id,
                            'title': d.title,
                            'url': d.url,
                            'company_symbol': d.company_symbol,
                            'publish_date': d.publish_date.isoformat(),
                            'document_type': d.document_type
                        }
                        for d in documents
                    ])
                    documents_df.to_excel(writer, sheet_name='Documents', index=False)
            
            logger.info(f"Excel 데이터 저장 완료: {excel_file}")
            
        except Exception as e:
            logger.error(f"Excel 저장 실패: {str(e)}")
