"""뉴스 챗봇 도구 정의"""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# ============================================
# Mock News Database (RAG용)
# ============================================

NEWS_DATABASE = [
    {
        "id": "news_001",
        "title": "엔비디아, 4분기 실적 발표... AI 칩 수요 폭발",
        "date": "2025-01-28",
        "category": "경제",
        "source": "한국경제",
        "content": "엔비디아가 2024년 4분기 실적을 발표했다. 매출은 전년 대비 120% 증가한 220억 달러를 기록했으며, 이는 시장 예상치를 크게 상회하는 수치다. AI 칩 수요 증가가 실적 호조의 주요 원인으로 분석된다. 젠슨 황 CEO는 AI 시대의 본격적인 시작이라며 향후 전망에 대해 낙관적인 견해를 밝혔다."
    },
    {
        "id": "news_002",
        "title": "삼성전자, HBM3E 양산 본격화... 엔비디아 납품 확대",
        "date": "2025-01-27",
        "category": "경제",
        "source": "매일경제",
        "content": "삼성전자가 차세대 고대역폭메모리(HBM) HBM3E 양산을 본격화한다. 엔비디아 차세대 GPU에 탑재될 예정이며, SK하이닉스와의 경쟁에서 격차를 좁히겠다는 전략이다. 삼성전자 관계자는 품질 문제를 해결하고 수율을 크게 개선했다고 밝혔다."
    },
    {
        "id": "news_003",
        "title": "OpenAI, GPT-5 개발 막바지... 연내 출시 예정",
        "date": "2025-01-25",
        "category": "IT",
        "source": "ZDNet Korea",
        "content": "OpenAI가 차세대 AI 모델 GPT-5를 개발 중이며, 2025년 내 출시할 예정이라고 밝혔다. 새로운 모델은 멀티모달 기능이 크게 강화되고, 추론 능력이 대폭 향상될 것으로 알려졌다. 샘 알트만 CEO는 GPT-5가 AGI를 향한 중요한 이정표가 될 것이라고 언급했다."
    },
    {
        "id": "news_004",
        "title": "테슬라 로보택시, 6월 출시 확정... 자율주행 시대 개막",
        "date": "2025-01-24",
        "category": "자동차",
        "source": "조선일보",
        "content": "테슬라가 완전 자율주행 로보택시를 올해 6월 출시한다고 공식 발표했다. 운전대와 페달이 없는 완전 자율주행 차량으로, 초기에는 텍사스와 캘리포니아에서 서비스를 시작할 예정이다. 일론 머스크는 이것이 테슬라 역사상 가장 중요한 제품이라고 강조했다."
    },
    {
        "id": "news_005",
        "title": "애플, 비전프로2 개발 착수... 더 가볍고 저렴하게",
        "date": "2025-01-23",
        "category": "IT",
        "source": "디지털타임스",
        "content": "애플이 비전프로 후속작 개발에 착수했다. 2세대 모델은 무게를 30% 줄이고, 가격도 2,000달러대로 낮출 계획이다. 공간 컴퓨팅 시장 확대를 위해 대중화 전략을 펼치겠다는 의도로 분석된다."
    },
    {
        "id": "news_006",
        "title": "카카오, AI 검색 서비스 '카나' 출시",
        "date": "2025-01-22",
        "category": "IT",
        "source": "전자신문",
        "content": "카카오가 AI 기반 검색 서비스 '카나'를 정식 출시했다. 대화형 인터페이스로 검색 결과를 제공하며, 카카오톡과 연동되어 일상 대화 중에도 검색이 가능하다. 네이버의 큐(CUE:)와 경쟁 구도가 형성될 전망이다."
    },
    {
        "id": "news_007",
        "title": "현대차, 전고체 배터리 탑재 EV 2027년 출시",
        "date": "2025-01-20",
        "category": "자동차",
        "source": "한국경제",
        "content": "현대자동차가 전고체 배터리를 탑재한 전기차를 2027년 출시할 계획이라고 밝혔다. 삼성SDI와 협력하여 개발 중이며, 충전 시간 10분에 주행거리 600km를 목표로 하고 있다."
    },
    {
        "id": "news_008",
        "title": "네이버, 하이퍼클로바X 일본 시장 진출",
        "date": "2025-01-18",
        "category": "IT",
        "source": "매일경제",
        "content": "네이버가 자체 개발한 AI 모델 하이퍼클로바X를 일본 시장에 출시한다. 소프트뱅크와 파트너십을 맺고 일본 기업들에게 AI 솔루션을 제공할 예정이다. K-AI의 해외 진출 성공 사례가 될지 주목된다."
    },
    {
        "id": "news_009",
        "title": "정부, AI 반도체 투자에 10조원 지원 발표",
        "date": "2025-01-15",
        "category": "정책",
        "source": "연합뉴스",
        "content": "정부가 AI 반도체 산업 육성을 위해 향후 5년간 10조원을 투자한다고 발표했다. 팹리스 기업 육성, 인력 양성, R&D 지원 등에 집중 투자할 계획이다. 미국과 중국의 반도체 패권 경쟁 속에서 한국의 경쟁력을 강화하겠다는 목표다."
    },
    {
        "id": "news_010",
        "title": "SK텔레콤, 에이닷 2.0 출시... 멀티모달 AI 적용",
        "date": "2025-01-12",
        "category": "IT",
        "source": "전자신문",
        "content": "SK텔레콤이 AI 비서 에이닷의 대규모 업데이트를 발표했다. 에이닷 2.0은 이미지 인식, 음성 대화, 실시간 번역 기능이 강화되었다. 통신사 AI 경쟁에서 우위를 점하겠다는 전략이다."
    },
]


# ============================================
# Tool 1: RAG - 뉴스 아카이브 검색
# ============================================

_vector_store = None


def _get_or_initialize_vector_store():
    """벡터 스토어 초기화 (lazy loading)"""
    global _vector_store
    
    if _vector_store is not None:
        return _vector_store
    
    logger.info("[RAG] 벡터 스토어 초기화 중...")
    
    # Document 생성
    documents = []
    for news in NEWS_DATABASE:
        doc = Document(
            page_content=f"{news['title']}\n\n{news['content']}",
            metadata={
                "id": news["id"],
                "title": news["title"],
                "date": news["date"],
                "category": news["category"],
                "source": news["source"],
            }
        )
        documents.append(doc)
    
    # 임베딩 및 FAISS 인덱스 생성
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    _vector_store = FAISS.from_documents(documents, embeddings)
    
    logger.info(f"[RAG] 벡터 스토어 초기화 완료 (문서 수: {len(documents)})")
    return _vector_store


class SearchNewsArchiveInput(BaseModel):
    """뉴스 아카이브 검색 입력"""
    query: str = Field(description="검색 키워드 또는 질문")


@tool(args_schema=SearchNewsArchiveInput)
def search_news_archive(query: str) -> str:
    """저장된 뉴스 아카이브에서 관련 기사를 검색합니다.
    
    과거 뉴스 기사, 배경 정보, 분석 자료를 찾을 때 사용합니다.
    
    Args:
        query: 검색 키워드 또는 질문
        
    Returns:
        검색된 뉴스 기사 목록
    """
    logger.info(f"[Tool Call] search_news_archive | query='{query}'")
    
    try:
        vector_store = _get_or_initialize_vector_store()
        docs = vector_store.similarity_search(query, k=3)
        
        if not docs:
            logger.info("[Tool Result] 검색 결과 없음")
            return "관련 뉴스를 찾을 수 없습니다."
        
        results = []
        for i, doc in enumerate(docs, 1):
            result = (
                f"[기사 {i}]\n"
                f"제목: {doc.metadata.get('title', 'N/A')}\n"
                f"날짜: {doc.metadata.get('date', 'N/A')}\n"
                f"출처: {doc.metadata.get('source', 'N/A')}\n"
                f"카테고리: {doc.metadata.get('category', 'N/A')}\n"
                f"내용: {doc.page_content[:300]}..."
            )
            results.append(result)
        
        output = "\n\n---\n\n".join(results)
        logger.info(f"[Tool Result] {len(docs)}개 기사 검색됨")
        return output
        
    except Exception as e:
        logger.error(f"[Tool Error] search_news_archive 실패: {e}")
        return f"뉴스 검색 중 오류가 발생했습니다: {e}"


# ============================================
# Tool 2: Python Function - 날짜 계산
# ============================================

class CalculateDateRangeInput(BaseModel):
    """날짜 범위 계산 입력"""
    time_value: int = Field(ge=0, description="시간 값 (숫자, 0이면 오늘)")
    time_unit: str = Field(description="시간 단위: days, weeks, months")


@tool(args_schema=CalculateDateRangeInput)
def calculate_date_range(time_value: int, time_unit: str) -> str:
    """상대적 시간 표현을 실제 날짜 범위로 계산합니다.
    
    예: "7일 전" → 시작일과 종료일 반환
    예: "2주 이내" → 해당 날짜 범위 계산
    
    Args:
        time_value: 숫자 값 (예: 7)
        time_unit: 단위 - "days", "weeks", "months"
        
    Returns:
        시작일과 종료일 (YYYY-MM-DD 형식)
    """
    logger.info(f"[Tool Call] calculate_date_range | value={time_value}, unit='{time_unit}'")
    
    today = datetime.now()
    
    if time_unit == "days":
        delta = timedelta(days=time_value)
    elif time_unit == "weeks":
        delta = timedelta(weeks=time_value)
    elif time_unit == "months":
        delta = timedelta(days=time_value * 30)  # 근사값
    else:
        logger.warning(f"[Tool Warning] 지원하지 않는 단위: {time_unit}")
        return f"지원하지 않는 시간 단위입니다: {time_unit}. 'days', 'weeks', 'months' 중 하나를 사용하세요."
    
    start_date = today - delta
    
    result = (
        f"날짜 범위 계산 결과:\n"
        f"- 시작일: {start_date.strftime('%Y-%m-%d')}\n"
        f"- 종료일: {today.strftime('%Y-%m-%d')}\n"
        f"- 기간: 최근 {time_value} {time_unit}"
    )
    
    logger.info(f"[Tool Result] {start_date.strftime('%Y-%m-%d')} ~ {today.strftime('%Y-%m-%d')}")
    return result


# ============================================
# Tool 3: External API - SERPER 뉴스 검색
# ============================================

class SearchRecentNewsInput(BaseModel):
    """최신 뉴스 검색 입력"""
    query: str = Field(description="검색 키워드")
    date_from: Optional[str] = Field(default=None, description="검색 시작 날짜 (YYYY-MM-DD)")

    @field_validator("date_from", mode="before")
    @classmethod
    def coerce_date_from(cls, v):
        """정수가 들어오면 문자열로 변환"""
        if v is None:
            return None
        return str(v)


@tool(args_schema=SearchRecentNewsInput)
def search_recent_news(query: str, date_from: Optional[str] = None) -> str:
    """SERPER API를 사용하여 최신 뉴스를 검색합니다.
    
    실시간 뉴스, 최신 소식, 현재 이슈를 찾을 때 사용합니다.
    
    Args:
        query: 검색 키워드
        date_from: 검색 시작 날짜 (선택)
        
    Returns:
        최신 뉴스 검색 결과
    """
    logger.info(f"[Tool Call] search_recent_news | query='{query}', date_from='{date_from}'")
    
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        logger.warning("[Tool Warning] SERPER_API_KEY가 설정되지 않음")
        return "SERPER API를 사용할 수 없습니다. API 키를 확인해주세요."
    
    url = "https://google.serper.dev/news"
    
    payload = {
        "q": query,
        "gl": "kr",
        "hl": "ko",
        "num": 5,
    }
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "news" not in data or not data["news"]:
            logger.info("[Tool Result] 검색 결과 없음")
            return "최신 뉴스를 찾을 수 없습니다."
        
        results = []
        for i, article in enumerate(data["news"][:5], 1):
            result = (
                f"[뉴스 {i}]\n"
                f"제목: {article.get('title', 'N/A')}\n"
                f"출처: {article.get('source', 'N/A')}\n"
                f"날짜: {article.get('date', 'N/A')}\n"
                f"요약: {article.get('snippet', 'N/A')}\n"
                f"링크: {article.get('link', 'N/A')}"
            )
            results.append(result)
        
        output = "\n\n---\n\n".join(results)
        logger.info(f"[Tool Result] {len(data['news'][:5])}개 뉴스 검색됨")
        return output
        
    except requests.exceptions.Timeout:
        logger.error("[Tool Error] SERPER API 타임아웃")
        return "뉴스 검색 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
    except requests.exceptions.RequestException as e:
        logger.error(f"[Tool Error] SERPER API 요청 실패: {e}")
        return f"뉴스 검색 중 오류가 발생했습니다: {e}"


# ============================================
# 도구 목록 (노드에서 사용)
# ============================================

TOOLS = [search_news_archive, calculate_date_range, search_recent_news]
