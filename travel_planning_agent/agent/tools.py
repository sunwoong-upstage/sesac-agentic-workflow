"""
여행 계획 에이전트 - 도구(Tool) 정의

이 모듈은 에이전트가 사용할 수 있는 도구들을 정의합니다.

포함된 도구:
1. search_travel_knowledge: 여행 지식베이스 검색 (RAG - FAISS 벡터 검색)
2. estimate_budget: 여행 예산 추정
3. web_search: 실시간 웹 검색 (Serper Google Search API)

핵심 기술:
- FAISS 벡터 스토어를 활용한 Agentic RAG
- Serper API를 활용한 실시간 외부 웹 검색
- Lazy Initialization으로 임포트 실패 방지
- 키워드 기반 폴백 검색
- 모든 도구에 try/except 에러 핸들링
"""

import os
import requests
from langchain_core.tools import tool
from langchain_core.documents import Document
from pydantic import BaseModel, Field


# =============================================================================
# Mock 데이터베이스
# =============================================================================

# 여행 지식베이스 (RAG용)
TRAVEL_KNOWLEDGE_BASE = [
    # 한국 여행지
    {
        "id": "KR-001",
        "category": "국내여행",
        "title": "제주도 여행 가이드",
        "content": "제주도는 한국 최대의 섬으로, 한라산, 성산일출봉, 만장굴 등 유네스코 세계자연유산을 보유하고 있습니다. 봄에는 유채꽃, 여름에는 해수욕, 가을에는 억새, 겨울에는 한라산 설경이 아름답습니다. 렌터카 여행이 가장 편리하며, 동쪽-서쪽 해안도로 드라이브가 인기입니다. 흑돼지, 해산물, 감귤이 대표 먹거리입니다. 주요 관광지로는 성산일출봉, 한라산 국립공원, 만장굴, 천지연폭포, 우도, 중문관광단지, 협재해수욕장이 있습니다. 맛집으로는 흑돼지거리(노형동), 동문시장, 제주 해산물 맛집거리, 한림 칼국수거리가 유명합니다. 애월 카페거리, 월정리 카페거리, 협재 카페거리도 인기입니다. 감귤따기 체험, 해녀 체험, 승마 체험, 올레길 트레킹 등의 활동도 가능합니다.",
    },
    {
        "id": "KR-002",
        "category": "국내여행",
        "title": "부산 여행 가이드",
        "content": "부산은 한국 제2의 도시로, 해운대, 광안리, 감천문화마을, 자갈치시장이 대표 관광지입니다. KTX로 서울에서 2시간 30분이면 도착합니다. 돼지국밥, 밀면, 씨앗호떡이 대표 음식이며, 영화의전당, BIFF 광장 등 문화시설도 풍부합니다. 여름 해수욕과 불꽃축제가 특히 유명합니다. 주요 관광지로는 해운대해수욕장, 광안대교(광안리), 감천문화마을, 해동용궁사, 태종대, 영도 흰여울문화마을이 있습니다. 자갈치시장, 부평깡통시장, 서면 먹자골목, 해운대 포장마차촌이 맛집 지역으로 유명하며, 영도 카페거리, 해운대 카페거리, 전포카페거리도 인기입니다. 요트 투어, 야경 크루즈, 부산시티투어 버스 등의 체험 활동을 즐길 수 있습니다.",
    },
    {
        "id": "KR-003",
        "category": "국내여행",
        "title": "경주 여행 가이드",
        "content": "경주는 신라 천년의 수도로, 불국사, 석굴암, 첨성대, 안압지(동궁과 월지) 등 역사 유적이 풍부합니다. 도보와 자전거로 시내 유적을 둘러볼 수 있습니다. 경주빵, 황남빵이 대표 간식이며, 보문관광단지에서 숙박하면 편리합니다. 봄 벚꽃과 가을 단풍 시즌이 최적입니다.",
    },
    {
        "id": "KR-004",
        "category": "국내여행",
        "title": "강릉 여행 가이드",
        "content": "강릉은 동해안의 대표 여행지로, 경포대, 주문진, 안목해변 카페거리가 유명합니다. 서울에서 KTX로 약 2시간 소요됩니다. 초당순두부, 감자옹심이가 대표 음식이며, 커피의 도시로 불릴 만큼 카페 문화가 발달했습니다. 여름 해수욕과 겨울 서핑이 인기입니다.",
    },
    {
        "id": "KR-005",
        "category": "국내여행",
        "title": "서울 여행 가이드",
        "content": "서울은 한국의 수도로, 경복궁, 북촌한옥마을, 명동, 홍대, 이태원, 남산타워가 대표 관광지입니다. 지하철이 매우 편리하여 대중교통만으로 주요 관광지를 모두 방문할 수 있습니다. 한식, 길거리 음식, 다양한 세계 요리를 즐길 수 있으며, 쇼핑과 K-pop 문화 체험도 가능합니다.",
    },
    # 해외 여행지
    {
        "id": "JP-001",
        "category": "해외여행",
        "title": "도쿄 여행 가이드",
        "content": "도쿄는 일본의 수도로, 시부야, 신주쿠, 아사쿠사(센소지), 하라주쿠, 아키하바라가 대표 관광지입니다. 인천에서 비행기로 약 2시간 30분 소요됩니다. 스시, 라멘, 텐푸라 등 일본 음식의 본고장이며, 도쿄 디즈니랜드와 팀랩 전시도 인기입니다. 교통카드(Suica/PASMO)로 편리하게 이동할 수 있습니다. 주요 관광지로는 센소지(아사쿠사), 시부야 스크램블, 도쿄타워, 메이지신궁, 도쿄 스카이트리, 우에노공원이 있습니다. 츠키지 외시장, 라멘 요코초, 긴자 스시거리, 신주쿠 이자카야 골목이 맛집 지역으로 유명합니다. 하라주쿠 다케시타도리, 아키하바라, 시부야109, 긴자 백화점거리에서 쇼핑을 즐길 수 있으며, 팀랩(teamLab), 도쿄 디즈니랜드/시, 로봇 레스토랑, 기모노 체험 등의 활동도 인기입니다.",
    },
    {
        "id": "JP-002",
        "category": "해외여행",
        "title": "오사카 여행 가이드",
        "content": "오사카는 일본의 '먹자골목' 도시로, 도톤보리, 오사카성, 유니버설 스튜디오 재팬이 대표 관광지입니다. 타코야키, 오코노미야키, 쿠시카츠가 대표 음식이며, '쿠이다오레(먹다 쓰러지다)' 문화가 유명합니다. 교토, 나라와 가까워 함께 여행하기 좋습니다.",
    },
    {
        "id": "TH-001",
        "category": "해외여행",
        "title": "방콕 여행 가이드",
        "content": "방콕은 태국의 수도로, 왕궁, 왓아룬, 왓포, 카오산로드가 대표 관광지입니다. 인천에서 약 5시간 30분 소요됩니다. 팟타이, 똠얌꿍, 망고스티키라이스 등 태국 음식이 저렴하고 맛있습니다. 야시장과 수상시장 체험, 태국 마사지가 인기입니다. 11~2월 건기가 여행 최적기입니다. 주요 관광지로는 왕궁, 왓아룬, 왓포, 짜뚜짝 주말시장, 카오산로드가 있습니다. 야와랏(차이나타운), 방람푸 길거리 음식, 아이콘시암 푸드코트가 맛집으로 유명합니다. 수상시장(담넌사두악), 태국 전통 마사지, 무에타이 관람, 쿠킹클래스 등의 체험 활동도 즐길 수 있습니다.",
    },
    {
        "id": "VN-001",
        "category": "해외여행",
        "title": "다낭 여행 가이드",
        "content": "다낭은 베트남 중부의 해양 도시로, 미케비치, 바나힐, 호이안 올드타운, 오행산이 대표 관광지입니다. 인천에서 약 4시간 30분 소요됩니다. 쌀국수(퍼), 반미, 분짜가 대표 음식이며, 물가가 저렴하여 가성비 여행지로 인기입니다. 3~8월이 여행 최적기입니다. 주요 관광지로는 미케비치, 바나힐(골든브릿지), 오행산, 한강 드래곤브릿지, 참 조각 박물관이 있습니다. 한시장, 미케비치 해산물, 꽝남 길거리 음식이 맛집 지역으로 유명합니다. 호이안 올드타운(UNESCO), 바구니배 체험, 스노클링/다이빙 등의 활동을 즐길 수 있습니다.",
    },
    {
        "id": "FR-001",
        "category": "해외여행",
        "title": "파리 여행 가이드",
        "content": "파리는 프랑스의 수도로, 에펠탑, 루브르 박물관, 개선문, 몽마르뜨, 세느강 크루즈가 대표 관광지입니다. 인천에서 약 12시간 소요됩니다. 크루아상, 에스카르고, 와인 등 프랑스 미식의 중심지이며, 패션과 예술의 도시입니다. 봄(4~6월)과 가을(9~10월)이 여행 최적기입니다. 주요 관광지로는 에펠탑, 루브르 박물관, 개선문, 몽마르뜨/사크레쾨르, 노트르담 대성당, 세느강 크루즈가 있습니다. 마레지구 레스토랑, 생제르맹 카페, 몽마르뜨 비스트로가 맛집으로 유명합니다. 샹젤리제 거리, 갤러리 라파예트, 봉 마르셰에서 쇼핑을 즐길 수 있으며, 베르사유 궁전(당일치기), 와인 테이스팅, 세느강 크루즈 디너 등의 체험 활동도 인기입니다.",
    },
    # 여행 팁
    {
        "id": "TIP-001",
        "category": "여행팁",
        "title": "여행 짐 싸기 체크리스트",
        "content": "여행 짐 싸기 필수 항목: 여권, 항공권, 숙소 예약 확인서, 여행자보험 서류, 현지 통화/카드, 충전기, 어댑터(국가별 확인), 상비약(두통약, 소화제, 밴드), 세면도구, 자외선차단제, 우산/우비. 짐을 줄이려면 3일 기준 옷을 준비하고 숙소에서 세탁하세요.",
    },
    {
        "id": "TIP-002",
        "category": "여행팁",
        "title": "환전 가이드",
        "content": "환전 팁: 1) 출발 전 시중은행에서 기본 금액 환전 (수수료 우대쿠폰 활용) 2) 현지 ATM에서 필요한 만큼 인출 (수수료 확인) 3) 트래블월렛, 트래블로그 등 해외 전용 체크카드 활용 4) 공항 환전은 수수료가 높으니 최소한만. 일본은 현금 문화가 강하니 넉넉히, 유럽은 카드 위주로 준비하세요.",
    },
    {
        "id": "TIP-003",
        "category": "여행팁",
        "title": "해외여행 교통 가이드",
        "content": "해외 교통 팁: 1) 일본: 교통카드(Suica/PASMO) 필수, JR패스 고려 2) 태국: 그랩(Grab) 앱 필수, BTS/MRT 이용 3) 유럽: 유레일패스 또는 저가항공(라이언에어 등) 4) 동남아: 그랩 택시가 안전하고 편리 5) 구글맵 또는 네이버맵(일본) 오프라인 지도 다운로드 추천",
    },
    {
        "id": "TIP-004",
        "category": "여행팁",
        "title": "여행 예산 절약 팁",
        "content": "여행 경비 절약 방법: 1) 항공권: 비수기 예약, 스카이스캐너로 가격 비교, 경유편 활용 2) 숙소: 에어비앤비, 호스텔, 게스트하우스 활용 3) 식비: 현지 시장/편의점 이용, 점심에 비싼 레스토랑 방문 (런치 세트가 저렴) 4) 교통: 대중교통 패스 활용, 도보 관광 5) 관광: 무료 투어, 박물관 무료 입장일 확인",
    },
    {
        "id": "TIP-005",
        "category": "여행팁",
        "title": "동남아 여행 시 주의사항",
        "content": "동남아 여행 주의사항: 1) 물은 반드시 생수 구매 (수돗물 금지) 2) 길거리 음식은 사람 많은 곳에서 3) 소매치기 주의 (크로스백 권장) 4) 사원 방문 시 복장 규정 확인 (긴바지, 어깨 가리기) 5) 자외선 차단제 필수 6) 모기 기피제 준비 7) 여행자보험 가입 필수",
    },
]

# 예산 데이터 (Mock, 1인 기준, 원화)
BUDGET_DB = {
    "제주도": {"숙박": 100000, "식비": 50000, "교통": 40000, "관광": 20000, "항공": 100000},
    "부산": {"숙박": 80000, "식비": 40000, "교통": 25000, "관광": 15000, "항공/KTX": 60000},
    "도쿄": {"숙박": 120000, "식비": 50000, "교통": 20000, "관광": 25000, "항공": 350000},
    "방콕": {"숙박": 70000, "식비": 30000, "교통": 15000, "관광": 20000, "항공": 400000},
    "다낭": {"숙박": 60000, "식비": 25000, "교통": 12000, "관광": 15000, "항공": 350000},
    "파리": {"숙박": 150000, "식비": 70000, "교통": 20000, "관광": 35000, "항공": 1000000},
}


# =============================================================================
# FAISS 벡터 스토어 (Lazy Initialization)
#
# 교육용 참고:
# 벡터 스토어는 처음 검색 요청 시 초기화됩니다 (Lazy Initialization).
# 모듈 로드 시점이 아닌 첫 사용 시점에 초기화하여:
# 1. API 키 없이도 다른 도구는 정상 작동
# 2. 임포트 실패 방지
# 3. 불필요한 API 호출 비용 절감
#
# 프로덕션에서는 벡터 스토어를 사전 구축하고 파일로 저장해두는 것이 일반적입니다.
# =============================================================================

_vector_store = None  # 모듈 레벨 캐시


def _create_knowledge_base_documents() -> list:
    """여행 지식베이스를 LangChain Document 객체 리스트로 변환합니다."""
    documents = []
    for item in TRAVEL_KNOWLEDGE_BASE:
        doc = Document(
            page_content=item["content"],
            metadata={
                "id": item["id"],
                "category": item["category"],
                "title": item["title"],
            }
        )
        documents.append(doc)
    return documents


def _get_or_initialize_vector_store():
    """
    FAISS 벡터 스토어를 반환합니다. 첫 호출 시 초기화.

    초기화 전략:
    1. UpstageEmbeddings + FAISS 시도
    2. 실패 시 None 반환 (폴백 검색 사용)
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    try:
        from langchain_upstage import UpstageEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        documents = _create_knowledge_base_documents()
        _vector_store = FAISS.from_documents(documents, embeddings)
        print("[INFO] FAISS 벡터 스토어 초기화 성공")
        return _vector_store
    except Exception as e:
        print(f"[경고] FAISS 초기화 실패, 키워드 검색으로 폴백: {e}")
        return None


def _keyword_fallback_search(query: str) -> str:
    """FAISS 실패 시 키워드 기반 폴백 검색"""
    query_lower = query.lower()
    for item in TRAVEL_KNOWLEDGE_BASE:
        if any(word in (item["title"] + " " + item["content"]).lower()
               for word in query_lower.split() if len(word) >= 2):
            return f"[{item['category']}] {item['title']}\n{item['content']}"
    return "관련 여행 정보를 찾지 못했습니다."


# =============================================================================
# 도구 정의
# =============================================================================

class TravelSearchInput(BaseModel):
    """여행 지식 검색 입력 스키마"""
    query: str = Field(description="검색 쿼리 (예: '제주도 여행', '환전 팁')")


class WebSearchInput(BaseModel):
    """웹 검색 입력 스키마"""
    query: str = Field(description="검색할 쿼리 (예: '제주도 맛집 추천 2024', '도쿄 벚꽃 시기')")


@tool(args_schema=TravelSearchInput)
def search_travel_knowledge(query: str) -> str:
    """
    여행 관련 지식베이스를 검색합니다. (FAISS 벡터 검색)

    여행지 정보, 여행 팁, 가이드 등을 검색할 때 사용합니다.

    Args:
        query: 검색 쿼리

    Returns:
        관련 여행 정보
    """
    try:
        vector_store = _get_or_initialize_vector_store()
        if vector_store is not None:
            docs = vector_store.similarity_search(query, k=3)
            return "\n\n".join(
                f"[{doc.metadata.get('category', '')}] {doc.metadata.get('title', '')}\n{doc.page_content}"
                for doc in docs
            )
        # 폴백: 키워드 매칭
        return _keyword_fallback_search(query)
    except Exception as e:
        # 어떤 오류든 폴백으로 처리
        return _keyword_fallback_search(query)


class BudgetInput(BaseModel):
    """예산 추정 입력 스키마"""
    destination: str = Field(description="여행지 이름 (예: '제주도', '도쿄')")
    duration_days: int = Field(description="여행 기간 (일수)", ge=1)


@tool(args_schema=BudgetInput)
def estimate_budget(destination: str, duration_days: int) -> str:
    """
    여행 예산을 추정합니다.

    여행 계획 시 예상 경비를 알고 싶을 때 사용합니다.
    1인 기준 예상 비용을 항목별로 계산합니다.

    Args:
        destination: 여행지 이름
        duration_days: 여행 기간 (일수)

    Returns:
        항목별 예산 추정 결과
    """
    try:
        # 유연한 여행지 매칭 (예: "제주" → "제주도")
        matched_destination = None
        for key in BUDGET_DB.keys():
            if destination in key or key in destination:
                matched_destination = key
                break
        
        if not matched_destination:
            return f"'{destination}' 예산 정보가 없습니다. 지원 도시: {', '.join(BUDGET_DB.keys())}"
        
        budget_data = BUDGET_DB[matched_destination]

        result = f"{matched_destination} {duration_days}일 여행 예산 (1인 기준):\n\n"

        total = 0
        for cost_item, daily_cost in budget_data.items():
            if "항공" in cost_item or "KTX" in cost_item:
                # 교통비(항공/KTX)는 왕복 1회
                cost = daily_cost
                result += f"  - {cost_item} (왕복): {cost:,}원\n"
            else:
                cost = daily_cost * duration_days
                result += f"  - {cost_item} ({daily_cost:,}원/일 × {duration_days}일): {cost:,}원\n"
            total += cost

        result += f"\n  총 예상 비용: {total:,}원"
        result += f"\n\n참고: 실제 비용은 시즌, 예약 시기, 개인 소비 습관에 따라 달라질 수 있습니다."

        return result

    except Exception as e:
        return f"예산 추정 중 오류가 발생했습니다: {e}"


@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """
    Serper API를 사용하여 Google 웹 검색을 수행합니다.

    실시간 여행 정보, 최신 리뷰, 현재 이벤트 등을 검색할 때 사용합니다.
    내부 지식베이스에 없는 최신 정보가 필요할 때 유용합니다.

    Args:
        query: 검색 쿼리

    Returns:
        검색 결과 요약
    """
    try:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return "웹 검색을 사용할 수 없습니다 (SERPER_API_KEY 미설정). 내부 지식베이스를 활용해주세요."

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "gl": "kr",
            "hl": "ko",
            "num": 5,
        }

        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []

        # Knowledge Graph (있는 경우)
        knowledge_graph = data.get("knowledgeGraph", {})
        if knowledge_graph:
            knowledge_graph_info = f"[지식 패널] {knowledge_graph.get('title', '')}"
            if knowledge_graph.get("description"):
                knowledge_graph_info += f": {knowledge_graph['description']}"
            results.append(knowledge_graph_info)

        # Organic 검색 결과
        for item in data.get("organic", [])[:5]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            results.append(f"- {title}: {snippet}")

        if not results:
            return f"'{query}'에 대한 검색 결과가 없습니다."

        return f"웹 검색 결과 ({query}):\n\n" + "\n".join(results)

    except requests.exceptions.Timeout:
        return f"웹 검색 시간 초과. 내부 지식베이스를 활용해주세요."
    except requests.exceptions.RequestException as e:
        return f"웹 검색 중 오류 발생: {e}. 내부 지식베이스를 활용해주세요."
    except Exception as e:
        return f"웹 검색 처리 중 오류: {e}"


# =============================================================================
# 도구 그룹
# =============================================================================

# 모든 조사 도구 (research_node에서 LLM에 바인딩)
RESEARCH_TOOLS = [search_travel_knowledge, estimate_budget, web_search]
