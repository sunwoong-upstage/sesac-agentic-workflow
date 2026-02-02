# 여행 계획 에이전트 학습 활동지

> **대상:** AI 에이전트 패턴을 배우는 초급 개발자
> **사전 요구사항:** 기본 Python, Practice01-09 노트북 완료
> **예상 소요 시간:** 총 4-6시간

---

## 목차

1. [활동 1A: 상태(State) 정의 (빈칸 채우기)](#활동-1a-상태state-정의-빈칸-채우기)
2. [활동 1B: 도구(Tool) 정의 (빈칸 채우기)](#활동-1b-도구tool-정의-빈칸-채우기)
3. [활동 1C: 그래프 구성 (빈칸 채우기)](#활동-1c-그래프-구성-빈칸-채우기)
4. [활동 3A: 개념 매핑 테이블](#활동-3a-개념-매핑-테이블)
5. [활동 5A: 날씨 도구 추가 (초급)](#활동-5a-새-도구-추가---날씨-예보-초급)
6. [활동 5B: 입력 검증 노드 추가 (중급)](#활동-5b-새-노드-추가---입력-검증-중급)
7. [활동 5C: 평가 시스템 개선 (중급)](#활동-5c-평가-최적화-루프-개선-중급)
8. [활동 5D: 개인화 추천 노드 (고급)](#활동-5d-새-노드-추가---개인화-추천-고급)
9. [활동 5E: 안전 가드레일 (고급)](#활동-5e-안전-가드레일-노드-추가-고급)
10. [제출 체크리스트](#제출-체크리스트)

---

## 활동 1: 빈칸 채우기 연습

### 활동 1A: 상태(State) 정의 (빈칸 채우기)

**파일:** `agent/state.py`
**학습 목표:** TypedDict, Pydantic, Annotated 이해하기
**예상 시간:** 20분

```python
# =============================================================================
# 연습문제 1A: 상태(State) 정의 완성하기
# 파일: agent/state.py
# 학습 목표: TypedDict, Pydantic, Annotated 이해하기
# =============================================================================

# 문제 1: 기본 import 완성하기
import ______  # 누적 연산을 위한 모듈 (힌트: add 함수 사용)
from typing import Annotated, List, ______  # 제한된 값만 허용하는 타입
from ______ import BaseModel, Field  # 데이터 검증 라이브러리

# 문제 2: 메시지 누적을 위한 상태 필드 정의
class TravelPlanningState(TypedDict):
    """여행 계획 에이전트의 워크플로우 상태"""

    # messages 필드: 새 메시지가 추가될 때마다 누적되어야 함
    # 힌트: Annotated와 operator.add를 조합하세요
    messages: Annotated[List[BaseMessage], ______]

    # tool_results 필드: 도구 호출 결과도 누적되어야 함
    tool_results: Annotated[______, operator.add]

    # 일반 필드: 누적 없이 덮어쓰기
    user_input: ______
    final_response: ______
    quality_score: ______  # 1-10 점수

# 문제 3: 의도 분류 스키마 정의 (Pydantic)
class IntentClassification(______):  # 어떤 클래스를 상속?
    """의도 분류 결과 스키마"""
    intent: Literal[
        "destination_research",
        "itinerary_planning",
        "______",  # 예산 관련 의도
        "general_travel"
    ] = ______(description="문의 유형")  # 필드 설명 추가 함수

# 문제 4: 여행 계획 스키마 - 필드 검증 추가
class TravelPlan(BaseModel):
    """여행 계획 스키마 (Plan-and-Solve)"""
    destination: str = Field(description="주요 여행지")
    # duration_days: 최소 1일 이상이어야 함 (ge=greater or equal)
    duration_days: int = Field(description="여행 기간 (일수)", ______=1)
    steps: ______ = Field(description="실행 계획 단계")  # 문자열 리스트 타입

# 문제 5: 선호도 추출 스키마 - Optional 필드
class ExtractedPreferences(BaseModel):
    """대화에서 추출한 여행 선호도 스키마"""
    # Optional: 값이 없을 수 있음 (None 허용)
    destination: ______ = Field(default=______, description="여행지")
    budget: int | None = Field(default=None, description="예산 (원화)", ge=______)
```

<details>
<summary>정답 보기</summary>

```python
# 문제 1
import operator
from typing import Annotated, List, Literal
from pydantic import BaseModel, Field

# 문제 2
messages: Annotated[List[BaseMessage], operator.add]
tool_results: Annotated[List[dict], operator.add]
user_input: str
final_response: str
quality_score: int

# 문제 3
class IntentClassification(BaseModel):
    intent: Literal[..., "budget_estimation", ...] = Field(description="문의 유형")

# 문제 4
duration_days: int = Field(..., ge=1)
steps: List[str] = Field(...)

# 문제 5
destination: str | None = Field(default=None, ...)
budget: int | None = Field(default=None, ..., ge=0)
```

</details>

---

### 활동 1B: 도구(Tool) 정의 (빈칸 채우기)

**파일:** `agent/tools.py`
**학습 목표:** @tool 데코레이터, Pydantic 스키마, FAISS 이해하기
**예상 시간:** 25분

```python
# =============================================================================
# 연습문제 1B: 도구(Tool) 정의 완성하기
# 파일: agent/tools.py
# 학습 목표: @tool 데코레이터, Pydantic 스키마, FAISS 이해하기
# =============================================================================

from langchain_core.tools import ______  # 도구 데코레이터
from pydantic import BaseModel, Field

# 문제 1: 도구 입력 스키마 정의
class TravelSearchInput(______):  # 어떤 클래스 상속?
    query: str = ______(description="검색 쿼리 (예: '제주도 여행', '환전 팁')")

class BudgetInput(BaseModel):
    destination: str = Field(description="여행지 이름")
    duration_days: int = Field(description="여행 기간 (일수)", ______=1)  # 최소값 검증
    user_budget: ______ = Field(default=None, description="사용자 예산 (선택)")  # Optional int

# 문제 2: FAISS 벡터 스토어 초기화
def _get_or_initialize_vector_store():
    """FAISS 벡터 스토어 초기화 (싱글톤 패턴)"""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    from langchain_upstage import ______  # 임베딩 클래스
    from langchain_community.vectorstores import ______  # 벡터 스토어 클래스

    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    documents = _create_knowledge_base_documents()
    _vector_store = ______.from_documents(documents, embeddings)  # 문서로부터 생성
    return _vector_store

# 문제 3: 검색 도구 정의
@______(args_schema=TravelSearchInput)  # 데코레이터 이름
def search_travel_knowledge(query: str) -> str:
    """여행 지식 베이스에서 정보를 검색합니다."""
    vector_store = _get_or_initialize_vector_store()
    if vector_store is not None:
        # similarity_search: 유사도 기반 검색, k=3은 상위 3개 결과
        docs = vector_store.______(query, k=______)
        result = "\n\n".join(
            f"[{doc.metadata.get('category')}] {doc.______.get('title')}\n{doc.______}"
            for doc in docs
        )
        return result
    return _keyword_fallback_search(query)

# 문제 4: 예산 추정 도구 (복잡한 로직)
@tool(args_schema=BudgetInput)
def estimate_budget(destination: str, duration_days: int, user_budget: int | None = None) -> str:
    """여행 예산을 추정합니다."""
    # destination이 BUDGET_DB에 있는지 확인
    matched_destination = None
    for key in BUDGET_DB.______():  # dict의 키 순회
        if destination in key or key in destination:
            matched_destination = key
            ______  # 찾으면 루프 탈출

    if not matched_destination:
        return f"'{destination}' 예산 정보가 없습니다."

    # 일일 비용 * 일수 계산 (항공은 제외)
    for cost_item, daily_cost in option.______():  # dict 아이템 순회
        if "항공" in cost_item:
            cost = daily_cost  # 항공은 왕복 1회
        else:
            cost = daily_cost ______ duration_days  # 일일 비용 * 일수 (곱하기 연산자)
        total += cost

    return result

# 문제 5: 웹 검색 도구 (API 호출)
@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """Google 웹 검색을 수행합니다."""
    import os
    import ______  # HTTP 요청 라이브러리

    api_key = os.______("SERPER_API_KEY")  # 환경변수 읽기
    if not api_key:
        return "SERPER_API_KEY 미설정"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/______",  # JSON 컨텐츠 타입
    }

    response = requests.______(  # POST 메서드
        "https://google.serper.dev/search",
        headers=headers,
        json=payload,
        timeout=10,
    )
    response.______()  # 에러 시 예외 발생
    return response.json()

# 문제 6: 도구 리스트 정의 (research_node에서 사용)
RESEARCH_TOOLS = [______, ______, ______]  # 3개 도구
```

<details>
<summary>정답 보기</summary>

```python
# 문제 1
from langchain_core.tools import tool
class TravelSearchInput(BaseModel):
    query: str = Field(description="...")
class BudgetInput(BaseModel):
    duration_days: int = Field(..., ge=1)
    user_budget: int | None = Field(...)

# 문제 2
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
_vector_store = FAISS.from_documents(documents, embeddings)

# 문제 3
@tool(args_schema=TravelSearchInput)
docs = vector_store.similarity_search(query, k=3)
f"[{doc.metadata.get('category')}] {doc.metadata.get('title')}\n{doc.page_content}"

# 문제 4
for key in BUDGET_DB.keys():
    break
for cost_item, daily_cost in option.items():
    cost = daily_cost * duration_days

# 문제 5
import requests
api_key = os.getenv("SERPER_API_KEY")
"Content-Type": "application/json"
response = requests.post(...)
response.raise_for_status()

# 문제 6
RESEARCH_TOOLS = [search_travel_knowledge, estimate_budget, web_search]
```

</details>

---

### 활동 1C: 그래프 구성 (빈칸 채우기)

**파일:** `agent/graph.py`
**학습 목표:** LangGraph StateGraph, 엣지, 조건부 라우팅 이해하기
**예상 시간:** 25분

```python
# =============================================================================
# 연습문제 1C: 워크플로우 그래프 구성하기
# 파일: agent/graph.py
# 학습 목표: LangGraph StateGraph, 엣지, 조건부 라우팅 이해하기
# =============================================================================

from langgraph.graph import StateGraph, ______, ______  # 시작/종료 상수
from langgraph.checkpoint.memory import ______  # 체크포인터
from langgraph.store.memory import ______  # 사용자 프로필 저장소

# 문제 1: 노드 함수 import
from agent.nodes import (
    classify_intent_node,
    ______,  # 선호도 추출
    plan_node,
    ______,  # 조사/도구 호출
    synthesize_node,
    ______,  # 품질 평가
    improve_response_node,
    save_memory_node,
    ______,  # 조건부 라우팅 함수
)

def create_travel_planning_graph(with_memory: bool = True):
    """여행 계획 에이전트 그래프 생성"""

    # 문제 2: StateGraph 생성
    builder = ______(TravelPlanningState)  # 어떤 클래스?

    # 문제 3: 노드 추가 (8개)
    builder.______(______, classify_intent_node)  # 메서드와 노드명
    builder.add_node("extract_preferences", extract_preferences_node)
    builder.add_node("plan", ______)  # plan_node 연결
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("evaluate_response", evaluate_response_node)
    builder.add_node("______", improve_response_node)  # 노드 이름
    builder.add_node("save_memory", save_memory_node)

    # 문제 4: 시작 엣지 (START -> 첫 번째 노드)
    builder.______(______, "classify_intent")

    # 문제 5: 조건부 엣지 (빈 입력 처리)
    builder.add_conditional_edges(
        "classify_intent",
        ______ s: "skip" if s.get("______") else "continue",  # 람다 함수
        {"continue": "______", "skip": "save_memory"}  # 라우팅 맵
    )

    # 문제 6: 순차 엣지 (Plan-and-Solve 파이프라인)
    builder.add_edge("extract_preferences", "______")
    builder.add_edge("plan", "______")
    builder.add_edge("research", "______")
    builder.add_edge("synthesize", "______")

    # 문제 7: 조건부 엣지 (품질 평가 후 분기)
    builder.______(
        "evaluate_response",
        should_improve_response,  # 라우팅 함수
        {
            "______": "improve_response",  # 개선 필요 시
            "______": "save_memory",  # 통과 시
        }
    )

    # 문제 8: 개선 후 재평가 루프
    builder.add_edge("______", "______")  # improve -> evaluate

    # 문제 9: 종료 엣지
    builder.add_edge("save_memory", ______)

    # 문제 10: 그래프 컴파일
    if with_memory:
        memory = ______()
        graph = builder.______(checkpointer=memory, store=user_store)
    else:
        graph = builder.compile()

    return graph
```

<details>
<summary>정답 보기</summary>

```python
# Import
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# 문제 1
from agent.nodes import (
    extract_preferences_node,
    research_node,
    evaluate_response_node,
    should_improve_response,
)

# 문제 2
builder = StateGraph(TravelPlanningState)

# 문제 3
builder.add_node("classify_intent", classify_intent_node)
builder.add_node("plan", plan_node)
builder.add_node("improve_response", improve_response_node)

# 문제 4
builder.add_edge(START, "classify_intent")

# 문제 5
lambda s: "skip" if s.get("skip_to_end") else "continue"
{"continue": "extract_preferences", "skip": "save_memory"}

# 문제 6
builder.add_edge("extract_preferences", "plan")
builder.add_edge("plan", "research")
builder.add_edge("research", "synthesize")
builder.add_edge("synthesize", "evaluate_response")

# 문제 7
builder.add_conditional_edges(
    "evaluate_response",
    should_improve_response,
    {"improve": "improve_response", "end": "save_memory"}
)

# 문제 8
builder.add_edge("improve_response", "evaluate_response")

# 문제 9
builder.add_edge("save_memory", END)

# 문제 10
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, store=user_store)
```

</details>

---

## 활동 3A: 개념 매핑 테이블

**학습 목표:** Practice 노트북에서 배운 개념이 실제 코드 어디에 적용되었는지 연결하기
**예상 시간:** 30분

### 안내

1. 아래 표의 빈칸 (라인 번호)을 찾아 채우세요
2. 해당 파일을 직접 열어 코드를 확인하세요
3. Practice 노트북의 개념과 비교해보세요

| Practice 노트북 | 핵심 개념 | 코드 파일 | 라인 번호 |
|---|---|---|---|
| **Practice01** | AI 에이전트 정의 | `main.py` | 전체 |
| **Practice02** | Workflow vs Agent | `graph.py` | 30-76 |
| **Practice02** | Plan-and-Solve: plan_node | `nodes.py` | ___-___ |
| **Practice02** | Plan-and-Solve: research_node | `nodes.py` | ___-___ |
| **Practice02** | Plan-and-Solve: synthesize_node | `nodes.py` | ___-___ |
| **Practice02** | Evaluator-Optimizer 루프 | `graph.py` | ___-___ |
| **Practice03** | Tool 정의 (@tool): search | `tools.py` | ___-___ |
| **Practice03** | Tool 정의 (@tool): budget | `tools.py` | ___-___ |
| **Practice03** | Tool 정의 (@tool): web | `tools.py` | ___-___ |
| **Practice03** | Tool Calling (bind_tools) | `nodes.py` | ___ |
| **Practice03** | Pydantic 스키마 (BudgetInput) | `tools.py` | ___-___ |
| **Practice05** | FAISS 벡터 스토어 초기화 | `tools.py` | ___-___ |
| **Practice05** | similarity_search | `tools.py` | ___ |
| **Practice05** | Agentic RAG (LLM이 도구 결정) | `nodes.py` | ___-___ |
| **Practice06** | Short-term Memory (MemorySaver) | `graph.py` | ___ |
| **Practice06** | Long-term Memory (InMemoryStore) | `graph.py` | ___ |
| **Practice06** | 메시지 누적 (operator.add) | `state.py` | ___ |
| **Practice07** | Structured Output | `nodes.py` | ___, ___, ___, ___ |
| **Practice09** | LLM-as-Judge | `nodes.py` | ___-___ |
| **Practice09** | quality_score 필드 | `state.py` | ___-___ |

<details>
<summary>정답 보기</summary>

| Practice 노트북 | 핵심 개념 | 코드 파일 | 라인 번호 |
|---|---|---|---|
| **Practice02** | Plan-and-Solve: plan_node | `nodes.py` | 146-187 |
| **Practice02** | Plan-and-Solve: research_node | `nodes.py` | 190-254 |
| **Practice02** | Plan-and-Solve: synthesize_node | `nodes.py` | 257-280 |
| **Practice02** | Evaluator-Optimizer 루프 | `graph.py` | 56-63 |
| **Practice03** | Tool 정의 (@tool): search | `tools.py` | 333-353 |
| **Practice03** | Tool 정의 (@tool): budget | `tools.py` | 361-447 |
| **Practice03** | Tool 정의 (@tool): web | `tools.py` | 450-509 |
| **Practice03** | Tool Calling (bind_tools) | `nodes.py` | 194 |
| **Practice03** | Pydantic 스키마 (BudgetInput) | `tools.py` | 355-358 |
| **Practice05** | FAISS 벡터 스토어 초기화 | `tools.py` | 295-312 |
| **Practice05** | similarity_search | `tools.py` | 340 |
| **Practice05** | Agentic RAG | `nodes.py` | 221-243 |
| **Practice06** | Short-term Memory | `graph.py` | 70 |
| **Practice06** | Long-term Memory | `graph.py` | 27 |
| **Practice06** | 메시지 누적 | `state.py` | 20 |
| **Practice07** | Structured Output | `nodes.py` | 52, 95, 150, 287 |
| **Practice09** | LLM-as-Judge | `nodes.py` | 283-317 |
| **Practice09** | quality_score | `state.py` | 50-51 |

</details>

---

## 활동 5: 실습 구현

### 활동 5A: 새 도구 추가 - 날씨 예보 (초급)

**난이도:** ★☆☆ (초급)
**예상 시간:** 30분
**수정할 파일:** `agent/tools.py`

#### 목표
날씨 예보 도구를 추가하여 여행지의 계절별 날씨 정보를 제공합니다.

#### Step 1: 날씨 데이터베이스 추가

`tools.py`의 `BUDGET_DB` 아래에 다음 코드를 추가하세요:

```python
WEATHER_DB = {
    "제주도": {
        "봄": "🌸 3-5월: 평균 15°C, 유채꽃 시즌, 가끔 비",
        "여름": "☀️ 6-8월: 평균 28°C, 해수욕 최적, 태풍 주의",
        "가을": "🍂 9-11월: 평균 18°C, 억새꽃, 맑은 날씨",
        "겨울": "❄️ 12-2월: 평균 5°C, 한라산 설경, 바람 강함",
    },
    "부산": {
        "봄": "🌸 3-5월: 평균 14°C, 벚꽃 시즌, 화창함",
        "여름": "☀️ 6-8월: 평균 26°C, 해운대 해수욕, 장마",
        "가을": "🍂 9-11월: 평균 17°C, 불꽃축제, 쾌적함",
        "겨울": "❄️ 12-2월: 평균 4°C, 온화한 편, 건조함",
    },
    "도쿄": {
        "봄": "🌸 3-5월: 평균 15°C, 벚꽃 시즌 (3월말-4월초)",
        "여름": "☀️ 6-8월: 평균 28°C, 장마 (6-7월), 매우 습함",
        "가을": "🍂 9-11월: 평균 18°C, 단풍, 쾌적한 날씨",
        "겨울": "❄️ 12-2월: 평균 5°C, 건조, 맑은 날씨 많음",
    },
    "방콕": {
        "봄": "☀️ 3-5월: 평균 35°C, 가장 더운 시기",
        "여름": "🌧️ 6-8월: 평균 30°C, 우기, 스콜",
        "가을": "🌧️ 9-11월: 평균 28°C, 우기 후반, 습함",
        "겨울": "☀️ 12-2월: 평균 26°C, 건기, 여행 최적기",
    },
    "파리": {
        "봄": "🌸 3-5월: 평균 12°C, 꽃 만개, 변덕스러운 날씨",
        "여름": "☀️ 6-8월: 평균 22°C, 일조량 많음, 바캉스 시즌",
        "가을": "🍂 9-11월: 평균 12°C, 단풍, 비 자주",
        "겨울": "❄️ 12-2월: 평균 5°C, 추움, 크리스마스 마켓",
    },
}
```

#### Step 2: Pydantic 입력 스키마 정의

```python
class WeatherInput(BaseModel):
    destination: str = Field(description="여행지 이름 (예: '제주도', '도쿄')")
    season: str | None = Field(default=None, description="계절 (봄/여름/가을/겨울, 선택사항)")
```

#### Step 3: 도구 함수 구현

```python
@tool(args_schema=WeatherInput)
def get_weather_info(destination: str, season: str | None = None) -> str:
    """여행지의 계절별 날씨 정보를 제공합니다."""
    logger.info(f"[Tool Call] get_weather_info | destination='{destination}', season={season}")

    # TODO: 여러분이 구현하세요!
    # 1. WEATHER_DB에서 destination 찾기 (부분 매칭 허용)
    # 2. season이 주어지면 해당 계절만, 아니면 모든 계절 정보 반환
    # 3. 없는 여행지는 안내 메시지 반환

    pass  # 이 부분을 구현하세요
```

#### Step 4: RESEARCH_TOOLS에 추가

```python
RESEARCH_TOOLS = [search_travel_knowledge, estimate_budget, web_search, get_weather_info]
```

#### Step 5: 테스트

```bash
# 터미널에서 실행
cd travel_planning_agent
python -c "from agent.tools import get_weather_info; print(get_weather_info.invoke({'destination': '도쿄', 'season': '봄'}))"
```

#### 검증 체크리스트

- [ ] `get_weather_info("제주도")` → 4계절 모두 출력
- [ ] `get_weather_info("도쿄", "봄")` → 봄 정보만 출력
- [ ] `get_weather_info("런던")` → "날씨 정보가 없습니다" 출력
- [ ] `main.py` 실행 시 에러 없음

<details>
<summary>예시 구현 보기</summary>

```python
@tool(args_schema=WeatherInput)
def get_weather_info(destination: str, season: str | None = None) -> str:
    """여행지의 계절별 날씨 정보를 제공합니다."""
    logger.info(f"[Tool Call] get_weather_info | destination='{destination}', season={season}")

    # 여행지 매칭
    matched_destination = None
    for key in WEATHER_DB.keys():
        if destination in key or key in destination:
            matched_destination = key
            break

    if not matched_destination:
        available = ", ".join(WEATHER_DB.keys())
        return f"'{destination}' 날씨 정보가 없습니다. 지원 여행지: {available}"

    weather_data = WEATHER_DB[matched_destination]

    # 특정 계절 요청
    if season:
        season_normalized = season.strip()
        if season_normalized in weather_data:
            return f"🌍 {matched_destination} {season_normalized} 날씨\n\n{weather_data[season_normalized]}"
        else:
            return f"'{season}' 계절 정보가 없습니다. 가능한 계절: 봄, 여름, 가을, 겨울"

    # 모든 계절 정보
    result = f"🌍 {matched_destination} 계절별 날씨 정보\n\n"
    for s, info in weather_data.items():
        result += f"**{s}**\n{info}\n\n"

    return result.strip()
```

</details>

---

### 활동 5B: 새 노드 추가 - 입력 검증 (중급)

**난이도:** ★★☆ (중급)
**예상 시간:** 45분
**수정할 파일:** `agent/state.py`, `agent/nodes.py`, `agent/graph.py`

> **참고:** 아래 예시 구현은 하나의 접근 방식입니다. 여러분만의 방법으로 구현해도 됩니다.

#### 목표
사용자 입력을 검증하여 부적절한 입력을 조기에 필터링하는 노드를 추가합니다.

#### Step 1: state.py에 새 필드 추가

```python
class TravelPlanningState(TypedDict):
    # ... 기존 필드들 ...

    is_valid_input: bool
    """입력 유효성 검사 결과"""

    validation_message: str
    """유효성 검사 메시지 (실패 시)"""
```

#### Step 2: nodes.py에 검증 노드 함수 작성

```python
def validate_input_node(state: TravelPlanningState) -> dict:
    """사용자 입력을 검증합니다."""
    logger.info("[Node] validate_input 시작")
    query = state.get("user_input", "")

    # 검증 규칙 정의
    TRAVEL_KEYWORDS = [
        "여행", "관광", "숙소", "호텔", "맛집", "예산",
        "일정", "추천", "가볼만한", "교통", "항공", "투어",
        "비용", "경비", "코스", "명소"
    ]

    # TODO: 검증 로직 구현
    # 1. 최소 길이 검사 (2자 이상)
    # 2. 여행 관련 키워드 포함 여부
    # 3. 적절한 상태 반환

    pass  # 구현하세요
```

#### Step 3: graph.py 수정

```python
def create_travel_planning_graph(with_memory: bool = True):
    builder = StateGraph(TravelPlanningState)

    # 노드 추가 (validate_input을 맨 앞에)
    builder.add_node("validate_input", validate_input_node)  # 새로 추가!
    builder.add_node("classify_intent", classify_intent_node)
    # ... 나머지 노드들 ...

    # 엣지 수정: START -> validate_input
    builder.add_edge(START, "validate_input")

    # 조건부 엣지: 유효하면 classify로, 아니면 바로 종료
    builder.add_conditional_edges(
        "validate_input",
        lambda s: "valid" if s.get("is_valid_input", True) else "invalid",
        {
            "valid": "classify_intent",
            "invalid": "save_memory",
        }
    )

    # 기존 classify_intent의 START 엣지 제거 (이미 validate에서 연결됨)
    # ... 나머지 엣지들 ...
```

#### 검증 체크리스트

- [ ] 빈 입력 "" → `validation_message` 출력 후 종료
- [ ] "안녕하세요" → 여행 키워드 없음 안내
- [ ] "제주도 여행 추천해주세요" → 정상적으로 파이프라인 진행
- [ ] `main.py` 실행 시 정상 동작

<details>
<summary>예시 구현 보기</summary>

```python
def validate_input_node(state: TravelPlanningState) -> dict:
    """사용자 입력을 검증합니다."""
    logger.info("[Node] validate_input 시작")
    query = state.get("user_input", "")

    TRAVEL_KEYWORDS = [
        "여행", "관광", "숙소", "호텔", "맛집", "예산",
        "일정", "추천", "가볼만한", "교통", "항공", "투어",
        "비용", "경비", "코스", "명소"
    ]

    # 1. 최소 길이 검사
    if len(query.strip()) < 2:
        logger.warning("입력 길이 부족")
        return {
            "is_valid_input": False,
            "validation_message": "질문을 더 자세히 입력해주세요. 예: '제주도 3박4일 여행 계획'",
            "final_response": "질문을 더 자세히 입력해주세요.",
            "skip_to_end": True,
        }

    # 2. 여행 키워드 포함 여부
    has_travel_keyword = any(kw in query for kw in TRAVEL_KEYWORDS)

    if not has_travel_keyword:
        logger.warning("여행 관련 키워드 없음")
        return {
            "is_valid_input": False,
            "validation_message": "여행 관련 질문을 입력해주세요.",
            "final_response": "죄송합니다. 저는 여행 관련 질문에 답변드릴 수 있습니다. 예: '도쿄 여행 추천해줘'",
            "skip_to_end": True,
        }

    # 3. 유효한 입력
    logger.info("입력 검증 통과")
    return {
        "is_valid_input": True,
        "validation_message": "",
    }
```

</details>

---

### 활동 5C: 평가-최적화 루프 개선 (중급)

**난이도:** ★★☆ (중급)
**예상 시간:** 60분
**수정할 파일:** `agent/state.py`, `agent/nodes.py`, `agent/prompts.py`

> **참고:** 아래 예시 구현은 하나의 접근 방식입니다. 여러분만의 방법으로 구현해도 됩니다.

#### 목표
품질 평가를 세분화하여 정확성, 완성도, 관련성, 가독성 각각을 평가하고, 가장 취약한 영역을 집중 개선합니다.

#### Step 1: state.py - 세분화된 평가 스키마 추가

```python
class DetailedQualityEvaluation(BaseModel):
    """세분화된 응답 품질 평가 스키마"""

    accuracy_score: int = Field(description="정보 정확성 (1-10)", ge=1, le=10)
    completeness_score: int = Field(description="정보 완성도 (1-10)", ge=1, le=10)
    relevance_score: int = Field(description="질문 관련성 (1-10)", ge=1, le=10)
    readability_score: int = Field(description="가독성/구성 (1-10)", ge=1, le=10)

    overall_score: int = Field(description="종합 점수 (1-10)", ge=1, le=10)
    weakest_area: Literal["accuracy", "completeness", "relevance", "readability"] = Field(
        description="가장 개선이 필요한 영역"
    )
    improvement_suggestion: str = Field(description="구체적 개선 제안")
```

#### Step 2: state.py - 상태 필드 추가

```python
class TravelPlanningState(TypedDict):
    # ... 기존 필드들 ...

    detailed_scores: dict
    """세분화된 점수 {accuracy, completeness, relevance, readability}"""

    weakest_area: str
    """가장 취약한 영역"""
```

#### Step 3: prompts.py - 세분화된 평가 프롬프트 추가

```python
DETAILED_EVALUATION_PROMPT = """당신은 여행 상담 응답 품질 평가 전문가입니다.

다음 4가지 기준으로 응답을 평가하세요:

1. **정확성 (Accuracy)**: 제공된 정보가 사실과 일치하는가?
   - 여행지 정보, 가격, 시간 등이 정확한가?

2. **완성도 (Completeness)**: 질문에 필요한 모든 정보를 포함하는가?
   - 누락된 중요 정보가 없는가?

3. **관련성 (Relevance)**: 질문과 직접적으로 관련있는 내용인가?
   - 불필요한 정보가 포함되지 않았는가?

4. **가독성 (Readability)**: 읽기 쉽고 잘 구조화되어 있는가?
   - 명확한 제목, 목록, 문단 구분이 있는가?

각 항목을 1-10점으로 평가하고, 가장 점수가 낮은 영역에 대한 구체적인 개선 방안을 제시하세요.

사용자 질문: {query}

평가 대상 응답:
{response}
"""
```

#### Step 4: nodes.py - 세분화된 평가 노드

```python
def evaluate_response_detailed_node(state: TravelPlanningState) -> dict:
    """세분화된 품질 평가를 수행합니다."""
    logger.info("[Node] evaluate_response_detailed 시작")

    llm = ChatUpstage(model="solar-pro2", temperature=0.0)
    structured_llm = llm.with_structured_output(DetailedQualityEvaluation)

    # TODO: 구현하세요
    # 1. DETAILED_EVALUATION_PROMPT 사용
    # 2. DetailedQualityEvaluation 스키마로 결과 받기
    # 3. 세분화된 점수와 weakest_area 반환

    pass
```

#### Step 5: nodes.py - 타겟 개선 노드

```python
def improve_response_targeted_node(state: TravelPlanningState) -> dict:
    """취약한 영역을 집중적으로 개선합니다."""
    logger.info("[Node] improve_response_targeted 시작")

    weakest_area = state.get("weakest_area", "general")

    # 영역별 맞춤 개선 지침
    IMPROVEMENT_GUIDELINES = {
        "accuracy": "사실 정보를 재확인하고 정확한 데이터(가격, 시간, 장소명)로 수정하세요.",
        "completeness": "질문에 답하는 데 필요한 누락된 정보를 추가하세요.",
        "relevance": "질문과 관련 없는 불필요한 내용을 제거하세요.",
        "readability": "제목, 목록, 문단을 활용하여 구조를 개선하세요.",
    }

    # TODO: 타겟 개선 로직 구현

    pass
```

#### 검증 체크리스트

- [ ] 4가지 점수가 모두 출력됨
- [ ] `weakest_area`가 정확히 식별됨
- [ ] 해당 영역에 맞는 개선이 수행됨
- [ ] 기존 기능이 깨지지 않음

---

### 활동 5D: 새 노드 추가 - 개인화 추천 (고급)

**난이도:** ★★★ (고급)
**예상 시간:** 90분
**수정할 파일:** `agent/state.py`, `agent/nodes.py`, `agent/graph.py`

> **참고:** 이 활동은 정해진 정답이 없는 자유 활동입니다. 여러분만의 창의적인 방법으로 구현하세요.

#### 목표
사용자의 과거 여행 이력과 선호도를 분석하여 개인화된 추천을 제공하는 노드를 추가합니다.

#### 핵심 개념
- Long-term Memory 활용 (`user_profile`)
- 과거 패턴 분석
- LLM 기반 개인화

#### 구현 아이디어

```python
# state.py
class TravelPlanningState(TypedDict):
    # ...
    personalized_recommendations: List[dict]
    """개인화된 추천 [{type, recommendation, reason}]"""

# nodes.py
def personalize_recommendations_node(state: TravelPlanningState) -> dict:
    """사용자 프로필 기반 개인화 추천을 생성합니다."""

    user_profile = state.get("user_profile", {})
    preferred_destinations = user_profile.get("preferred_destinations", [])
    query_history = user_profile.get("query_history", [])
    current_preferences = state.get("extracted_preferences", {})

    # 개인화 로직 아이디어:
    # 1. 과거 방문지와 유사한 새로운 여행지 추천
    # 2. 과거 예산 패턴 기반 적정 예산 제안
    # 3. 선호 여행 스타일 반영

    # TODO: LLM을 활용한 개인화 추천 구현

# graph.py - 노드 위치 결정
# Option A: research 후, synthesize 전
# Option B: synthesize 후, evaluate 전
```

#### 설계 결정 과제
노드를 어디에 배치할지 결정하고, 그 이유를 설명하세요.

#### 검증 체크리스트

- [ ] 신규 사용자에게도 기본 추천 제공
- [ ] 기존 사용자는 과거 이력 기반 추천
- [ ] 추천 이유가 함께 출력됨
- [ ] 기존 워크플로우가 정상 동작

---

### 활동 5E: 안전 가드레일 노드 추가 (고급)

**난이도:** ★★★ (고급)
**예상 시간:** 90분
**참고:** Practice08-safety-guardrails.ipynb

> **참고:** 이 활동은 정해진 정답이 없는 자유 활동입니다. 여러분만의 창의적인 방법으로 구현하세요.

#### 목표
Practice08에서 배운 안전 가드레일을 적용하여 부적절한 입력/출력을 필터링합니다.

#### 핵심 개념
- 입력 안전 검사 (Input Guardrail)
- 출력 안전 검사 (Output Guardrail)
- Groundedness 검증 (Hallucination 방지)

#### 구현 아이디어

```python
# state.py
class TravelPlanningState(TypedDict):
    # ...
    safety_check_passed: bool
    flagged_content: List[str]
    is_grounded: bool

# nodes.py
def check_input_safety_node(state: TravelPlanningState) -> dict:
    """입력에서 부적절한 내용을 감지합니다."""
    # 규칙 기반 + LLM 기반 필터링
    pass

def check_output_safety_node(state: TravelPlanningState) -> dict:
    """출력이 제공된 정보에 기반하는지 검증합니다."""
    # Groundedness 검사: 응답이 tool_results에 기반하는지
    pass

# graph.py - 새로운 워크플로우
# START -> check_input_safety -> classify_intent -> ...
#                            ↓ (unsafe)
#                       save_memory -> END
#
# ... -> synthesize -> check_output_safety -> evaluate_response
#                            ↓ (hallucination)
#                       research (재조사)
```

#### 구현 힌트

1. **입력 안전 검사**: 불법/위험 키워드 목록 + LLM 판단
2. **출력 안전 검사**: tool_results와 final_response 비교
3. **Hallucination 감지**: 응답에 없는 정보가 포함되었는지 확인

#### 검증 체크리스트

- [ ] 부적절한 입력 시 적절한 메시지 출력
- [ ] Hallucination 감지 시 재조사 수행
- [ ] 정상 입력은 기존대로 처리
- [ ] 로깅이 적절히 수행됨

---

## 제출 체크리스트

### 빈칸 채우기 (1A, 1B, 1C)

- [ ] 모든 빈칸을 채웠는가?
- [ ] Python 문법 오류가 없는가?
- [ ] 실제 코드와 비교하여 검증했는가?
- [ ] 각 개념을 설명할 수 있는가?

### 개념 매핑 (3A)

- [ ] 모든 라인 번호를 찾았는가?
- [ ] Practice 노트북과 코드를 비교했는가?
- [ ] 개념 간 연결 관계를 이해했는가?

### 구현 실습 (5A-5E)

- [ ] 코드가 에러 없이 실행되는가?
- [ ] 테스트 케이스를 모두 통과하는가?
- [ ] 기존 기능이 깨지지 않았는가?
- [ ] `main.py` 실행 시 정상 동작하는가?
- [ ] 코드에 적절한 주석을 달았는가?
- [ ] 로깅을 추가했는가?

---

## 권장 학습 순서

| 주차 | 활동 | 학습 포인트 |
|------|------|------------|
| 1주차 | 1A, 1B, 1C | 코드 구조 이해 |
| 2주차 | 3A | Practice 노트북 ↔ 코드 연결 |
| 3주차 | 5A, 5B | 기본 확장 (도구, 노드 추가) |
| 4주차 | 5C, 5D, 5E | 고급 확장 (평가, 개인화, 안전) |

---

## 추가 참고 자료

- **TEACHING_GUIDE.md**: 교육자용 상세 가이드
- **ARCHITECTURE.md**: 기술 아키텍처 문서
- **WORKFLOW_DESIGN_GUIDE.md**: 워크플로우 설계 템플릿

---

*최종 수정일: 2025-01-31*
