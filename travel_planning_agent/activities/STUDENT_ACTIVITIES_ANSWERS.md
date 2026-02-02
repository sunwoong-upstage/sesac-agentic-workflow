# 여행 계획 에이전트 학습 활동지 - 정답지

> **주의:** 이 문서는 교육자용 정답지입니다. 학생에게는 배포하지 마세요.

---

## 목차

1. [활동 1A: 상태(State) 정의 - 정답](#활동-1a-상태state-정의---정답)
2. [활동 1B: 도구(Tool) 정의 - 정답](#활동-1b-도구tool-정의---정답)
3. [활동 1C: 그래프 구성 - 정답](#활동-1c-그래프-구성---정답)
4. [활동 3A: 개념 매핑 테이블 - 정답](#활동-3a-개념-매핑-테이블---정답)
5. [활동 5A: 날씨 도구 추가 - 예시 구현](#활동-5a-새-도구-추가---날씨-예보---예시-구현)
6. [활동 5B: 입력 검증 노드 추가 - 예시 구현](#활동-5b-새-노드-추가---입력-검증---예시-구현)
7. [활동 5C: 평가 시스템 개선 - 자유 활동](#활동-5c-평가-최적화-루프-개선---자유-활동)
8. [활동 5D: 개인화 추천 노드 - 자유 활동](#활동-5d-새-노드-추가---개인화-추천---자유-활동)
9. [활동 5E: 안전 가드레일 - 자유 활동](#활동-5e-안전-가드레일-노드-추가---자유-활동)

---

## 활동 1A: 상태(State) 정의 - 정답

**파일:** `agent/state.py`

### 문제 1: 기본 import 완성하기

```python
import operator
from typing import Annotated, List, Literal
from pydantic import BaseModel, Field
```

### 문제 2: 메시지 누적을 위한 상태 필드 정의

```python
class TravelPlanningState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    tool_results: Annotated[List[dict], operator.add]
    user_input: str
    final_response: str
    quality_score: int
```

### 문제 3: 의도 분류 스키마 정의 (Pydantic)

```python
class IntentClassification(BaseModel):
    intent: Literal[
        "destination_research",
        "itinerary_planning",
        "budget_estimation",
        "general_travel"
    ] = Field(description="문의 유형")
```

### 문제 4: 여행 계획 스키마 - 필드 검증 추가

```python
class TravelPlan(BaseModel):
    destination: str = Field(description="주요 여행지")
    duration_days: int = Field(description="여행 기간 (일수)", ge=1)
    steps: List[str] = Field(description="실행 계획 단계")
```

### 문제 5: 선호도 추출 스키마 - Optional 필드

```python
class ExtractedPreferences(BaseModel):
    destination: str | None = Field(default=None, description="여행지")
    budget: int | None = Field(default=None, description="예산 (원화)", ge=0)
```

---

## 활동 1B: 도구(Tool) 정의 - 정답

**파일:** `agent/tools.py`

### 문제 1: 도구 입력 스키마 정의

```python
from langchain_core.tools import tool

class TravelSearchInput(BaseModel):
    query: str = Field(description="검색 쿼리 (예: '제주도 여행', '환전 팁')")

class BudgetInput(BaseModel):
    destination: str = Field(description="여행지 이름")
    duration_days: int = Field(description="여행 기간 (일수)", ge=1)
    user_budget: int | None = Field(default=None, description="사용자 예산 (선택)")
```

### 문제 2: FAISS 벡터 스토어 초기화

```python
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS

_vector_store = FAISS.from_documents(documents, embeddings)
```

### 문제 3: 검색 도구 정의

```python
@tool(args_schema=TravelSearchInput)
def search_travel_knowledge(query: str) -> str:
    docs = vector_store.similarity_search(query, k=3)
    result = "\n\n".join(
        f"[{doc.metadata.get('category')}] {doc.metadata.get('title')}\n{doc.page_content}"
        for doc in docs
    )
```

### 문제 4: 예산 추정 도구

```python
for key in BUDGET_DB.keys():
    if destination in key or key in destination:
        matched_destination = key
        break

for cost_item, daily_cost in option.items():
    if "항공" in cost_item:
        cost = daily_cost
    else:
        cost = daily_cost * duration_days
    total += cost
```

### 문제 5: 웹 검색 도구

```python
import requests

api_key = os.getenv("SERPER_API_KEY")

headers = {
    "X-API-KEY": api_key,
    "Content-Type": "application/json",
}

response = requests.post(
    "https://google.serper.dev/search",
    headers=headers,
    json=payload,
    timeout=10,
)
response.raise_for_status()
```

### 문제 6: 도구 리스트 정의

```python
RESEARCH_TOOLS = [search_travel_knowledge, estimate_budget, web_search]
```

---

## 활동 1C: 그래프 구성 - 정답

**파일:** `agent/graph.py`

### Import 문

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
```

### 문제 1: 노드 함수 import

```python
from agent.nodes import (
    classify_intent_node,
    extract_preferences_node,
    plan_node,
    research_node,
    synthesize_node,
    evaluate_response_node,
    improve_response_node,
    save_memory_node,
    should_improve_response,
)
```

### 문제 2: StateGraph 생성

```python
builder = StateGraph(TravelPlanningState)
```

### 문제 3: 노드 추가

```python
builder.add_node("classify_intent", classify_intent_node)
builder.add_node("extract_preferences", extract_preferences_node)
builder.add_node("plan", plan_node)
builder.add_node("research", research_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("evaluate_response", evaluate_response_node)
builder.add_node("improve_response", improve_response_node)
builder.add_node("save_memory", save_memory_node)
```

### 문제 4: 시작 엣지

```python
builder.add_edge(START, "classify_intent")
```

### 문제 5: 조건부 엣지 (빈 입력 처리)

```python
builder.add_conditional_edges(
    "classify_intent",
    lambda s: "skip" if s.get("skip_to_end") else "continue",
    {"continue": "extract_preferences", "skip": "save_memory"}
)
```

### 문제 6: 순차 엣지 (Plan-and-Solve 파이프라인)

```python
builder.add_edge("extract_preferences", "plan")
builder.add_edge("plan", "research")
builder.add_edge("research", "synthesize")
builder.add_edge("synthesize", "evaluate_response")
```

### 문제 7: 조건부 엣지 (품질 평가 후 분기)

```python
builder.add_conditional_edges(
    "evaluate_response",
    should_improve_response,
    {
        "improve": "improve_response",
        "end": "save_memory",
    }
)
```

### 문제 8: 개선 후 재평가 루프

```python
builder.add_edge("improve_response", "evaluate_response")
```

### 문제 9: 종료 엣지

```python
builder.add_edge("save_memory", END)
```

### 문제 10: 그래프 컴파일

```python
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, store=user_store)
```

---

## 활동 3A: 개념 매핑 테이블 - 정답

| Practice 노트북 | 핵심 개념 | 코드 파일 | 라인 번호 |
|---|---|---|---|
| **Practice01** | AI 에이전트 정의 | `main.py` | 전체 |
| **Practice02** | Workflow vs Agent | `graph.py` | 30-76 |
| **Practice02** | Plan-and-Solve: plan_node | `nodes.py` | **146-187** |
| **Practice02** | Plan-and-Solve: research_node | `nodes.py` | **190-254** |
| **Practice02** | Plan-and-Solve: synthesize_node | `nodes.py` | **257-280** |
| **Practice02** | Evaluator-Optimizer 루프 | `graph.py` | **56-63** |
| **Practice03** | Tool 정의 (@tool): search | `tools.py` | **333-353** |
| **Practice03** | Tool 정의 (@tool): budget | `tools.py` | **361-447** |
| **Practice03** | Tool 정의 (@tool): web | `tools.py` | **450-509** |
| **Practice03** | Tool Calling (bind_tools) | `nodes.py` | **194** |
| **Practice03** | Pydantic 스키마 (BudgetInput) | `tools.py` | **355-358** |
| **Practice05** | FAISS 벡터 스토어 초기화 | `tools.py` | **295-312** |
| **Practice05** | similarity_search | `tools.py` | **340** |
| **Practice05** | Agentic RAG (LLM이 도구 결정) | `nodes.py` | **221-243** |
| **Practice06** | Short-term Memory (MemorySaver) | `graph.py` | **70** |
| **Practice06** | Long-term Memory (InMemoryStore) | `graph.py` | **27** |
| **Practice06** | 메시지 누적 (operator.add) | `state.py` | **20** |
| **Practice07** | Structured Output | `nodes.py` | **52, 95, 150, 287** |
| **Practice09** | LLM-as-Judge | `nodes.py` | **283-317** |
| **Practice09** | quality_score 필드 | `state.py` | **50-51** |

---

## 활동 5A: 새 도구 추가 - 날씨 예보 - 예시 구현

**파일:** `agent/tools.py`

### 완성된 코드

```python
# Step 1: 날씨 데이터베이스 (BUDGET_DB 아래에 추가)
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


# Step 2: Pydantic 입력 스키마
class WeatherInput(BaseModel):
    destination: str = Field(description="여행지 이름 (예: '제주도', '도쿄')")
    season: str | None = Field(default=None, description="계절 (봄/여름/가을/겨울, 선택사항)")


# Step 3: 도구 함수 구현
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


# Step 4: RESEARCH_TOOLS 수정
RESEARCH_TOOLS = [search_travel_knowledge, estimate_budget, web_search, get_weather_info]
```

### 테스트 결과 예시

```bash
$ python -c "from agent.tools import get_weather_info; print(get_weather_info.invoke({'destination': '도쿄', 'season': '봄'}))"

🌍 도쿄 봄 날씨

🌸 3-5월: 평균 15°C, 벚꽃 시즌 (3월말-4월초)
```

---

## 활동 5B: 새 노드 추가 - 입력 검증 - 예시 구현

**파일:** `agent/state.py`, `agent/nodes.py`, `agent/graph.py`

### state.py 수정

```python
class TravelPlanningState(TypedDict):
    # ... 기존 필드들 ...

    is_valid_input: bool
    """입력 유효성 검사 결과"""

    validation_message: str
    """유효성 검사 메시지 (실패 시)"""
```

### nodes.py 추가

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

### graph.py 수정

```python
from agent.nodes import (
    validate_input_node,  # 추가
    classify_intent_node,
    # ... 나머지 import
)

def create_travel_planning_graph(with_memory: bool = True):
    builder = StateGraph(TravelPlanningState)

    # 노드 추가 (validate_input을 맨 앞에)
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("extract_preferences", extract_preferences_node)
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("evaluate_response", evaluate_response_node)
    builder.add_node("improve_response", improve_response_node)
    builder.add_node("save_memory", save_memory_node)

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

    # 기존 classify_intent 조건부 엣지 (그대로 유지)
    builder.add_conditional_edges(
        "classify_intent",
        lambda s: "skip" if s.get("skip_to_end") else "continue",
        {"continue": "extract_preferences", "skip": "save_memory"}
    )

    # 나머지 엣지들 (그대로 유지)
    builder.add_edge("extract_preferences", "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "synthesize")
    builder.add_edge("synthesize", "evaluate_response")

    builder.add_conditional_edges(
        "evaluate_response",
        should_improve_response,
        {"improve": "improve_response", "end": "save_memory"}
    )

    builder.add_edge("improve_response", "evaluate_response")
    builder.add_edge("save_memory", END)

    # ... 컴파일 로직
```

---

## 활동 5C: 평가-최적화 루프 개선 - 자유 활동

> **이 활동은 정해진 정답이 없는 자유 활동입니다.**
>
> 학생들이 자유롭게 구현하도록 하고, 다양한 접근 방식을 토론하세요.

### 평가 포인트

1. `DetailedQualityEvaluation` 스키마가 올바르게 정의되었는가?
2. 4가지 평가 기준이 모두 구현되었는가?
3. `weakest_area`를 기반으로 타겟 개선이 이루어지는가?
4. 기존 워크플로우와 호환되는가?

---

## 활동 5D: 새 노드 추가 - 개인화 추천 - 자유 활동

> **이 활동은 정해진 정답이 없는 자유 활동입니다.**
>
> 학생들이 자유롭게 구현하도록 하고, 다양한 접근 방식을 토론하세요.

### 평가 포인트

1. `user_profile`에서 정보를 올바르게 읽어오는가?
2. 개인화 추천이 의미있는가? (단순 반복이 아닌 분석 기반)
3. 신규 사용자에 대한 처리가 있는가?
4. 노드 위치 결정에 대한 근거가 합리적인가?

### 노드 위치 토론 포인트

**Option A: research → personalize → synthesize**
- 장점: 조사 결과에 개인화 정보를 추가할 수 있음
- 단점: synthesize 프롬프트 수정 필요

**Option B: synthesize → personalize → evaluate**
- 장점: 응답 생성 후 개인화 추가, 독립적 모듈
- 단점: 응답 구조가 일관되지 않을 수 있음

---

## 활동 5E: 안전 가드레일 노드 추가 - 자유 활동

> **이 활동은 정해진 정답이 없는 자유 활동입니다.**
>
> 학생들이 자유롭게 구현하도록 하고, 다양한 접근 방식을 토론하세요.

### 평가 포인트

1. 입력 안전 검사가 적절한 키워드/패턴을 감지하는가?
2. Groundedness 검사가 hallucination을 감지하는가?
3. 안전하지 않은 입력에 대한 응답이 적절한가?
4. 성능 영향이 합리적인가? (과도한 LLM 호출 없음)

### 토론 포인트

1. **규칙 기반 vs LLM 기반**: 어떤 방식이 더 효과적인가?
2. **False Positive**: 정상 입력이 차단되는 경우 어떻게 처리할 것인가?
3. **Hallucination 정의**: 어느 수준의 창작이 허용되는가?

---

*최종 수정일: 2025-01-31*
