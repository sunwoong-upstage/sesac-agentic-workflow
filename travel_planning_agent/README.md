# 여행 계획 에이전트

LangGraph 기반 Plan-and-Solve 여행 계획 AI 에이전트 프로젝트입니다.

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [워크플로우 구조](#워크플로우-구조)
3. [설치 및 실행](#설치-및-실행)
4. [평가 (Evaluation)](#평가-evaluation)
5. [프로젝트 구조](#프로젝트-구조)
6. [적용된 기술](#적용된-기술)
7. [모듈 상세 설명](#모듈-상세-설명)
8. [확장 방법](#확장-방법)

---

## 프로젝트 개요

이 프로젝트는 Jupyter Notebook(Practice01~09)에서 학습한 내용을 통합하여 구현한 **여행 계획 에이전트**입니다.

### 주요 기능

| 기능 | 설명 | 관련 기술 |
|------|------|----------|
| 의도 분류 | 문의를 4가지 카테고리로 분류 | Structured Output |
| Plan-and-Solve | 계획 수립 → 조사 → 종합 파이프라인 | Plan-and-Solve Prompting |
| RAG 검색 | FAISS 벡터 스토어 기반 여행 지식 검색 | Agentic RAG |
| 도구 호출 | 3개 도구 (RAG, 예산, 웹 검색) | Tool Calling |
| 이중 메모리 | 단기(대화) + 장기(사용자 프로필) | Memory Management |
| 품질 평가 | 응답 품질 평가 및 개선 루프 | Evaluator-Optimizer |

### 지원 여행지

- **국내**: 제주도, 부산, 경주, 강릉, 서울
- **해외**: 도쿄, 오사카, 방콕, 다낭, 파리

---

## 워크플로우 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   START                                                         │
│     │                                                           │
│     ▼                                                           │
│   ┌─────────────────┐                                           │
│   │ classify_intent │ ← 의도 분류 (Structured Output)           │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────────┐                                       │
│   │ extract_preferences │ ← 사용자 선호도 추출                  │
│   └────────┬────────────┘                                       │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │      plan       │ ← Plan-and-Solve: Plan 단계              │
│   │                 │   실행 계획(plan_steps) 생성               │
│   └────────┬────────┘                                           │
│            │  plan_steps 전달                                   │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │    research     │ ← Plan-and-Solve: Solve 단계             │
│   │                 │   LLM이 필요한 도구 자율 호출             │
│   │                 │   (RAG 검색, 예산 추정, 웹 검색)          │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │   synthesize    │ ← Plan-and-Solve: Synthesize 단계        │
│   │                 │   계획 + 조사 결과 종합 → 최종 응답        │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌───────────────────────┐                                     │
│   │  evaluate_response    │ ← Evaluator-Optimizer 패턴          │
│   └───────────┬───────────┘                                     │
│               │                                                 │
│    ┌──────────┴──────────┐                                      │
│    │  pass? (7점 이상)   │                                      │
│    └──────────┬──────────┘                                      │
│         Yes │          │ No                                     │
│             ▼          ▼                                        │
│   ┌──────────────┐  ┌─────────────────┐                        │
│   │ save_memory  │  │improve_response │                        │
│   │              │  └────────┬────────┘                        │
│   │ 단기: 체크포인터│         │                                  │
│   │ 장기: 프로필    │         └──► evaluate_response             │
│   └──────┬───────┘                                             │
│          │                                                      │
│          ▼                                                      │
│        END                                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 설치 및 실행

### 1. 의존성 설치

```bash
cd travel_planning_agent
uv sync
```

### 2. 환경 변수 설정

`.env` 파일을 편집하여 API 키를 설정합니다:

```env
UPSTAGE_API_KEY=your-api-key-here
```

### 3. 실행

#### LangGraph Studio로 실행 (권장)

```bash
langgraph dev
```

브라우저에서 `http://localhost:2024`로 자동 접속되며, 시각화된 워크플로우를 확인할 수 있습니다.

**사용 예시**:
- "제주도 3박 4일 여행 계획 세워줘"
- "도쿄 예산 50만원으로 갈 수 있을까?"
- "부산 맛집 추천해줘"

각 노드의 실행 과정, State 변화, Tool Call 정보를 실시간으로 확인할 수 있습니다.

#### Python 스크립트 실행 (데모용)

```bash
uv run python main.py
```

고정된 2개의 데모 쿼리를 실행하여 멀티턴 대화와 메모리 시스템을 시연합니다.

---

## 평가 (Evaluation)

LangSmith를 사용한 에이전트 품질 평가 프레임워크입니다.

### 평가 실행

```bash
python -m evaluate.run
```

### 평가 데이터셋 (20개 예제)

| 카테고리 | 예제 수 | 예시 질문 |
|----------|---------|-----------|
| Budget Estimation | 5 | "제주도 3박4일 예산은?", "도쿄 여행 비용 알려줘" |
| Destination Research | 5 | "제주도 맛집 추천", "교토 벚꽃 명소" |
| Itinerary Planning | 5 | "부산 2박3일 일정", "파리 1주일 계획" |
| General Travel | 5 | "여행 준비물", "환전 어디서?" |

### 평가 기준 (LLM-as-Judge)

| 평가자 | 설명 | 기준 |
|--------|------|------|
| **Correctness** | 정답과 의미적으로 일치하는지 | 핵심 정보(예산, 장소) 일치 |
| **Groundedness** | 검색된 문서에 기반하는지 | 환각(hallucination) 없음 |
| **Concision** | 답변 길이가 적절한지 | 정답의 3배 이하 |

### 평가 결과 예시

```
Total examples: 20
Correctness:   90%  ✓ 사실적으로 정확한 답변
Groundedness:  25%  △ 검색 결과 외 정보 추가 경향
Concision:      0%  × 상세한 답변 (의도된 동작)
```

### 평가 결과 확인

평가 완료 후 LangSmith 대시보드에서 상세 결과 확인:
- https://smith.langchain.com/

---

## 프로젝트 구조

```
travel_planning_agent/
├── .env                  # API 키 설정
├── .python-version       # Python 버전 (3.11)
├── pyproject.toml        # 프로젝트 설정
├── uv.lock               # 의존성 잠금
├── README.md             # 이 문서
├── main.py               # 실행 진입점
├── agent/                # 에이전트 코드
│   ├── __init__.py       # 패키지 초기화 및 공개 API
│   ├── state.py          # 상태 정의 (TypedDict, Pydantic)
│   ├── prompts.py        # 시스템 프롬프트 (Plan-and-Solve 포함)
│   ├── tools.py          # 도구 정의 (RAG, 예산, 웹 검색)
│   ├── nodes.py          # 노드 함수들 (핵심 로직)
│   └── graph.py          # 워크플로우 그래프 구성
├── evaluate/             # 에이전트 평가 (LangSmith)
│   ├── dataset.py        # 평가 데이터셋 (20개 Q&A)
│   ├── evaluators.py     # LLM-as-Judge 평가 함수
│   └── run.py            # 평가 실행 스크립트
└── tests/                # 코드 테스트 (pytest)
    ├── conftest.py       # pytest 설정 (.env 로드)
    ├── test_edge_cases.py
    └── test_multiturn.py
```

> **참고**: `tests/`는 코드 검증, `evaluate/`는 에이전트 품질 평가 용도입니다.

---

## 적용된 기술

### 1. Plan-and-Solve 프롬프팅

**적용 위치**: `nodes.py` - `plan_node()`, `research_node()`, `synthesize_node()`

Plan-and-Solve는 복잡한 문제를 3단계로 나누어 해결하는 프롬프팅 기법입니다:

```python
# Plan 단계: 실행 계획 수립
def plan_node(state):
    structured_llm = llm.with_structured_output(TravelPlan)
    plan = structured_llm.invoke([...])
    return {"plan_steps": plan.steps}

# Solve 단계: 계획에 따라 도구 호출
def research_node(state):
    plan_steps = state["plan_steps"]
    llm_with_tools.invoke([SystemMessage(prompt), *messages])

# Synthesize 단계: 결과 종합
def synthesize_node(state):
    llm.invoke([SystemMessage(prompt), HumanMessage(query)])
```

### 2. FAISS 벡터 스토어 (Agentic RAG)

**적용 위치**: `tools.py` - `search_travel_knowledge()`

```python
# Lazy Initialization: 첫 검색 시 초기화
def _get_vector_store():
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    _vector_store = FAISS.from_documents(docs, embeddings)
    return _vector_store

# 벡터 검색 + 키워드 폴백
@tool
def search_travel_knowledge(query: str) -> str:
    vs = _get_vector_store()
    if vs:
        docs = vs.similarity_search(query, k=3)
        return format(docs)
    return _keyword_fallback_search(query)  # 폴백
```

### 3. LLM 도구 호출 (Tool Calling)

**적용 위치**: `nodes.py` - `research_node()`

```python
def research_node(state):
    llm_with_tools = llm.bind_tools(RESEARCH_TOOLS)
    response = llm_with_tools.invoke(messages)

    # LLM이 필요한 도구를 자율적으로 선택하여 호출
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = tool_map[tool_call["name"]].invoke(tool_call["args"])
```

### 4. 이중 메모리 시스템

**적용 위치**: `graph.py`, `nodes.py` - `save_memory_node()`

```python
# 단기 메모리: MemorySaver 체크포인터 (자동)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

memory = MemorySaver()
user_store = InMemoryStore()
graph = builder.compile(checkpointer=memory, store=user_store)
config = {"configurable": {"thread_id": "user-123", "user_id": "profile-123"}}

# 장기 메모리: InMemoryStore (사용자 프로필)
def run_travel_planning(query, thread_id, user_id):
    # 호출 전: 기존 프로필 로드
    existing_profile = user_store.get(("users",), user_id)

    # 그래프 실행
    result = graph.invoke(
        {"messages": [HumanMessage(query)]},
        config={"configurable": {"thread_id": thread_id, "user_id": user_id}}
    )

    # 호출 후: 업데이트된 프로필 저장
    user_store.put(("users",), user_id, result["user_profile"])
    return result
```

### 5. Evaluator-Optimizer 패턴

**적용 위치**: `nodes.py` - `evaluate_response_node()`, `improve_response_node()`

```python
# 평가자: 품질 점수 산출
result = structured_llm.invoke(evaluation_prompt)
passed = result.score >= 7

# 조건부 라우팅: 7점 미만이면 개선 루프
builder.add_conditional_edges(
    "evaluate_response",
    should_improve_response,
    {"improve": "improve_response", "end": "save_memory"}
)
```

---

## 모듈 상세 설명

### state.py - 상태 정의

```python
class TravelPlanningState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # 대화 누적
    user_input: str
    travel_plan: str           # Plan-and-Solve 계획
    plan_steps: List[str]      # 실행 단계 리스트
    intent: str                # 의도 분류 결과
    tool_results: Annotated[List[dict], operator.add]  # 도구 결과 누적
    retrieved_context: str     # RAG 검색 결과
    quality_score: int         # 품질 점수
    evaluation_passed: bool    # 통과 여부
    user_profile: dict         # 장기 메모리
    error_log: Annotated[List[str], operator.add]  # 에러 누적
```

### tools.py - 도구 정의

| 도구 | 용도 | 데이터 소스 |
|------|------|-----------|
| `search_travel_knowledge` | 여행 지식 검색 (RAG) | FAISS 벡터 스토어 |
| `estimate_budget` | 여행 예산 추정 | Mock 데이터 |
| `web_search` | 실시간 웹 검색 | Serper API |

### graph.py - 그래프 구성

```python
# 선형 Plan-and-Solve 파이프라인
builder.add_edge(START, "classify_intent")
builder.add_edge("classify_intent", "extract_preferences")
builder.add_edge("extract_preferences", "plan")
builder.add_edge("plan", "research")
builder.add_edge("research", "synthesize")
builder.add_edge("synthesize", "evaluate_response")

# 평가-개선 루프
builder.add_conditional_edges(
    "evaluate_response", should_improve_response,
    {"improve": "improve_response", "end": "save_memory"}
)
```

---

## 확장 방법

### 1. 새로운 여행지 추가

```python
# tools.py에 데이터 추가
TRAVEL_KNOWLEDGE_BASE.append({
    "id": "KR-006",
    "category": "국내여행",
    "title": "여수 여행 가이드",
    "content": "여수는..."
})

BUDGET_DB["여수"] = {
    "options": [
        {"name": "저예산", "숙박": 60000, ...},
        {"name": "중예산", "숙박": 90000, ...},
    ]
}
```

### 2. 실제 API 연동

```python
# tools.py의 Mock 데이터를 실제 API로 교체
import requests

@tool
def get_weather_info(destination: str, month: str) -> str:
    response = requests.get(f"https://api.weather.com/...")
    return response.json()
```

### 3. 영속적 메모리 구현

```python
# nodes.py의 USER_PROFILES를 Redis로 교체
import redis

r = redis.Redis()

def save_memory_node(state, config):
    user_id = config["configurable"]["user_id"]
    r.hset(f"user:{user_id}", mapping=profile)
```

### 4. 벡터 스토어 영속화

```python
# tools.py에서 FAISS 인덱스를 파일로 저장/로드
vector_store = FAISS.from_documents(docs, embeddings)
vector_store.save_local("faiss_index")  # 저장

vector_store = FAISS.load_local("faiss_index", embeddings)  # 로드
```

---

## 참고 자료

- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)
- [LangChain 문서](https://python.langchain.com/)
- [Upstage API](https://developers.upstage.ai/)
- [Plan-and-Solve Prompting (Wang et al., 2023)](https://arxiv.org/abs/2305.04091)
