# 여행 계획 에이전트

LangGraph 기반 Plan-and-Solve 여행 계획 AI 에이전트 프로젝트입니다.

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [워크플로우 구조](#워크플로우-구조)
3. [설치 및 실행](#설치-및-실행)
4. [프로젝트 구조](#프로젝트-구조)
5. [적용된 기술](#적용된-기술)
6. [모듈 상세 설명](#모듈-상세-설명)
7. [확장 방법](#확장-방법)

---

## 프로젝트 개요

이 프로젝트는 Jupyter Notebook(Practice01~09)에서 학습한 내용을 통합하여 구현한 **여행 계획 에이전트**입니다.

### 주요 기능

| 기능 | 설명 | 관련 기술 |
|------|------|----------|
| 의도 분류 | 문의를 4가지 카테고리로 분류 | Structured Output |
| Plan-and-Solve | 계획 수립 → 조사 → 종합 파이프라인 | Plan-and-Solve Prompting |
| RAG 검색 | FAISS 벡터 스토어 기반 여행 지식 검색 | Agentic RAG |
| 도구 호출 | 3개 도구 + 방어적 폴백 전략 | Tool Calling |
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
│   ┌─────────────────┐                                           │
│   │ plan_and_solve  │ ← Plan-and-Solve: Plan 단계              │
│   │                 │   실행 계획(plan_steps) 생성               │
│   └────────┬────────┘                                           │
│            │  plan_steps 전달                                   │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │    research     │ ← Plan-and-Solve: Solve 단계             │
│   │                 │   도구 호출 (RAG + 날씨 + 관광지 + 예산)  │
│   │                 │   폴백: plan_steps 기반 직접 실행          │
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

```bash
# 대화형 모드
uv run python main.py

# 단일 쿼리
uv run python main.py --query "제주도 3박 4일 여행 계획 세워줘"

# 빠른 모드 (품질 평가 생략)
uv run python main.py --fast

# 데모 실행
uv run python main.py --demo

# 워크플로우 구조만 확인
uv run python main.py --workflow
```

### 4. 대화형 모드 명령어

```
/help              도움말 표시
/search 제주도      여행 지식 검색
/weather 도쿄 7월   날씨 정보 조회
/profile           사용자 프로필 확인 (장기 메모리)
/workflow          워크플로우 구조 표시
/quit              종료
```

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
└── agent/
    ├── __init__.py       # 패키지 초기화 및 공개 API
    ├── state.py          # 상태 정의 (TypedDict, Pydantic)
    ├── prompts.py        # 시스템 프롬프트 (Plan-and-Solve 포함)
    ├── tools.py          # 도구 정의 (RAG, 날씨, 관광지, 예산)
    ├── nodes.py          # 노드 함수들 (핵심 로직)
    └── graph.py          # 워크플로우 그래프 구성
```

---

## 적용된 기술

### 1. Plan-and-Solve 프롬프팅

**적용 위치**: `nodes.py` - `plan_and_solve_node()`, `research_node()`, `synthesize_node()`

Plan-and-Solve는 복잡한 문제를 3단계로 나누어 해결하는 프롬프팅 기법입니다:

```python
# Plan 단계: 실행 계획 수립
def plan_and_solve_node(state):
    structured_llm = llm.with_structured_output(TravelPlan)
    plan = structured_llm.invoke([...])
    return {"plan_steps": plan.steps}  # research_node에 전달

# Solve 단계: 계획에 따라 도구 호출
def research_node(state):
    plan_steps = state["plan_steps"]  # Plan 단계에서 받은 계획
    # plan_steps를 프롬프트에 포함하여 도구 호출

# Synthesize 단계: 결과 종합
def synthesize_node(state):
    # plan + research 결과를 종합하여 최종 응답 생성
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

### 3. 도구 호출 + 방어적 폴백

**적용 위치**: `nodes.py` - `research_node()`

```python
def research_node(state):
    llm_with_tools = llm.bind_tools(RESEARCH_TOOLS)
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        # 전략 1: LLM이 반환한 tool_calls 실행
        for tool_call in response.tool_calls:
            result = tool_map[tool_call["name"]].invoke(tool_call["args"])
    else:
        # 전략 2 (폴백): plan_steps 기반으로 직접 도구 실행
        _execute_tools_from_plan(query, plan_steps)
```

### 4. 이중 메모리 시스템

**적용 위치**: `graph.py`, `nodes.py` - `save_memory_node()`

```python
# 단기 메모리: MemorySaver 체크포인터 (자동)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "user-123"}}

# 장기 메모리: USER_PROFILES 딕셔너리 (수동)
USER_PROFILES = {}  # {user_id: {preferred_destinations, query_history, ...}}

def save_memory_node(state, config):
    user_id = config["configurable"]["user_id"]
    USER_PROFILES[user_id]["query_history"].append(...)
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
builder.add_edge("classify_intent", "plan_and_solve")
builder.add_edge("plan_and_solve", "research")
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

WEATHER_DB["여수"] = {"7월": "평균 26°C, ..."}
ATTRACTIONS_DB["여수"] = {"관광": [...], "맛집": [...]}
BUDGET_DB["여수"] = {"moderate": {...}}
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
