# 여행 계획 AI 에이전트 - 강의 가이드

> **대상**: Python 에이전트 교육 강사  
> **목표**: `agent/` 폴더의 각 모듈을 단계별로 설명하고 학습 포인트 강조

---

## 📚 전체 구조 개요

```
travel_planning_agent/
├── main.py              # ⭐ 실행 진입점 (데모 시작)
└── agent/
    ├── __init__.py      # 패키지 초기화 및 공개 API
    ├── state.py         # ⭐ 1단계: 상태 정의 (TypedDict, Pydantic)
    ├── prompts.py       # ⭐ 2단계: 프롬프트 엔지니어링
    ├── tools.py         # ⭐ 3단계: 도구 정의 (RAG, 예산, 웹 검색)
    ├── nodes.py         # ⭐ 4단계: 노드 함수 (핵심 로직)
    └── graph.py         # ⭐ 5단계: 워크플로우 그래프 구성
```

---

## 🎯 교육 진행 순서 (권장)

### **1일차: 기초 - 상태와 프롬프트**
- ✅ `state.py`: 상태 관리의 개념
- ✅ `prompts.py`: 프롬프트 엔지니어링 기초

### **2일차: 도구와 RAG**
- ✅ `tools.py`: Tool Calling, RAG (FAISS), 웹 검색 API

### **3일차: 노드와 워크플로우**
- ✅ `nodes.py`: Plan-and-Solve, Evaluator-Optimizer 패턴
- ✅ `graph.py`: LangGraph 워크플로우, 이중 메모리

### **4일차: 실습 및 실행**
- ✅ `main.py`: 전체 실행 및 디버깅
- ✅ 학생 프로젝트 시작

---

## 📖 모듈별 강의 가이드

---

## 1️⃣ `state.py` - 상태 정의 (40분)

### 🎓 학습 목표
- TypedDict를 사용한 상태 정의
- Pydantic을 사용한 구조화된 출력(Structured Output)
- `operator.add`를 활용한 상태 누적

### 📝 주요 개념

#### 1.1. `TravelPlanningState` (TypedDict)
```python
class TravelPlanningState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # 누적 필드
    user_input: str  # 단일 값 필드
    extracted_preferences: dict  # 다중 턴 대화 지원
```

**강의 포인트:**
- ✅ `TypedDict`: 딕셔너리 타입 힌트 (타입 안전성)
- ✅ `Annotated[..., operator.add]`: 값을 누적하는 필드 (append)
- ✅ `messages`: LangChain의 메시지 히스토리를 저장
- ✅ `extracted_preferences`: 다중 턴 대화에서 사용자 정보 누적

**실습 질문:**
> Q: `messages`는 왜 `operator.add`를 사용할까요?  
> A: 대화가 진행될수록 메시지가 계속 추가(append)되어야 하므로

#### 1.2. Pydantic 스키마 (구조화된 출력)
```python
class IntentClassification(BaseModel):
    intent: Literal["destination_research", "itinerary_planning", "budget_estimation", "general_travel"]
```

**강의 포인트:**
- ✅ LLM이 JSON 대신 Pydantic 스키마로 출력하도록 강제
- ✅ `Literal`: 특정 값 중 하나만 허용 (Enum과 유사)
- ✅ `Field(description=...)`: LLM에게 각 필드의 의미 설명

**실습:**
```python
# 학생들이 직접 새로운 스키마 작성
class HotelRecommendation(BaseModel):
    name: str = Field(description="호텔 이름")
    price: int = Field(description="1박 가격 (원)", ge=0)
    rating: float = Field(description="평점", ge=0, le=5)
```

#### 1.3. `ExtractedPreferences` (다중 턴 지원)
```python
class ExtractedPreferences(BaseModel):
    destination: str | None = Field(default=None, description="여행지")
    budget: int | None = Field(default=None, description="예산")
```

**강의 포인트:**
- ✅ 모든 필드가 `Optional` (`| None`)
- ✅ 언급된 정보만 추출, 나머지는 `None`
- ✅ 이전 턴과 병합하여 누적 (nodes.py에서 구현)

### 🧪 실습 과제
1. 새로운 필드 `travel_style: str` 추가해보기
2. `ExtractedPreferences`에 `meal_preference: str | None` 추가

---

## 2️⃣ `prompts.py` - 프롬프트 엔지니어링 (50분)

### 🎓 학습 목표
- 역할(Role) 기반 프롬프트 작성
- Plan-and-Solve 기법의 프롬프트 구조
- 동적 프롬프트 포맷팅

### 📝 주요 개념

#### 2.1. 프롬프트 구조 패턴
```python
INTENT_CLASSIFICATION_PROMPT = """당신은 여행 문의 분류 전문가입니다.

역할:
사용자의 여행 관련 문의를 분석하여 적절한 카테고리로 분류합니다.

카테고리:
1. destination_research: 여행지 조사
2. itinerary_planning: 일정 계획
...

고객 문의:
{query}
"""
```

**강의 포인트:**
- ✅ **역할(Role) 명시**: "당신은 ~전문가입니다"
- ✅ **구체적 지시사항**: 카테고리 나열, 예시 제공
- ✅ **출력 형식 명시**: (Pydantic 스키마가 자동 포맷 제공)
- ✅ **동적 변수**: `{query}` → `.format(query=...)`로 삽입

#### 2.2. Plan-and-Solve 프롬프트 (핵심 기법)

**3단계 구조:**
1. **Plan**: 문제 분석 + 실행 계획 수립
2. **Solve**: 계획에 따라 도구 호출
3. **Synthesize**: 결과 종합

```python
PLAN_AND_SOLVE_PROMPT = """당신은 여행 계획 전문가입니다.

## Plan-and-Solve 기법 - Plan 단계

지시사항:
1. 사용자가 원하는 것을 정확히 파악하세요.
2. 답변에 필요한 정보가 무엇인지 파악하세요.
3. 어떤 조사 단계가 필요한지 구체적으로 나열하세요.
"""
```

**강의 포인트:**
- ✅ Plan-and-Solve는 **복잡한 문제를 작은 단계로 분해**
- ✅ 참고 논문: "Plan-and-Solve Prompting" (Wang et al., 2023)
- ✅ 각 프롬프트가 파이프라인의 특정 단계를 담당

#### 2.3. 동적 프롬프트 (예: RESEARCH_PROMPT)
```python
RESEARCH_PROMPT = """...
예산 추정 시 여행지는 다음 중 하나로 정확히 입력하세요: {budget_destinations}
"""
```

**강의 포인트:**
- ✅ `{budget_destinations}`는 런타임에 `BUDGET_DB.keys()` 값으로 대체
- ✅ 하드코딩 대신 데이터 참조 → 유지보수성 향상

### 🧪 실습 과제
1. 새로운 프롬프트 작성: "여행지 날씨 추천 전문가"
2. 프롬프트에 예시(Few-shot) 추가해보기

---

## 3️⃣ `tools.py` - 도구 정의 (60분)

### 🎓 학습 목표
- `@tool` 데코레이터를 사용한 도구 정의
- FAISS 벡터 스토어 기반 RAG 구현
- 외부 API (Serper) 연동
- 에러 핸들링 및 폴백 전략

### 📝 주요 개념

#### 3.1. 도구 정의 기본 구조
```python
class BudgetInput(BaseModel):
    destination: str = Field(description="여행지 이름")
    duration_days: int = Field(description="여행 기간", ge=1)

@tool(args_schema=BudgetInput)
def estimate_budget(destination: str, duration_days: int) -> str:
    """여행 예산을 추정합니다."""
    # ... 구현
```

**강의 포인트:**
- ✅ `@tool`: LangChain 도구로 등록
- ✅ `args_schema`: 입력 검증 (Pydantic)
- ✅ Docstring: LLM이 도구 사용법을 이해하는 데 사용
- ✅ 반환값은 항상 `str` (LLM이 읽을 수 있는 텍스트)

#### 3.2. RAG - FAISS 벡터 검색
```python
def _get_or_initialize_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    documents = _create_knowledge_base_documents()
    _vector_store = FAISS.from_documents(documents, embeddings)
```

**강의 포인트:**
- ✅ **Lazy Initialization**: 첫 사용 시에만 초기화 (API 비용 절감)
- ✅ **Embedding**: 텍스트 → 벡터로 변환 (Upstage Embeddings)
- ✅ **FAISS**: 벡터 유사도 검색 (Facebook AI Similarity Search)
- ✅ **Fallback**: FAISS 실패 시 키워드 검색으로 대체

#### 3.3. 폴백 검색 전략
```python
def _keyword_fallback_search(query: str) -> str:
    query_lower = query.lower()
    for item in TRAVEL_KNOWLEDGE_BASE:
        if any(word in (item["title"] + " " + item["content"]).lower()
               for word in query_lower.split() if len(word) >= 2):
            return f"[{item['category']}] {item['title']}\n{item['content']}"
```

**강의 포인트:**
- ✅ API 실패, 임베딩 오류 등 예외 상황 대비
- ✅ 완전 실패보다는 낮은 품질이라도 결과 제공
- ✅ 방어적 프로그래밍(Defensive Programming)

#### 3.4. 웹 검색 (Serper API)
```python
@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "gl": "kr", "hl": "ko", "num": 5}
    response = requests.post("https://google.serper.dev/search", ...)
```

**강의 포인트:**
- ✅ **Serper API**: Google 검색 결과를 JSON으로 제공
- ✅ `timeout=10`: 네트워크 오류 대비
- ✅ `try/except`: 모든 예외 상황 처리 → 항상 문자열 반환

### 🧪 실습 과제
1. 새로운 도구 추가: `get_weather(city: str) -> str`
2. Mock 데이터로 날씨 정보 반환 구현

---

## 4️⃣ `nodes.py` - 노드 함수 (90분)

### 🎓 학습 목표
- LangGraph 노드의 개념과 역할
- Plan-and-Solve 파이프라인 구현
- Evaluator-Optimizer 패턴
- 이중 메모리 시스템 (단기/장기)

### 📝 주요 개념

#### 4.1. 노드 함수 기본 구조
```python
def classify_intent_node(state: TravelPlanningState) -> dict:
    llm = ChatUpstage(model="solar-pro2", temperature=0.0)
    structured_llm = llm.with_structured_output(IntentClassification)
    
    query = _get_user_input(state)
    result = structured_llm.invoke([SystemMessage(...), HumanMessage(query)])
    
    return {
        "intent": result.intent,
        "messages": [HumanMessage(content=query)],
    }
```

**강의 포인트:**
- ✅ **노드**: 상태를 입력받아 부분 업데이트 딕셔너리 반환
- ✅ `state` 읽기 → 처리 → 반환값이 상태에 병합
- ✅ `.with_structured_output()`: Pydantic 스키마 강제
- ✅ `temperature=0.0`: 결정론적 출력 (분류 작업)

#### 4.2. 다중 턴 대화 - 선호도 추출
```python
def extract_preferences_node(state: TravelPlanningState) -> dict:
    messages = state.get("messages", [])
    result = structured_llm.invoke([SystemMessage(...), *messages])  # 전체 히스토리
    
    current_prefs = state.get("extracted_preferences", {})
    new_prefs = result.model_dump(exclude_none=True)
    merged_prefs = {**current_prefs, **new_prefs}  # 병합
    
    return {"extracted_preferences": merged_prefs}
```

**강의 포인트:**
- ✅ `*messages`: 전체 대화 히스토리를 LLM에 전달
- ✅ `exclude_none=True`: 언급되지 않은 정보는 제외
- ✅ 딕셔너리 병합: 이전 정보 + 새 정보 누적

#### 4.3. Plan-and-Solve 노드들

**1) plan_node (Plan 단계)**
```python
def plan_node(state: TravelPlanningState) -> dict:
    plan: TravelPlan = structured_llm.invoke(messages)
    return {
        "travel_plan": plan_text,
        "plan_steps": plan.steps,  # 다음 노드에서 사용
    }
```

**2) research_node (Solve 단계)**
```python
def research_node(state: TravelPlanningState) -> dict:
    plan_steps = state.get("plan_steps", [])
    llm_with_tools = llm.bind_tools(RESEARCH_TOOLS)
    response = llm_with_tools.invoke(messages)
    
    for tool_call in response.tool_calls:
        tool_output = tool_map[tool_name].invoke(tool_args)
```

**3) synthesize_node (Synthesize 단계)**
```python
def synthesize_node(state: TravelPlanningState) -> dict:
    prompt = SYNTHESIZE_PROMPT.format(
        travel_plan=state.get("travel_plan"),
        retrieved_context=state.get("retrieved_context"),
        budget_info=state.get("budget_info"),
        web_search_info=state.get("web_search_info"),
    )
    response = llm.invoke([SystemMessage(content=prompt), ...])
    return {"final_response": response.content}
```

**강의 포인트:**
- ✅ Plan → Solve → Synthesize: 3단계 파이프라인
- ✅ `plan_steps`가 research에서 도구 호출 결정에 사용됨
- ✅ 모든 조사 결과를 synthesize에서 통합

#### 4.4. Evaluator-Optimizer 패턴
```python
def evaluate_response_node(state) -> dict:
    result = structured_llm.invoke([...])
    return {
        "quality_score": result.score,
        "evaluation_passed": result.passed,  # 7점 이상?
    }

def should_improve_response(state) -> Literal["improve", "end"]:
    if state.get("evaluation_passed", False):
        return "end"
    if state.get("iteration") >= state.get("max_iterations"):
        return "end"
    return "improve"
```

**강의 포인트:**
- ✅ **Evaluator**: 응답 품질 평가 (1-10점)
- ✅ **Optimizer**: 피드백 기반 응답 개선
- ✅ 7점 미만 → 개선 → 재평가 (루프)
- ✅ 최대 반복 횟수 제한 (무한 루프 방지)

#### 4.5. 이중 메모리 시스템
```python
USER_PROFILES: dict[str, dict] = {}  # 장기 메모리

def save_memory_node(state, config) -> dict:
    user_id = config.get("configurable", {}).get("user_id")
    USER_PROFILES[user_id]["query_history"].append(...)
```

**강의 포인트:**
- ✅ **단기 메모리**: MemorySaver (graph.py에서 설명)
- ✅ **장기 메모리**: USER_PROFILES 딕셔너리
- ✅ `user_id`로 사용자 선호도 누적
- ✅ 프로세스 재시작 시 초기화 (프로덕션에서는 DB 사용)

### 🧪 실습 과제
1. 새로운 노드 추가: `recommend_hotels_node()`
2. `evaluation_passed` 기준을 8점으로 변경해보기

---

## 5️⃣ `graph.py` - 워크플로우 그래프 (60분)

### 🎓 학습 목표
- LangGraph 워크플로우 구성
- 조건부 라우팅 (Conditional Edges)
- MemorySaver를 사용한 단기 메모리

### 📝 주요 개념

#### 5.1. 그래프 구조
```python
builder = StateGraph(TravelPlanningState)

# 노드 추가
builder.add_node("classify_intent", classify_intent_node)
builder.add_node("plan", plan_node)
# ...

# 엣지 추가
builder.add_edge(START, "classify_intent")
builder.add_edge("plan", "research")

# 조건부 라우팅
builder.add_conditional_edges(
    "evaluate_response",
    should_improve_response,
    {"improve": "improve_response", "end": "save_memory"}
)
```

**강의 포인트:**
- ✅ `StateGraph`: 상태 기반 그래프 정의
- ✅ `add_node`: 노드 함수 등록
- ✅ `add_edge`: 고정 경로 연결
- ✅ `add_conditional_edges`: 조건부 분기 (if/else와 유사)

#### 5.2. 워크플로우 다이어그램
```
START
  ↓
classify_intent ─── 의도 분류
  ↓
extract_preferences ─── 선호도 추출
  ↓
plan ─── 계획 수립
  ↓
research ─── 도구 호출
  ↓
synthesize ─── 응답 생성
  ↓
evaluate_response ─── 품질 평가
  ↓
[개선 필요?] ──→ improve_response ──→ (재평가)
  ↓ (통과)
save_memory ─── 메모리 저장
  ↓
END
```

**강의 포인트:**
- ✅ 선형 파이프라인 + 평가-개선 루프
- ✅ `skip_to_end` 플래그로 빈 입력 건너뛰기
- ✅ 각 노드는 독립적 → 모듈화

#### 5.3. MemorySaver (단기 메모리)
```python
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 실행 시
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(initial_state, config)
```

**강의 포인트:**
- ✅ `thread_id`: 대화 세션 식별자
- ✅ 같은 `thread_id`로 재호출 → 이전 대화 맥락 자동 복원
- ✅ 체크포인터가 자동으로 상태 저장/복원
- ✅ `messages` 필드가 누적되어 대화 히스토리 유지

#### 5.4. 실행 헬퍼 함수
```python
def run_travel_planning(query: str, thread_id: str, user_id: str):
    graph = create_travel_planning_graph(with_memory=True)
    initial_state = create_initial_state(query)
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    return graph.invoke(initial_state, config)
```

**강의 포인트:**
- ✅ `thread_id`: 단기 메모리용 (대화 세션)
- ✅ `user_id`: 장기 메모리용 (사용자 프로필)
- ✅ 편의 함수로 복잡도 숨김

### 🧪 실습 과제
1. 새로운 조건부 엣지 추가: `quality_score >= 9`이면 다른 경로
2. 그래프 구조를 Mermaid로 시각화해보기

---

## 6️⃣ `main.py` - 실행 스크립트 (30분)

### 🎓 학습 목표
- 전체 에이전트 실행 흐름
- `graph.stream()`으로 노드별 진행 상황 확인
- 단기/장기 메모리 시연

### 📝 주요 개념

#### 6.1. 실행 흐름
```python
graph = create_travel_planning_graph(with_memory=True)
config = {"configurable": {"thread_id": "demo-thread-001", "user_id": "user-001"}}

for step in graph.stream(initial_state, config):
    node_name = list(step.keys())[0]
    print(f"[{node_count}] {node_name} 완료")

final_state = graph.get_state(config)
print(final_state.values.get("final_response"))
```

**강의 포인트:**
- ✅ `graph.stream()`: 각 노드 실행을 스트리밍
- ✅ `graph.invoke()`: 최종 상태만 반환 (스트림 불필요 시)
- ✅ `graph.get_state(config)`: 현재 저장된 상태 확인

#### 6.2. 다중 쿼리 실행 (단기 메모리 시연)
```python
# 쿼리 1
initial_state = create_initial_state("제주도 3박 4일 여행 계획 세워줘")
graph.stream(initial_state, config)

# 쿼리 2 (같은 thread_id)
followup_state = create_initial_state("거기 맛집도 추천해줘")
graph.stream(followup_state, config)  # 이전 대화 맥락 유지
```

**강의 포인트:**
- ✅ 같은 `thread_id` 사용 → MemorySaver가 대화 맥락 복원
- ✅ "거기"가 "제주도"를 의미함을 LLM이 이해
- ✅ 다중 턴 대화의 핵심

### 🧪 실습 과제
1. 3번째 쿼리 추가: "예산은 얼마나 들까?"
2. 다른 `thread_id`로 실행해서 맥락이 끊기는지 확인

---

## 🎯 전체 워크플로우 요약 (복습용)

```
┌─────────────────────────────────────────────────────────┐
│ 1. state.py    → 상태 정의 (TypedDict, Pydantic)       │
│ 2. prompts.py  → 프롬프트 작성 (Plan-and-Solve)        │
│ 3. tools.py    → 도구 정의 (RAG, 예산, 웹 검색)        │
│ 4. nodes.py    → 노드 함수 (핵심 로직)                 │
│ 5. graph.py    → 그래프 구성 (워크플로우)              │
│ 6. main.py     → 실행 및 시연                          │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 적용된 기술 체크리스트

| 기술 | 모듈 | 핵심 개념 |
|------|------|----------|
| **Plan-and-Solve** | `prompts.py`, `nodes.py` | 계획 → 실행 → 종합 |
| **Agentic RAG** | `tools.py` | FAISS 벡터 검색 + 폴백 |
| **Tool Calling** | `tools.py`, `nodes.py` | LLM이 도구 호출 결정 |
| **Structured Output** | `state.py`, `nodes.py` | Pydantic 스키마 강제 |
| **Evaluator-Optimizer** | `nodes.py` | 품질 평가 + 개선 루프 |
| **이중 메모리** | `nodes.py`, `graph.py` | 단기(MemorySaver) + 장기(USER_PROFILES) |
| **다중 턴 대화** | `state.py`, `nodes.py` | `extracted_preferences` 누적 |

---

## 🚀 강의 팁

### ✅ 실습 중심 접근
- 각 모듈을 설명한 후 즉시 코드 수정 실습
- 작은 변경 → 실행 → 결과 확인 사이클 반복

### ✅ 디버깅 가이드
```python
# 각 노드의 출력 확인
for step in graph.stream(initial_state, config):
    print(f"Step: {step}")  # 노드별 출력 확인
```

### ✅ 흔한 오류 및 해결
1. **UPSTAGE_API_KEY 오류**
   - → `.env` 파일 확인
2. **FAISS 초기화 실패**
   - → 키워드 폴백 검색으로 자동 대체 (문제 없음)
3. **도구 호출 실패**
   - → `try/except`로 처리됨, 오류 메시지 확인

### ✅ 확장 아이디어 (학생 프로젝트)
1. 새로운 여행지 추가 (`TRAVEL_KNOWLEDGE_BASE`)
2. 새로운 도구 추가 (날씨, 환율 등)
3. 평가 기준 변경 (8점 → 9점)
4. 새로운 의도 추가 (`"flight_booking"`)

---

## 📚 참고 자료

### 논문
- **Plan-and-Solve Prompting**: Wang et al., 2023
- **ReAct**: Yao et al., 2022 (Tool Calling 기반)

### 공식 문서
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangChain 도구 가이드](https://python.langchain.com/docs/modules/agents/tools/)
- [Upstage API 문서](https://console.upstage.ai/docs)

---

## ❓ FAQ (자주 묻는 질문)

### Q1. MemorySaver와 USER_PROFILES의 차이는?
- **MemorySaver**: 대화 세션(thread_id) 기반, 메시지 히스토리 자동 관리
- **USER_PROFILES**: 사용자(user_id) 기반, 선호도 수동 관리

### Q2. 왜 Plan-and-Solve를 사용하나요?
- 복잡한 문제를 작은 단계로 분해 → LLM이 더 정확한 답변 생성
- "한 번에 모든 것"보다 "계획 → 실행 → 종합"이 품질 향상

### Q3. temperature는 언제 0, 언제 0.3/0.5?
- **0.0**: 분류, 평가 등 결정론적 작업
- **0.3-0.5**: 창의적 답변 생성 (응답 작성)

### Q4. 왜 모든 도구가 `str`을 반환하나요?
- LLM은 텍스트만 이해 → 숫자/객체를 문자열로 변환

### Q5. `operator.add`는 언제 사용하나요?
- 값을 누적해야 하는 필드 (`messages`, `tool_results`, `error_log`)

---

## 🎓 최종 점검 체크리스트 (강의 후)

- [ ] 학생들이 `state.py`에서 TypedDict와 Pydantic 차이를 이해했는가?
- [ ] Plan-and-Solve의 3단계(Plan-Solve-Synthesize)를 설명할 수 있는가?
- [ ] RAG와 일반 검색의 차이를 이해했는가?
- [ ] 노드 함수가 상태를 어떻게 업데이트하는지 이해했는가?
- [ ] 단기 메모리와 장기 메모리의 차이를 설명할 수 있는가?
- [ ] `main.py`를 직접 실행하여 결과를 확인했는가?

---

**🎉 이 가이드를 활용하여 효과적인 에이전트 교육을 진행하세요!**
