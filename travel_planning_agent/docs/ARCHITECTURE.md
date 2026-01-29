# Travel Planning Agent - Architecture

## Project Structure

```
travel_planning_agent/
├── agent/
│   ├── __init__.py      # Exports
│   ├── graph.py         # Workflow graph
│   ├── nodes.py         # Node functions
│   ├── prompts.py       # Prompt templates
│   ├── state.py         # State definition
│   └── tools.py         # 3 tools (RAG, budget, web_search)
├── tests/
│   ├── __init__.py
│   └── test_edge_cases.py
├── main.py              # Demo runner
├── langgraph.json       # LangGraph Studio config
├── pyproject.toml
└── README.md
```

---

## Workflow Diagram

```
START
  ↓
[classify_intent] ──(skip_to_end?)──→ [save_memory] → END
  ↓ (continue)
[extract_preferences]
  ↓
[plan]
  ↓
[research]
  ↓
[synthesize]
  ↓
[evaluate_response] ←──┐
  ↓                    │
(improve?) ────────────┤
  ↓ (end)              │ (improve)
[improve_response] ────┘
  ↓
[save_memory]
  ↓
END
```

---

## Nodes (8 total)

### 1. classify_intent
- **Function**: `classify_intent_node()`
- **File**: `nodes.py:48`
- **Purpose**: Classifies user query into 4 categories
  - `destination_research`: 여행지 조사/추천
  - `itinerary_planning`: 일정 계획
  - `budget_estimation`: 예산 추정
  - `general_travel`: 일반 여행 문의
- **Also handles**: Empty input early exit (sets `skip_to_end=True`)
- **LLM calls**: 1 (structured output with `IntentClassification` schema)
- **Needed**: YES - Entry point, handles empty input

### 2. extract_preferences
- **Function**: `extract_preferences_node()`
- **File**: `nodes.py:91`
- **Purpose**: Extracts travel preferences from conversation history
- **Output**: merged_prefs dict with destination, duration, budget, etc.
- **LLM calls**: 1 (structured output with `ExtractedPreferences` schema)
- **Needed**: YES - Accumulates user preferences across turns

### 3. plan
- **Function**: `plan_node()`
- **File**: `nodes.py:146`
- **Purpose**: Plan-and-Solve "Plan" phase
- **Output**: Generates execution steps (e.g., `["여행지 정보 검색", "예산 추정", "웹 검색"]`)
- **LLM calls**: 1 (structured output with `TravelPlan` schema)
- **Needed**: MAYBE - Could merge with research

### 4. research
- **Function**: `research_node()`
- **File**: `nodes.py:190`
- **Purpose**: Plan-and-Solve "Solve" phase
- **Behavior**:
  1. Binds tools to LLM and sends plan_steps in prompt
  2. If LLM returns `tool_calls` → execute them
  3. If no `tool_calls` → fallback to keyword-based tool execution
- **Tools called**:
  - `search_travel_knowledge` (RAG with FAISS)
  - `estimate_budget` (internal calculation)
  - `web_search` (Serper API)
- **LLM calls**: 1
- **Needed**: YES - Core tool execution

### 5. synthesize
- **Function**: `synthesize_node()`
- **File**: `nodes.py:257`
- **Purpose**: Plan-and-Solve "Synthesize" phase
- **Input**: Combines travel_plan, retrieved_context, budget_info, web_search_info
- **Output**: Final response to user
- **LLM calls**: 1
- **Needed**: YES - Generates final answer

### 6. evaluate_response
- **Function**: `evaluate_response_node()`
- **File**: `nodes.py:283`
- **Purpose**: Evaluator-Optimizer pattern - Evaluator role
- **Behavior**: Scores response 1-10, passes if ≥7
- **Output**: `quality_score`, `quality_feedback`, `evaluation_passed`
- **LLM calls**: 1 (structured output with `QualityEvaluation` schema)
- **Needed**: MAYBE - Adds quality but extra LLM call

### 7. improve_response
- **Function**: `improve_response_node()`
- **File**: `nodes.py:320`
- **Purpose**: Evaluator-Optimizer pattern - Optimizer role
- **Behavior**: Rewrites response based on evaluation feedback
- **LLM calls**: 1
- **Needed**: MAYBE - Only runs if score <7

### 8. save_memory
- **Function**: `save_memory_node()`
- **File**: `nodes.py:345`
- **Purpose**: Dual memory system
  - Short-term: MemorySaver (automatic, thread_id based)
  - Long-term: InMemoryStore (user_id based, persists across invocations)
- **Saves**: preferred_destinations, query_history
- **LLM calls**: 0
- **Needed**: YES - Demonstrates memory system with LangGraph Store API

---

## Edges (11 total)

| From | To | Type | Condition | Purpose |
|------|----|------|-----------|---------|
| START | classify_intent | Direct | - | Entry point |
| classify_intent | extract_preferences | Conditional | `skip_to_end == False` | Normal flow |
| classify_intent | save_memory | Conditional | `skip_to_end == True` | Empty input bypass |
| extract_preferences | plan | Direct | - | Preferences → Plan |
| plan | research | Direct | - | Plan → Solve |
| research | synthesize | Direct | - | Solve → Synthesize |
| synthesize | evaluate_response | Direct | - | Response → Quality check |
| evaluate_response | improve_response | Conditional | `score <7 && iteration < max` | Needs improvement |
| evaluate_response | save_memory | Conditional | `passed \|\| iteration >= max` | Quality OK or max reached |
| improve_response | evaluate_response | Direct | - | Re-evaluate after improvement (loop) |
| save_memory | END | Direct | - | Exit point |

---

## Routing Functions

### should_skip_pipeline()
- **File**: `graph.py:54`
- **Input**: `state.skip_to_end`
- **Returns**: `"continue"` or `"skip"`
- **Purpose**: Bypass full pipeline for empty input

### should_improve_response()
- **File**: `nodes.py:383`
- **Input**: `state.evaluation_passed`, `state.iteration`, `state.max_iterations`
- **Returns**: `"improve"` or `"end"`
- **Purpose**: Control evaluation-improvement loop

---

## Agentic Patterns Demonstrated

| Pattern | Component(s) | Educational Value |
|---------|--------------|-------------------|
| **Structured Output** | classify_intent, plan, evaluate | LLM returns typed JSON via Pydantic schema |
| **Plan-and-Solve** | plan → research → synthesize | Decompose complex task into phases |
| **Tool Calling + Fallback** | research | LLM decides tools; fallback if unsupported |
| **Evaluator-Optimizer** | evaluate ↔ improve loop | Self-improvement with quality gate |
| **Dual Memory** | save_memory + MemorySaver + InMemoryStore | Short-term (thread) + Long-term (user via Store API) |
| **Conditional Routing** | skip_to_end, should_improve | Dynamic graph flow based on state |

---

## Tools (3 total)

| Tool | Type | Purpose |
|------|------|---------|
| `search_travel_knowledge` | RAG (FAISS) | Search travel knowledge base |
| `estimate_budget` | Internal function | Calculate trip cost breakdown |
| `web_search` | External API (Serper) | Real-time web search |

---

## LLM Usage Per Request

| Scenario | LLM Calls | Nodes Used |
|----------|-----------|------------|
| Empty input | 1 | classify_intent → save_memory |
| Normal (pass on first eval) | 6 | classify → extract_prefs → plan → research → synthesize → evaluate → save |
| Normal (1 improvement) | 8 | ... + improve → evaluate |
| Normal (max 3 improvements) | 12 | ... + (improve → evaluate) × 3 |

---

## Simplification Options

### Option A: Remove Evaluation Loop
- **Remove**: `evaluate_response`, `improve_response`
- **Change**: `synthesize` → `save_memory` → `END`
- **Saves**: 2-6 LLM calls per request
- **Loses**: Self-improvement quality loop

### Option B: Merge Plan + Research
- **Merge**: `plan_node` into `research_node`
- **Rationale**: plan_steps only used internally by research
- **Saves**: 1 LLM call

### Option C: Keep as-is (Recommended for Education)
- Demonstrates all major agentic patterns
- Each node has clear, distinct responsibility
- Good for teaching LangGraph concepts

---

## State Fields

```python
class TravelPlanningState(TypedDict):
    # Conversation
    messages: Annotated[List[BaseMessage], operator.add]

    # Input/Output
    user_input: str
    final_response: str

    # Plan-and-Solve
    travel_plan: str
    plan_steps: List[str]

    # Intent
    intent: str  # destination_research | itinerary_planning | budget_estimation | general_travel

    # Tool Results
    tool_results: Annotated[List[dict], operator.add]
    retrieved_context: str  # RAG search result
    budget_info: str
    web_search_info: str

    # Quality Evaluation
    quality_score: int
    quality_feedback: str
    evaluation_passed: bool

    # Loop Control
    iteration: int
    max_iterations: int

    # Memory
    user_profile: dict
    extracted_preferences: dict  # Preferences extracted from conversation

    # Pipeline Control
    skip_to_end: bool

    # Errors
    error_log: Annotated[List[str], operator.add]
```

## Memory Architecture

### Short-term Memory (Checkpointer)
- **Component**: `MemorySaver` from `langgraph.checkpoint.memory`
- **Scope**: Per-thread conversation state
- **Persistence**: In-memory, lost on process restart
- **Access**: Automatic via `thread_id` in config

### Long-term Memory (Store)
- **Component**: `InMemoryStore` from `langgraph.store.memory`
- **Scope**: Cross-thread user profiles
- **Persistence**: In-memory, survives across `run_travel_planning()` calls within same process
- **Access**:
  - Load: `user_store.get(("users",), user_id)` before graph invocation
  - Save: `user_store.put(("users",), user_id, profile)` after graph invocation
- **Upgrade path**: Replace with `PostgresStore` for true persistence

### TravelContext (Runtime Context)
- **Location**: `state.py`
- **Purpose**: Dataclass for passing runtime context (prepared for future use)
- **Fields**: `user_id: str = "anonymous"`

## Pydantic Schemas (Simplified)

```python
class IntentClassification(BaseModel):
    intent: Literal["destination_research", "itinerary_planning", "budget_estimation", "general_travel"]

class TravelPlan(BaseModel):
    destination: str
    duration_days: int
    steps: List[str]

class QualityEvaluation(BaseModel):
    score: int  # 1-10
    feedback: str
    passed: bool  # score >= 7
```
