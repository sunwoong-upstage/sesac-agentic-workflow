# AI Agent Design Thinking Guide

> **Purpose:** A template for students to understand how to plan and design an AI agent before development.
> **Case Study:** News Chatbot Agent

---

## Table of Contents

1. [Phase 1: Problem Definition](#phase-1-problem-definition)
2. [Phase 2: User Journey Mapping](#phase-2-user-journey-mapping)
3. [Phase 3: Capability Analysis](#phase-3-capability-analysis)
4. [Phase 4: Tool Design](#phase-4-tool-design)
5. [Phase 5: Workflow Architecture](#phase-5-workflow-architecture)
6. [Phase 6: State Design](#phase-6-state-design)
7. [Template for Your Own Agent](#template-for-your-own-agent)

---

## Phase 1: Problem Definition

### The Question to Ask

> **"What problem does this agent solve that a simple chatbot cannot?"**

### News Chatbot Case Study

| Question | Answer |
|----------|--------|
| **Who is the user?** | Busy professionals who need to stay informed but lack time |
| **What's their pain point?** | Information overload: too many news sources, unclear which is important |
| **Why an agent, not a chatbot?** | Needs to execute actions (search archive, calculate dates, fetch real-time news) not just respond |
| **What makes it "agentic"?** | It decides WHICH tools to use based on the query, not a fixed script |

### Design Decision Record

```markdown
## ADR-0: Why Build an Agent?

**Context:** Users ask news questions that require:
- Searching through news archives
- Calculating date ranges ("last week", "past month")
- Finding real-time information

**Decision:** Build an AI agent with tool-calling capability

**Rationale:**
- Simple Q&A would hallucinate news facts
- RAG alone can't calculate "3 days ago"
- Users need synthesized summaries, not raw search results

**Consequences:**
- More complex than a chatbot
- Requires tool definitions
- LLM must support function calling
```

---

## Phase 2: User Journey Mapping

### The Question to Ask

> **"What does a successful interaction look like from start to finish?"**

### News Chatbot Case Study

```
User Journey Map:

1. USER ENTERS QUERY
   "최근 일주일간 엔비디아 관련 뉴스 알려줘"

   → Agent thinks: "This is news_search intent"

2. AGENT PLANS APPROACH
   → "I need to: calculate date range (7 days), search archive, maybe web search"

3. AGENT EXECUTES TOOLS
   → Calls date calculator with (time_value=7, time_unit="days")
   → Calls RAG search for "엔비디아"
   → Optionally calls Serper API for latest news

4. AGENT SYNTHESIZES RESPONSE
   → Combines all tool outputs into coherent summary
   → Lists 3 key news items with dates and sources

5. QUALITY CHECK
   → "Is this response complete? Does it answer all aspects?"

6. USER RECEIVES ANSWER
   → Personalized news summary with sources and dates
```

### Key Insight: Identify Required Capabilities

From this journey, we identify:

| User Need | Required Capability | Implementation |
|-----------|-------------------|----------------|
| "What's the latest on X?" | Knowledge retrieval | RAG tool (FAISS) |
| "News from last week" | Date calculation | Date range tool |
| "Breaking news today" | Real-time info | Web search tool (Serper) |
| "Summarize for me" | Synthesis | LLM with context |
| "Remember my interests" | Personalization | Memory system |

---

## Phase 3: Capability Analysis

### The Question to Ask

> **"What capabilities does the LLM need vs. what requires external tools?"**

### Capability Matrix

| Capability | LLM Alone? | Tool Required? | Why? |
|------------|------------|----------------|------|
| Understanding query intent | ✅ Yes | No | LLMs excel at classification |
| Generating news summaries | ✅ Yes | No | Creative text generation |
| Knowing specific news facts | ⚠️ Partial | ✅ RAG | LLM may hallucinate or be outdated |
| Calculating "3 days ago" | ❌ No | ✅ Calculator | LLMs make date calculation errors |
| Current breaking news | ❌ No | ✅ API | Real-time data required |
| Remembering user preferences | ❌ No | ✅ Memory | LLM context is ephemeral |

### Design Decision Record

```markdown
## ADR-1: Tool vs. LLM Responsibility

**Rule of Thumb:**
- LLM: Understanding, reasoning, generating text
- Tools: Facts, calculations, real-time data, persistence

**Applied to News Chatbot:**
| Task | Owner |
|------|-------|
| Parse "지난 일주일" → 7 days | LLM (structured output) |
| Calculate today - 7 days | Tool (date calculator) |
| Write news summary prose | LLM |
| Store user's preferred topics | Tool (memory) |
```

---

## Phase 4: Tool Design

### The Question to Ask

> **"For each tool, what is the minimal input/output contract?"**

### Tool Design Template

```markdown
## Tool: [Name]

### Purpose
One sentence explaining why this tool exists.

### Input Schema
```python
class ToolInput(BaseModel):
    param1: type = Field(description="...")
    param2: type = Field(description="...")
```

### Output Format
What the tool returns (string, structured data, etc.)

### Edge Cases
- What if input is invalid?
- What if external service is down?
- What's the fallback behavior?

### Why This Design?
Explain the reasoning behind input/output choices.
```

### News Chatbot Tools: Design Rationale

#### Tool 1: `search_news_archive`

```markdown
## Tool: search_news_archive

### Purpose
Retrieve news articles from internal knowledge base.

### Why This Tool?
- LLMs hallucinate news facts and dates
- We want consistent, curated news information
- RAG provides grounded, retrievable facts

### Input Design Decision
```python
class NewsSearchInput(BaseModel):
    query: str = Field(description="검색 쿼리 (예: '엔비디아 뉴스')")
```

**Why just `query`?**
- Simple interface for LLM to call
- Let RAG handle relevance matching
- Considered adding `date_range` filter but decided:
  - LLM already includes temporal info in natural query
  - Adding filters increases complexity without benefit

### Output Design Decision
Returns formatted string with news articles.

**Why string?**
- LLM will synthesize this with other info
- Structured output would require additional parsing
- For synthesis, prose format is more natural

### Fallback Strategy
1. Try FAISS vector search
2. If FAISS fails → keyword matching
3. If no matches → "관련 뉴스를 찾지 못했습니다"

**Why keyword fallback?**
- FAISS requires embeddings API (may fail)
- Students can test without API keys
- Demonstrates defensive programming
```

#### Tool 2: `calculate_date_range`

```markdown
## Tool: calculate_date_range

### Purpose
Convert relative time expressions to actual date ranges.

### Why This Tool?
- Date arithmetic requires precision
- LLMs make calculation errors ("3 days ago" = wrong date)
- Users expect exact dates, not approximations

### Input Design Decision
```python
class DateRangeInput(BaseModel):
    time_value: int = Field(description="시간 값 (예: 7)", ge=1)
    time_unit: str = Field(description="시간 단위: days/weeks/months")
```

**Why these parameters?**
- `time_value`: Required for delta calculation
- `time_unit`: Enables flexible time ranges (days/weeks/months)
- Considered adding `from_date` but removed for simplicity

**Why `time_value` as int?**
- LLM can extract "지난 일주일" → (7, "days") via structured output
- Integer enables direct datetime calculation
- Avoids parsing "일주일" string inside the tool

### Output Design Decision
Returns start_date and end_date as formatted strings.

**Why formatted strings?**
- Human-readable for display
- LLM can naturally incorporate into response
- Enables transparent date filtering explanation

### Why Not Use LLM for This?
- Teaching focus is on tool-calling pattern
- Demonstrates when NOT to use LLM
- Date math is deterministic, not probabilistic
```

#### Tool 3: `search_recent_news`

```markdown
## Tool: search_recent_news

### Purpose
Search the web for real-time news information.

### Why This Tool?
- Knowledge base is static (doesn't know today's news)
- Users ask "What's happening now?" or "Latest developments"
- Grounds responses in current reality

### Input Design Decision
```python
class WebSearchInput(BaseModel):
    query: str = Field(description="검색 쿼리")
```

**Why just `query`?**
- Serper API handles all search complexity
- LLM naturally formulates good search queries
- No need to specify search engine, language, etc.

### Why Serper Instead of SerpAPI?
- Simpler API (just POST with query)
- Cheaper for educational use
- Returns structured news results

### Graceful Degradation
```python
if not api_key:
    return "웹 검색을 사용할 수 없습니다 (SERPER_API_KEY 미설정)"
```

**Why not raise exception?**
- Agent should continue with other tools
- User still gets partial answer (from archive)
- Better UX than complete failure
```

---

## Phase 5: Workflow Architecture

### The Question to Ask

> **"In what order should operations happen, and why?"**

### Workflow Design Options

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Linear** | A → B → C → D | Simple, debuggable, sequential dependencies |
| **Branching** | A → (B or C) → D | Different paths based on intent |
| **ReAct Loop** | Think → Act → Observe → Repeat | Complex reasoning, unknown steps |
| **Plan-and-Solve** | Plan → Execute → Synthesize | Explicit planning before action |

### News Chatbot: Why Plan-and-Solve?

```markdown
## ADR-2: Choosing Plan-and-Solve Over ReAct

**Context:**
We need to decide how the agent approaches queries.

**Options Considered:**

1. **ReAct (Reasoning + Acting)**
   - Pros: Flexible, handles unexpected cases
   - Cons: Less predictable, harder to debug

2. **Plan-and-Solve**
   - Pros: Explicit steps, easier to trace
   - Cons: Less flexible to mid-execution changes

**Decision:** Plan-and-Solve

**Rationale:**
1. News search is inherently "plannable" - we know the steps
2. Educational clarity > flexibility for learning
3. Explicit `plan_steps` state makes debugging transparent
4. Students can SEE what the agent decided to do

**Trade-off Acknowledged:**
If user asks follow-up that invalidates the plan, we re-plan from scratch
rather than dynamically adjusting. This is acceptable for education.
```

### The Pipeline Design

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   START                                                         │
│     │                                                           │
│     ▼                                                           │
│   [classify_intent] ─── "What type of question is this?"        │
│     │                                                           │
│     ▼                                                           │
│   [extract_preferences] ─── "Extract topics/keywords/dates"     │
│     │                                                           │
│     ▼                                                           │
│   [plan] ─── "What steps do I need to answer this?"             │
│     │         Output: ["search for X", "calculate Y", ...]      │
│     ▼                                                           │
│   [research] ─── "Execute the tools from my plan"               │
│     │             Uses plan_steps to decide which tools         │
│     ▼                                                           │
│   [synthesize] ─── "Combine all tool results into answer"       │
│     │                                                           │
│     ▼                                                           │
│   [evaluate] ─── "Is this answer good enough?"                  │
│     │                                                           │
│     ├──(score < 7)──► [optimize] ──► [evaluate] (loop)          │
│     │                                                           │
│     ▼ (score >= 7)                                              │
│   [save_memory] ─── "Remember user preferences"                 │
│     │                                                           │
│     ▼                                                           │
│   END                                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why Each Node Exists

| Node | Purpose | Why Separate? |
|------|---------|---------------|
| `classify_intent` | Categorize query type | Different intents may need different prompts |
| `extract_preferences` | Extract topics/keywords/dates | Enables personalized memory storage |
| `plan` | Generate execution plan | Makes tool selection explicit and traceable |
| `research` | Execute tools | Separates planning from doing |
| `synthesize` | Combine results | Tool outputs need narrative synthesis |
| `evaluate` | Quality check | Ensures response meets quality bar |
| `optimize` | Fix issues | Enables self-correction without human intervention |
| `save_memory` | Persist preferences | Enables personalization across sessions |

---

## Phase 6: State Design

### The Question to Ask

> **"What information needs to flow between nodes?"**

### State Design Principles

1. **Minimal but Complete:** Only store what's needed
2. **Immutable Flow:** Nodes read state, return updates
3. **Traceable:** Include enough for debugging
4. **Typed:** Use Pydantic/TypedDict for safety

### News Chatbot State: Field-by-Field Rationale

```python
class NewsChatbotState(TypedDict):
    # ─── Input/Output ───
    user_input: str           # Original query (immutable after set)
    final_response: str       # Final answer to user

    # ─── Intent & Preferences ───
    intent: str               # Classified query type
    preferences: dict         # Extracted topics/keywords/dates

    # ─── Plan-and-Solve ───
    news_plan: str            # Human-readable plan description
    plan_steps: List[str]     # Machine-readable steps for research node

    # ─── Tool Results ───
    tool_results: List[dict]  # Raw outputs from all tools
    archive_news: str         # Extracted RAG result
    date_range_info: str      # Extracted date calculation result
    web_search_news: str      # Extracted web search result

    # ─── Quality Loop ───
    quality_score: int        # 1-10 rating
    quality_feedback: str     # What needs improvement
    evaluation_passed: bool   # Score >= 7?
    iteration: int            # Current loop iteration
    max_iterations: int       # Prevent infinite loops

    # ─── Memory ───
    user_profile: dict        # Long-term preferences

    # ─── Control Flow ───
    skip_to_end: bool         # Short-circuit for empty input

    # ─── Debugging ───
    messages: List[BaseMessage]  # Conversation history
    error_log: List[str]         # Accumulated errors
```

### Why Separate `tool_results` AND Individual Fields?

```markdown
## Design Decision: Tool Result Storage

**Option A:** Store only `tool_results: List[dict]`
- Pro: DRY (Don't Repeat Yourself)
- Con: Every node must filter for specific tool

**Option B:** Store both aggregate AND extracted fields
- Pro: `synthesize_node` directly accesses `archive_news`
- Con: Slight redundancy

**Decision:** Option B

**Rationale:**
1. `synthesize_node` prompt template expects named fields
2. Extraction happens once in `research_node`, not repeatedly
3. Clearer debugging: can inspect individual tool outputs
4. Educational clarity: students see explicit data flow
```

---

## Template for Your Own Agent

Use this template when designing your own AI agent:

### Step 1: Problem Definition

```markdown
## My Agent: [Name]

### Problem Statement
[One paragraph describing the problem]

### Why an Agent?
- [ ] Requires external data retrieval
- [ ] Requires calculations/logic the LLM can't do
- [ ] Requires real-time information
- [ ] Requires persistence/memory
- [ ] Requires multi-step reasoning

### Target User
[Who will use this agent?]

### Success Criteria
[How do we know the agent is working well?]
```

### Step 2: Capability Mapping

```markdown
## Capabilities

| User Need | LLM Alone? | Tool Needed? | Tool Name |
|-----------|------------|--------------|-----------|
| [Need 1]  | Yes/No     | Yes/No       | [Name]    |
| [Need 2]  | Yes/No     | Yes/No       | [Name]    |
| ...       | ...        | ...          | ...       |
```

### Step 3: Tool Specifications

```markdown
## Tool: [Name]

### Purpose
[Why this tool exists]

### Input
```python
class [Name]Input(BaseModel):
    field: type = Field(description="...")
```

### Output
[Description of output format]

### Fallback
[What happens if the tool fails?]
```

### Step 4: Workflow Design

```markdown
## Workflow

### Pattern Choice
[ ] Linear Pipeline
[ ] Branching
[ ] ReAct Loop
[ ] Plan-and-Solve
[ ] Custom: ___________

### Rationale
[Why this pattern?]

### Node Diagram
[ASCII or Mermaid diagram]
```

### Step 5: State Design

```markdown
## State Fields

| Field | Type | Set By | Used By | Purpose |
|-------|------|--------|---------|---------|
| ...   | ...  | ...    | ...     | ...     |
```

---

## Summary: Key Design Principles

1. **Start with the user journey**, not the technology
2. **Tools are for facts and actions**, LLM is for understanding and synthesis
3. **Make implicit reasoning explicit** through state and logging
4. **Design for failure** with fallbacks at every level
5. **Prioritize educational clarity** over production optimization
6. **Document decisions** with ADRs (Architecture Decision Records)

---

## References

- [Plan-and-Solve Prompting (Wang et al., 2023)](https://arxiv.org/abs/2305.04091)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/)
- [Building Agents with Tool Calling](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [AI Service Design Document](./AI_SERVICE_DESIGN.md) - News Chatbot specific examples
