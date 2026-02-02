# 뉴스 챗봇 에이전트 아키텍처

## 개요

뉴스 데이터 기반 자연어 챗봇 시스템으로, Plan-and-Solve 패턴과 LangGraph를 활용합니다.

## 워크플로우

```
User Query
    │
    ▼
┌─────────────────┐
│ classify_intent │ → 의도 분류 (news_search/trending/summary/general)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ extract_preferences │ → 주제/키워드/날짜 범위 추출
└────────┬────────────┘
         │
         ▼
┌────────┐
│  plan  │ → 실행 계획 수립 (Plan-and-Solve: Plan)
└────┬───┘
     │
     ▼
┌──────────┐
│ research │ → 도구 호출하여 정보 수집 (Plan-and-Solve: Solve)
└────┬─────┘
     │
     ▼
┌────────────┐
│ synthesize │ → 결과 종합하여 응답 생성 (Plan-and-Solve: Synthesize)
└────┬───────┘
     │
     ▼
┌──────────┐
│ evaluate │ → 품질 평가 (1-10점, 7점 이상 통과)
└────┬─────┘
     │
     ├── pass ───────────────────────┐
     │                               │
     ▼                               ▼
┌─────────┐                    ┌─────────────┐
│ improve │ ← fail ←───────────│ save_memory │ → END
└────┬────┘                    └─────────────┘
     │
     └─► evaluate (재평가)
```

## 3가지 도구

### 1. search_news_archive (RAG)
- **용도**: 저장된 뉴스 아카이브에서 관련 기사 검색
- **기술**: FAISS 벡터 스토어 + Upstage Embeddings
- **데이터**: Mock 뉴스 데이터베이스 (10개 기사)

### 2. calculate_date_range (Python Function)
- **용도**: 상대적 날짜 표현을 실제 날짜로 변환
- **입력**: time_value (숫자), time_unit (days/weeks/months)
- **출력**: 시작일, 종료일 (YYYY-MM-DD)

### 3. search_recent_news (External API)
- **용도**: SERPER API를 통한 실시간 뉴스 검색
- **기술**: Google 뉴스 검색 API
- **설정**: 한국어 (hl=ko), 한국 (gl=kr)

## 듀얼 메모리 시스템

### 단기 메모리 (MemorySaver)
- **범위**: thread_id 기반
- **용도**: 같은 대화 내 컨텍스트 유지
- **저장**: 메시지 히스토리, 추출된 선호도

### 장기 메모리 (InMemoryStore)
- **범위**: user_id 기반
- **용도**: 사용자 프로필 저장
- **저장**: 관심 주제, 검색 이력

## 상태 관리

```python
class NewsChatbotState(TypedDict):
    # 입력
    user_input: str
    messages: Annotated[List[BaseMessage], operator.add]

    # 분류 결과
    intent: str
    intent_confidence: float

    # 선호도
    topics: List[str]
    keywords: List[str]
    date_range: Optional[dict]

    # 실행
    execution_plan: str
    tool_results: Annotated[List[dict], operator.add]

    # 응답
    final_response: str

    # 평가
    quality_score: int
    evaluation_passed: bool
    iteration: int
    max_iterations: int

    # 메모리
    user_profile: dict
    error_log: Annotated[List[str], operator.add]
```

## 프로젝트 구조

```
news_chatbot_agent/
├── agent/
│   ├── state.py      # 상태 정의
│   ├── prompts.py    # 프롬프트 템플릿
│   ├── tools.py      # 3가지 도구
│   ├── nodes.py      # 워크플로우 노드
│   └── graph.py      # LangGraph 조립
├── evaluate/         # LangSmith 평가
├── tests/            # 테스트
├── docs/             # 문서
└── main.py           # 실행 스크립트
```
