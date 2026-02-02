# 뉴스 챗봇 에이전트

뉴스 데이터 기반 자연어 챗봇 시스템 - **교육용 LangGraph 에이전트**

> 이 프로젝트는 LangGraph를 활용한 에이전트 워크플로우 학습을 위한 교육용 자료입니다.
> 3가지 도구 유형(RAG, Python 함수, 외부 API)과 Evaluator-Optimizer 패턴을 실습할 수 있습니다.

## 목차

- [주요 기능](#주요-기능)
- [아키텍처](#아키텍처)
- [설치 및 실행](#설치-및-실행)
- [프로젝트 구조](#프로젝트-구조)
- [도구 상세](#도구-상세)
- [워크플로우 설명](#워크플로우-설명)
- [학습 자료](#학습-자료)
- [테스트](#테스트)
- [라이선스](#라이선스)

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **RAG 검색** | FAISS 벡터 스토어 + Upstage Embeddings를 활용한 뉴스 아카이브 검색 |
| **날짜 계산** | "7일 전", "이번 주" 등 상대적 날짜를 실제 날짜로 변환 |
| **실시간 뉴스** | SERPER API를 통한 최신 뉴스 검색 |
| **의도 분류** | 사용자 메시지를 4가지 의도로 분류 (news_search, trending, summary, general) |
| **품질 평가** | LLM 기반 응답 품질 자동 평가 및 개선 (Evaluator-Optimizer 패턴) |
| **멀티턴 대화** | 대화 히스토리와 사용자 선호도를 누적 관리 |

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LangGraph Workflow                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐    ┌───────────────────┐    ┌──────────┐                │
│   │ classify     │───▶│ extract           │───▶│ plan     │                │
│   │ _intent      │    │ _preferences      │    │          │                │
│   └──────────────┘    └───────────────────┘    └────┬─────┘                │
│                                                      │                      │
│                                                      ▼                      │
│   ┌──────────────┐    ┌───────────────────┐    ┌──────────┐                │
│   │ save_memory  │◀───│ synthesize        │◀───│ research │                │
│   │              │    │                   │    │ (tools)  │                │
│   └──────┬───────┘    └───────────────────┘    └──────────┘                │
│          │                      ▲                                           │
│          │            ┌─────────┴─────────┐                                 │
│          │            │                   │                                 │
│   ┌──────▼───────┐    │    ┌──────────┐   │                                │
│   │   evaluate   │────┼───▶│ improve  │───┘                                │
│   │  (품질 평가) │    │    │ (개선)   │                                    │
│   └──────────────┘    │    └──────────┘                                    │
│        pass ──────────┘                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 핵심 컴포넌트

- **State**: `TypedDict` + `Annotated[list, operator.add]`를 활용한 상태 누적
- **Tools**: 3가지 도구 유형 (RAG, Python, API)
- **Nodes**: 8개 노드로 구성된 워크플로우
- **Edges**: 조건부 라우팅 (evaluate → improve/save_memory)

---

## 설치 및 실행

### 사전 요구사항

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) 패키지 매니저

### 1. 의존성 설치

```bash
cd news_chatbot_agent
uv sync
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 API 키를 입력합니다:

```env
# Upstage API (LLM + Embeddings)
UPSTAGE_API_KEY=up_xxxxx

# SERPER API (실시간 뉴스 검색)
SERPER_API_KEY=xxxxx

# LangSmith (선택사항 - 추적 및 평가)
LANGSMITH_API_KEY=lsv2_xxxxx
LANGSMITH_TRACING=true
```

### 3. 실행 방법

#### CLI 실행
```bash
python main.py
```

#### LangGraph Studio (시각화)
```bash
langgraph dev
```

브라우저에서 `http://localhost:8123`으로 접속하여 워크플로우를 시각적으로 확인할 수 있습니다.

---

## 프로젝트 구조

```
news_chatbot_agent/
├── agent/
│   ├── __init__.py
│   ├── state.py          # AgentState 정의 (TypedDict)
│   ├── prompts.py        # 프롬프트 템플릿 (의도분류, 합성, 평가 등)
│   ├── tools.py          # 3가지 도구 정의
│   ├── nodes.py          # 8개 워크플로우 노드
│   └── graph.py          # StateGraph 조립 및 컴파일
├── data/
│   └── news_archive.json # Mock 뉴스 데이터 (RAG용)
├── tests/
│   ├── test_tools.py     # 도구 단위 테스트
│   ├── test_nodes.py     # 노드 단위 테스트
│   ├── test_graph.py     # 그래프 통합 테스트
│   └── test_multiturn.py # 멀티턴 대화 테스트
├── activities/
│   ├── STUDENT_ACTIVITIES.md         # 학습 활동 (빈칸 채우기)
│   └── STUDENT_ACTIVITIES_ANSWERS.md # 정답지
├── scripts/
│   └── evaluate.py       # LangSmith 평가 스크립트
├── main.py               # CLI 진입점
├── langgraph.json        # LangGraph Studio 설정
├── pyproject.toml        # 프로젝트 의존성
└── .env.example          # 환경 변수 템플릿
```

---

## 도구 상세

### 1. search_news_archive (RAG)

저장된 뉴스 아카이브에서 유사도 기반 검색을 수행합니다.

```python
@tool
def search_news_archive(query: str) -> str:
    """뉴스 아카이브에서 관련 기사를 검색합니다."""
```

**기술 스택:**
- **벡터 스토어**: FAISS (Facebook AI Similarity Search)
- **임베딩 모델**: Upstage `solar-embedding-1-large`
- **검색 결과**: 상위 3개 문서 반환

### 2. calculate_date_range (Python Function)

상대적 시간 표현을 실제 날짜 범위로 변환합니다.

```python
@tool
def calculate_date_range(time_value: int, time_unit: str) -> str:
    """상대적 날짜를 계산합니다. 예: (7, 'days') → 7일 전 ~ 오늘"""
```

**지원 단위:**
- `days`: 일 단위
- `weeks`: 주 단위
- `months`: 월 단위

**예시:**
```
"7일 전" → 2025-01-26 ~ 2025-02-02
"2주 전" → 2025-01-19 ~ 2025-02-02
```

### 3. search_recent_news (External API)

SERPER API를 활용하여 실시간 뉴스를 검색합니다.

```python
@tool
def search_recent_news(query: str, date_from: str = None) -> str:
    """SERPER API로 최신 뉴스를 검색합니다."""
```

**특징:**
- 한국어 뉴스 우선 (`gl=kr`, `hl=ko`)
- 뉴스 탭 검색 (`tbm=nws`)
- 날짜 필터링 지원

---

## 워크플로우 설명

### 노드별 역할

| 노드 | 역할 | 출력 |
|------|------|------|
| `classify_intent` | 사용자 의도 분류 | `intent`: news_search/trending/summary/general |
| `extract_preferences` | 검색 선호도 추출 | `topics`, `keywords`, `date_range` |
| `plan` | 도구 실행 계획 수립 | `execution_plan` |
| `research` | 도구 호출 및 정보 수집 | `tool_results` |
| `synthesize` | 결과 종합 및 응답 생성 | `response` |
| `evaluate` | 응답 품질 평가 (1-10점) | `score`, `feedback` |
| `improve` | 피드백 기반 응답 개선 | 개선된 `response` |
| `save_memory` | 대화 히스토리 저장 | `conversation_history` 업데이트 |

### 조건부 라우팅

```python
def should_improve(state: AgentState) -> str:
    """평가 점수가 7점 미만이면 개선, 이상이면 저장"""
    if state.get("evaluation_score", 0) >= 7:
        return "save_memory"
    if state.get("improvement_count", 0) >= 2:
        return "save_memory"  # 최대 2회 개선 시도
    return "improve"
```

---

## 학습 자료

이 프로젝트는 교육용으로 설계되었습니다.

### 학습 활동

`activities/STUDENT_ACTIVITIES.md`에서 다음 개념을 실습할 수 있습니다:

1. **State 구조 이해** - TypedDict와 Annotated를 활용한 상태 관리
2. **도구 스키마 분석** - Pydantic을 활용한 도구 입력 검증
3. **감성 분석 도구 추가** - 새로운 도구를 구현하고 통합
4. **입력 검증 강화** - Pydantic Field를 활용한 입력 검증
5. **평가 기준 개선** - Evaluator 프롬프트 최적화

### 정답 확인

`activities/STUDENT_ACTIVITIES_ANSWERS.md`에서 정답을 확인할 수 있습니다.

---

## 테스트

### 단위 테스트 실행

```bash
# 전체 테스트
pytest

# 특정 테스트 파일
pytest tests/test_tools.py -v

# 특정 테스트 함수
pytest tests/test_multiturn.py::test_date_range_tool_usage -v
```

### LangSmith 평가

```bash
python scripts/evaluate.py
```

평가 지표:
- **Correctness**: 요청에 맞는 정보 제공 여부
- **Groundedness**: 검색 결과 기반 응답 여부
- **Concision**: 응답 길이 적절성

---

## 예시 대화

```
사용자: 엔비디아 관련 최근 뉴스 알려줘

챗봇: 엔비디아 관련 최신 뉴스를 정리해드립니다.

**실시간 뉴스:**
1. 엔비디아, AI 반도체 시장 점유율 80% 돌파 (2025-02-01)
2. 엔비디아 CEO "2025년 AI 컴퓨팅 수요 폭발적 성장 전망" (2025-01-30)

**아카이브 뉴스:**
- 엔비디아 4분기 실적 발표: 매출 전년 대비 55% 증가

---

사용자: 삼성전자도 관련 있어?

챗봇: 삼성전자와 엔비디아의 관계에 대한 뉴스입니다.

삼성전자는 엔비디아에 HBM(고대역폭 메모리)을 공급하고 있으며...
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **프레임워크** | LangGraph, LangChain |
| **LLM** | Upstage `solar-pro2` |
| **임베딩** | Upstage `solar-embedding-1-large` |
| **벡터 스토어** | FAISS |
| **외부 API** | SERPER (뉴스 검색) |
| **추적/평가** | LangSmith |
| **패키지 관리** | uv |

---

## 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.
