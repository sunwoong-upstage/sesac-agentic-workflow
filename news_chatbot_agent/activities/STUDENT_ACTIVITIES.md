# 뉴스 챗봇 에이전트 학습 활동지

> **대상:** AI 에이전트 패턴을 배우는 초급 개발자
>
> **사전 요구사항:** 기본 Python, Practice01-09 노트북 완료

---

## 목차

1. [활동 1: 상태(State) 정의 (빈칸 채우기)](#활동-1-상태state-정의-빈칸-채우기)
2. [활동 2: 도구(Tool) 정의 (빈칸 채우기)](#활동-2-도구tool-정의-빈칸-채우기)
3. [활동 3: 감성 분석 도구 추가](#활동-3-새-도구-추가---감성-분석)
4. [활동 4: 입력 검증 노드 추가](#활동-4-새-노드-추가---입력-검증)
5. [활동 5: 평가 시스템 개선](#활동-5-평가-최적화-루프-개선)
6. [활동 6: 후속 질문 생성 노드 추가](#활동-6-새-노드-추가---후속-질문-생성)
7. [제출 체크리스트](#제출-체크리스트)

---

## 활동 1: 상태(State) 정의 (빈칸 채우기)

**파일:** `agent/state.py`

**학습 목표:** TypedDict, Pydantic, Annotated 이해하기


```python
# =============================================================================
# 연습문제 1A: 상태(State) 정의 완성하기
# 파일: agent/state.py
# 학습 목표: TypedDict, Pydantic, Annotated 이해하기
# =============================================================================

# 문제 1: 기본 import 완성하기
import ______  # 리스트 누적 연산에 사용
from typing import Annotated, List, ______  # 특정 값들만 허용하는 타입
from ______ import BaseModel, Field  # 데이터 검증 라이브러리

# 문제 2: 메시지 누적을 위한 상태 필드 정의
class NewsChatbotState(TypedDict):
    """뉴스 챗봇 에이전트의 워크플로우 상태"""

    # 사용자 입력
    user_input: ______  # 문자열 타입

    # messages 필드: 새 메시지가 추가될 때마다 누적되어야 함
    messages: Annotated[List[BaseMessage], ______]

    # 의도 분류 결과
    intent: str
    intent_confidence: ______  # 0.0 ~ 1.0 사이 값 (실수)

    # tool_results 필드: 도구 호출 결과도 누적되어야 함
    tool_results: Annotated[______, operator.add]

    # 평가 결과
    quality_score: ______  # 1-10 점수
    evaluation_passed: ______  # True/False

# 문제 3: 의도 분류 스키마 정의 (Pydantic)
class IntentClassification(______):  # 어떤 클래스를 상속?
    """의도 분류 결과 스키마"""
    intent: Literal["news_search", "trending", "summary", "general"] = ______(
        description="사용자 의도 유형"
    )
    confidence: float = Field(______=0.0, ______=1.0, description="분류 신뢰도")
    reasoning: str = Field(description="______")

# 문제 4: 날짜 범위 스키마 - 필드 검증 추가
class DateRange(BaseModel):
    """상대적 날짜 범위 스키마"""
    time_value: int = Field(description="시간 값", ______=0)  # 최소 0 이상
    time_unit: Literal["days", "______", "months"] = Field(description="시간 단위")

# 문제 5: 뉴스 선호도 추출 스키마 - Optional 필드
class NewsPreferences(BaseModel):
    """대화에서 추출한 뉴스 선호도 스키마"""
    # 관심 주제 리스트
    topics: ______ = Field(default_factory=list, description="관심 주제")
    # Optional: 값이 없을 수 있음 (None 허용)
    date_range: ______ = Field(default=______, description="날짜 범위")
```

<details>
<summary>정답 보기</summary>

```python
# 문제 1
import operator
from typing import Annotated, List, Literal
from pydantic import BaseModel, Field

# 문제 2
user_input: str
messages: Annotated[List[BaseMessage], operator.add]
intent_confidence: float
tool_results: Annotated[List[dict], operator.add]
quality_score: int
evaluation_passed: bool

# 문제 3
class IntentClassification(BaseModel):
    intent: ... = Field(description="사용자 의도 유형")
    confidence: float = Field(ge=0.0, le=1.0, ...)
    reasoning: str = Field(description="분류 근거")

# 문제 4
time_value: int = Field(..., ge=0)
time_unit: Literal["days", "weeks", "months"]

# 문제 5
topics: List[str] = Field(...)
date_range: Optional[DateRange] = Field(default=None, ...)
```

</details>

---

## 활동 2: 도구(Tool) 정의 (빈칸 채우기)

**파일:** `agent/tools.py`

**학습 목표:** @tool 데코레이터, Pydantic 스키마, FAISS 이해하기

```python
# =============================================================================
# 연습문제 2: 도구(Tool) 정의 완성하기
# 파일: agent/tools.py
# 학습 목표: @tool 데코레이터, Pydantic 스키마, FAISS 이해하기
# =============================================================================

from langchain_core.tools import ______  # 도구 데코레이터
from pydantic import BaseModel, Field
from datetime import datetime, ______  # 시간 간격 클래스

# 문제 1: 도구 입력 스키마 정의
class SearchNewsArchiveInput(______):  # 어떤 클래스 상속?
    query: str = ______(description="검색 키워드 또는 질문")

class CalculateDateRangeInput(BaseModel):
    time_value: int = Field(description="시간 값 (예: 7)", ______=0)  # 최소값 검증
    time_unit: Literal["days", "weeks", "months"] = Field(description="시간 단위")

# 문제 2: FAISS 벡터 스토어 초기화
def _initialize_vector_store():
    """FAISS 벡터 스토어 초기화"""
    from langchain_upstage import ______  # 임베딩 클래스
    from langchain_community.vectorstores import ______  # 벡터 스토어 클래스

    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    documents = _create_news_documents()
    vector_store = ______.from_documents(documents, embeddings)  # 문서로부터 생성
    return vector_store

# 문제 3: 뉴스 아카이브 검색 도구 정의
@______(args_schema=SearchNewsArchiveInput)  # 데코레이터 이름
def search_news_archive(query: str) -> str:
    """저장된 뉴스 아카이브에서 관련 기사를 검색합니다."""
    vector_store = _get_vector_store()
    # similarity_search: 유사도 기반 검색, k=3은 상위 3개 결과
    docs = vector_store.______(query, k=______)
    result = "\n\n".join(
        f"[{doc.metadata.get('category')}] {doc.______.get('title')}\n{doc.______}"
        for doc in docs
    )
    return result

# 문제 4: 날짜 계산 도구 (Python 함수)
@tool(args_schema=CalculateDateRangeInput)
def calculate_date_range(time_value: int, time_unit: str) -> str:
    """상대적 날짜 표현을 실제 날짜로 변환합니다."""
    today = datetime.______()  # 현재 날짜/시간 가져오기

    if time_unit == "days":
        delta = timedelta(______=time_value)
    elif time_unit == "weeks":
        delta = timedelta(______=time_value)
    elif time_unit == "months":
        delta = timedelta(days=time_value ______ 30)  # 곱하기 연산자

    start_date = today ______ delta  # 빼기 연산자
    end_date = today

    return f"검색 기간: {start_date.______('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"

# 문제 5: 웹 검색 도구 (API 호출)
@tool(args_schema=SearchRecentNewsInput)
def search_recent_news(query: str) -> str:
    """SERPER API를 사용하여 최신 뉴스를 검색합니다."""
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
        "https://google.serper.dev/news",
        headers=headers,
        json={"q": query, "gl": "kr", "hl": "ko"},
        timeout=10,
    )
    response.______()  # 에러 시 예외 발생
    return response.json()

# 문제 6: 도구 리스트 정의 (research_node에서 사용)
TOOLS = [______, ______, ______]  # 3개 도구
```

<details>
<summary>정답 보기</summary>

```python
# 문제 1
from langchain_core.tools import tool
from datetime import datetime, timedelta
class SearchNewsArchiveInput(BaseModel):
    query: str = Field(description="...")
class CalculateDateRangeInput(BaseModel):
    time_value: int = Field(..., ge=0)

# 문제 2
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
vector_store = FAISS.from_documents(documents, embeddings)

# 문제 3
@tool(args_schema=SearchNewsArchiveInput)
docs = vector_store.similarity_search(query, k=3)
f"[{doc.metadata.get('category')}] {doc.metadata.get('title')}\n{doc.page_content}"

# 문제 4
today = datetime.now()
delta = timedelta(days=time_value)
delta = timedelta(weeks=time_value)
delta = timedelta(days=time_value * 30)
start_date = today - delta
start_date.strftime('%Y-%m-%d')

# 문제 5
import requests
api_key = os.getenv("SERPER_API_KEY")
"Content-Type": "application/json"
response = requests.post(...)
response.raise_for_status()

# 문제 6
TOOLS = [search_news_archive, calculate_date_range, search_recent_news]
```

</details>

---

## 활동 3: 새 도구 추가 - 감성 분석

**수정할 파일:** `agent/tools.py`

#### 목표
뉴스 기사의 감성(긍정/부정/중립)을 분석하는 도구를 추가합니다.

#### Step 1: Pydantic 입력/출력 스키마 정의

```python
class SentimentAnalysisInput(BaseModel):
    """감성 분석 입력"""
    text: str = Field(description="분석할 텍스트")

class SentimentResult(BaseModel):
    """감성 분석 결과"""
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="감성")
    score: float = Field(ge=-1.0, le=1.0, description="감성 점수 (-1.0 ~ 1.0)")
    keywords: List[str] = Field(description="핵심 키워드")
```

#### Step 2: 도구 함수 구현

```python
@tool(args_schema=SentimentAnalysisInput)
def analyze_sentiment(text: str) -> str:
    """뉴스 기사의 감성을 분석합니다."""
    logger.info(f"[Tool Call] analyze_sentiment | text='{text[:50]}...'")

    # TODO: 여러분이 구현하세요!
    # 1. LLM을 사용하여 감성 분석
    # 2. SentimentResult 스키마로 구조화된 출력 받기
    # 3. 결과 문자열 반환

    pass  # 이 부분을 구현하세요
```

#### Step 3: TOOLS에 추가

```python
TOOLS = [search_news_archive, calculate_date_range, search_recent_news, analyze_sentiment]
```

#### 검증 체크리스트

- [ ] `analyze_sentiment("엔비디아 주가 사상 최고치 경신")` → 긍정 감성
- [ ] `analyze_sentiment("경기 침체 우려로 증시 하락")` → 부정 감성
- [ ] 감성 점수와 키워드가 함께 출력됨
- [ ] `main.py` 실행 시 에러 없음

---

## 활동 4: 새 노드 추가 - 입력 검증

**수정할 파일:** `agent/state.py`, `agent/nodes.py`, `agent/graph.py`

#### 목표

사용자 입력을 검증하여 뉴스와 무관한 입력을 조기에 필터링합니다.

#### Step 1: state.py에 새 필드 추가

```python
class NewsChatbotState(TypedDict):
    # ... 기존 필드들 ...

    is_valid_input: bool
    """입력 유효성 검사 결과"""

    validation_message: str
    """유효성 검사 메시지 (실패 시)"""
```

#### Step 2: nodes.py에 검증 노드 함수 작성

```python
def validate_input_node(state: NewsChatbotState) -> dict:
    """사용자 입력을 검증합니다."""
    logger.info("[Node] validate_input 시작")
    query = state.get("user_input", "")

    # 검증 규칙 정의
    NEWS_KEYWORDS = [
        "뉴스", "기사", "소식", "속보", "발표",
        "최신", "트렌드", "이슈", "검색", "알려줘"
    ]

    # TODO: 검증 로직 구현
    # 1. 최소 길이 검사 (2자 이상)
    # 2. 뉴스 관련 키워드 포함 여부
    # 3. 적절한 상태 반환

    pass  # 구현하세요
```

#### Step 3: graph.py 수정

```python
# 노드 추가 (validate_input을 맨 앞에)
builder.add_node("validate_input", validate_input_node)

# 엣지 수정: START -> validate_input
builder.add_edge(START, "validate_input")

# 조건부 엣지: 유효하면 classify로, 아니면 바로 종료
builder.add_conditional_edges(
    "validate_input",
    lambda s: "valid" if s.get("is_valid_input", True) else "invalid",
    {"valid": "classify_intent", "invalid": "save_memory"}
)
```

#### 검증 체크리스트

- [ ] 빈 입력 "" → `validation_message` 출력 후 종료
- [ ] "피자 맛집 추천해줘" → 뉴스 관련 아님 안내
- [ ] "엔비디아 뉴스 알려줘" → 정상적으로 파이프라인 진행
- [ ] `main.py` 실행 시 정상 동작

---

## 활동 5: 평가-최적화 루프 개선

**수정할 파일:** `agent/state.py`, `agent/nodes.py`, `agent/prompts.py`

#### 목표
품질 평가를 세분화하여 정확성, 관련성, 완성도, 가독성 각각을 평가합니다.

#### Step 1: state.py - 세분화된 평가 스키마 추가

```python
class DetailedQualityEvaluation(BaseModel):
    """세분화된 응답 품질 평가 스키마"""

    accuracy_score: int = Field(description="정보 정확성 (1-10)", ge=1, le=10)
    relevance_score: int = Field(description="질문 관련성 (1-10)", ge=1, le=10)
    completeness_score: int = Field(description="정보 완성도 (1-10)", ge=1, le=10)
    readability_score: int = Field(description="가독성/구성 (1-10)", ge=1, le=10)

    overall_score: int = Field(description="종합 점수 (1-10)", ge=1, le=10)
    weakest_area: Literal["accuracy", "relevance", "completeness", "readability"]
    improvement_suggestion: str = Field(description="구체적 개선 제안")
```

#### Step 2: 세분화된 평가 노드 구현

```python
def evaluate_response_detailed_node(state: NewsChatbotState) -> dict:
    """세분화된 품질 평가를 수행합니다."""
    # TODO: DetailedQualityEvaluation 스키마로 구조화된 출력 받기
    pass
```

#### 검증 체크리스트

- [ ] 4가지 점수가 모두 출력됨
- [ ] `weakest_area`가 정확히 식별됨
- [ ] 기존 기능이 깨지지 않음

---

## 활동 6: 새 노드 추가 - 후속 질문 생성

**수정할 파일:** `agent/state.py`, `agent/nodes.py`, `agent/graph.py`

#### 목표

응답 생성 후 사용자가 이어서 물어볼 만한 후속 질문 3개를 자동 생성합니다.
LLM의 구조화된 출력(structured output)을 활용하여 대화형 UX를 개선합니다.

#### Step 1: state.py에 스키마와 필드 추가

```python
class FollowUpQuestions(BaseModel):
    """후속 질문 생성 스키마"""
    questions: List[str] = Field(description="후속 질문 리스트 (3개)")
    reasoning: str = Field(description="질문 생성 근거")


class NewsChatbotState(TypedDict):
    # ... 기존 필드들 ...

    follow_up_questions: List[str]
    """생성된 후속 질문"""
```

#### Step 2: nodes.py에 후속 질문 생성 노드 작성

```python
def generate_follow_up_node(state: NewsChatbotState) -> dict:
    """후속 질문을 생성합니다."""
    logger.info("[Node] generate_follow_up 시작")

    # TODO: 구현하세요
    # 1. 사용자 입력과 최종 응답을 기반으로 프롬프트 작성
    # 2. LLM에 FollowUpQuestions 스키마로 구조화된 출력 요청
    # 3. follow_up_questions 필드 반환

    pass  # 구현하세요
```

#### Step 3: graph.py 수정

```python
# 노드 추가
builder.add_node("generate_follow_up", generate_follow_up_node)

# 엣지 수정: save_memory 전에 후속 질문 생성
# evaluate → (pass) → generate_follow_up → save_memory
```

#### 검증 체크리스트

- [ ] "엔비디아 뉴스" 질문 시 후속 질문 3개 생성됨
- [ ] 후속 질문이 원본 주제와 연관성이 있음
- [ ] `main.py` 실행 시 에러 없음
- [ ] 기존 기능이 깨지지 않음

---

## 제출 체크리스트

### 빈칸 채우기 (활동 1, 2)

- [ ] 모든 빈칸을 채웠는가?
- [ ] Python 문법 오류가 없는가?
- [ ] 실제 코드와 비교하여 검증했는가?
- [ ] 각 개념을 설명할 수 있는가?

### 구현 실습 (활동 3-6)

- [ ] 코드가 에러 없이 실행되는가?
- [ ] 테스트 케이스를 모두 통과하는가?
- [ ] 기존 기능이 깨지지 않았는가?
- [ ] `main.py` 실행 시 정상 동작하는가?
- [ ] 코드에 적절한 주석을 달았는가?
- [ ] 로깅을 추가했는가?

