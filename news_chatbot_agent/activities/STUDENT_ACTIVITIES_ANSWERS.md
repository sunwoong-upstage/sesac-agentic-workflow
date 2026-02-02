# 뉴스 챗봇 에이전트 학습 활동지 - 정답 및 해설

> **대상:** AI 에이전트 패턴을 배우는 초급 개발자
> **사전 요구사항:** 기본 Python, Practice01-09 노트북 완료
> **예상 소요 시간:** 총 4-6시간

---

## 목차

1. [활동 1: 상태(State) 정의 - 정답](#활동-1-상태state-정의---정답)
2. [활동 2: 도구(Tool) 정의 - 정답](#활동-2-도구tool-정의---정답)
3. [활동 3: 감성 분석 도구 추가 - 예시 구현](#활동-3-감성-분석-도구-추가---예시-구현)
4. [활동 4: 입력 검증 노드 추가 - 예시 구현](#활동-4-입력-검증-노드-추가---예시-구현)
5. [활동 5: 평가 시스템 개선 - 예시 구현](#활동-5-평가-시스템-개선---예시-구현)

---

## 활동 1: 상태(State) 정의 - 정답

**파일:** `agent/state.py`

```python
# 문제 1: 기본 import 완성하기
import operator  # 누적 연산을 위한 모듈 (힌트: add 함수 사용)
from typing import Annotated, List, Literal  # 제한된 값만 허용하는 타입
from pydantic import BaseModel, Field  # 데이터 검증 라이브러리

# 문제 2: 메시지 누적을 위한 상태 필드 정의
class NewsChatbotState(TypedDict):
    """뉴스 챗봇 에이전트의 워크플로우 상태"""

    # 사용자 입력
    user_input: str  # 문자열 타입

    # messages 필드: 새 메시지가 추가될 때마다 누적되어야 함
    messages: Annotated[List[BaseMessage], operator.add]

    # 의도 분류 결과
    intent: str
    intent_confidence: float  # 0.0 ~ 1.0 사이 값 (실수)

    # tool_results 필드: 도구 호출 결과도 누적되어야 함
    tool_results: Annotated[List[dict], operator.add]

    # 평가 결과
    quality_score: int  # 1-10 점수
    evaluation_passed: bool  # True/False

# 문제 3: 의도 분류 스키마 정의 (Pydantic)
class IntentClassification(BaseModel):  # BaseModel을 상속
    """의도 분류 결과 스키마"""
    intent: Literal[
        "news_search",
        "trending",  # 트렌딩 뉴스 요청
        "summary",
        "general"
    ] = Field(description="사용자 의도 유형")
    confidence: float = Field(ge=0.0, le=1.0, description="분류 신뢰도")
    reasoning: str = Field(description="분류 근거")

# 문제 4: 날짜 범위 스키마 - 필드 검증 추가
class DateRange(BaseModel):
    """상대적 날짜 범위 스키마"""
    time_value: int = Field(description="시간 값", ge=0)  # 최소 0 이상
    time_unit: Literal["days", "weeks", "months"] = Field(description="시간 단위")

# 문제 5: 뉴스 선호도 추출 스키마 - Optional 필드
class NewsPreferences(BaseModel):
    """대화에서 추출한 뉴스 선호도 스키마"""
    # 관심 주제 리스트
    topics: List[str] = Field(default_factory=list, description="관심 주제")
    # Optional: 값이 없을 수 있음 (None 허용)
    date_range: Optional[DateRange] = Field(default=None, description="날짜 범위")
```

#### 해설

**문제 1: Import 이해하기**
- `operator`: Python 내장 모듈로, `operator.add`는 리스트 누적(concatenation)에 사용됩니다.
- `Literal`: 특정 값만 허용하는 타입 (예: "news_search", "trending"만 가능)
- `pydantic`: 데이터 검증을 위한 라이브러리. `BaseModel`로 스키마 정의, `Field`로 검증 규칙 추가

**문제 2: Annotated와 operator.add**
```python
messages: Annotated[List[BaseMessage], operator.add]
```
- `Annotated`는 타입에 메타데이터를 추가합니다.
- `operator.add`는 LangGraph에게 "이 필드는 덮어쓰지 말고 누적하라"고 지시합니다.
- 예: 첫 노드에서 `["안녕"]`, 두 번째 노드에서 `["반가워"]` 반환 → 최종 상태는 `["안녕", "반가워"]`

**문제 3: Pydantic Literal과 Field**
- `Literal["news_search", "trending", ...]`: 이 4가지 값만 허용 (오타 방지)
- `Field(description="...")`: 스키마 문서화, LLM에게 힌트 제공
- `ge=0.0, le=1.0`: greater-than-or-equal (≥), less-than-or-equal (≤) 검증

**문제 4: 필드 검증**
```python
time_value: int = Field(description="...", ge=0)
```
- `ge=0`: 음수 불가 (0 이상만 허용)
- 잘못된 값 입력 시 Pydantic이 자동으로 `ValidationError` 발생

**문제 5: Optional 타입**
```python
date_range: Optional[DateRange] = Field(default=None, ...)
```
- `Optional[X]`는 `Union[X, None]`과 동일 (값이 있을 수도, 없을 수도 있음)
- `default=None`: 값이 주어지지 않으면 `None`으로 초기화

---

## 활동 2: 도구(Tool) 정의 - 정답

**파일:** `agent/tools.py`

```python
# 문제 1: 기본 import
from langchain_core.tools import tool  # 도구 데코레이터
from datetime import datetime, timedelta  # 시간 간격 클래스

class SearchNewsArchiveInput(BaseModel):  # BaseModel 상속
    query: str = Field(description="검색 키워드 또는 질문")

class CalculateDateRangeInput(BaseModel):
    time_value: int = Field(description="시간 값 (예: 7)", ge=0)  # 최소값 검증

# 문제 2: FAISS 벡터 스토어 초기화
def _initialize_vector_store():
    """FAISS 벡터 스토어 초기화"""
    from langchain_upstage import UpstageEmbeddings  # 임베딩 클래스
    from langchain_community.vectorstores import FAISS  # 벡터 스토어 클래스

    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    documents = _create_news_documents()
    vector_store = FAISS.from_documents(documents, embeddings)  # 문서로부터 생성
    return vector_store

# 문제 3: 뉴스 아카이브 검색 도구 정의
@tool(args_schema=SearchNewsArchiveInput)  # 데코레이터 이름
def search_news_archive(query: str) -> str:
    """저장된 뉴스 아카이브에서 관련 기사를 검색합니다."""
    vector_store = _get_vector_store()
    # similarity_search: 유사도 기반 검색, k=3은 상위 3개 결과
    docs = vector_store.similarity_search(query, k=3)
    result = "\n\n".join(
        f"[{doc.metadata.get('category')}] {doc.metadata.get('title')}\n{doc.page_content}"
        for doc in docs
    )
    return result

# 문제 4: 날짜 계산 도구 (Python 함수)
@tool(args_schema=CalculateDateRangeInput)
def calculate_date_range(time_value: int, time_unit: str) -> str:
    """상대적 날짜 표현을 실제 날짜로 변환합니다."""
    today = datetime.now()  # 현재 날짜/시간 가져오기

    if time_unit == "days":
        delta = timedelta(days=time_value)
    elif time_unit == "weeks":
        delta = timedelta(weeks=time_value)
    elif time_unit == "months":
        delta = timedelta(days=time_value * 30)  # 곱하기 연산자

    start_date = today - delta  # 빼기 연산자
    end_date = today

    return f"검색 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"

# 문제 5: 웹 검색 도구 (API 호출)
@tool(args_schema=SearchRecentNewsInput)
def search_recent_news(query: str) -> str:
    """SERPER API를 사용하여 최신 뉴스를 검색합니다."""
    import os
    import requests  # HTTP 요청 라이브러리

    api_key = os.getenv("SERPER_API_KEY")  # 환경변수 읽기
    if not api_key:
        return "SERPER_API_KEY 미설정"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",  # JSON 컨텐츠 타입
    }

    response = requests.post(  # POST 메서드
        "https://google.serper.dev/news",
        headers=headers,
        json={"q": query, "gl": "kr", "hl": "ko"},
        timeout=10,
    )
    response.raise_for_status()  # 에러 시 예외 발생
    return response.json()

# 문제 6: 도구 리스트 정의 (research_node에서 사용)
TOOLS = [search_news_archive, calculate_date_range, search_recent_news]  # 3개 도구
```

#### 해설

**문제 1: @tool 데코레이터**
```python
@tool(args_schema=SearchNewsArchiveInput)
def search_news_archive(query: str) -> str:
    ...
```
- `@tool`: LangChain 도구로 변환 (LLM이 호출 가능)
- `args_schema`: 입력 스키마 지정 (LLM에게 인자 설명 제공)
- 도구는 반드시 `str`을 반환해야 함 (LLM이 읽을 수 있는 형식)

**문제 2: FAISS 벡터 스토어**
```python
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
vector_store = FAISS.from_documents(documents, embeddings)
```
- **UpstageEmbeddings**: 텍스트를 벡터(숫자 배열)로 변환
- **FAISS**: Facebook AI가 개발한 유사도 검색 라이브러리 (빠르고 메모리 효율적)
- **from_documents**: 문서 리스트를 받아 임베딩 생성 후 인덱스 구축

**문제 3: similarity_search**
```python
docs = vector_store.similarity_search(query, k=3)
```
- 쿼리를 임베딩으로 변환 후, 코사인 유사도 계산
- `k=3`: 가장 유사한 상위 3개 문서 반환
- 반환값: `List[Document]` (각 문서는 `page_content`와 `metadata` 포함)

**문제 4: datetime과 timedelta**
```python
today = datetime.now()
delta = timedelta(days=7)
start_date = today - delta
```
- `datetime.now()`: 현재 날짜/시간 객체
- `timedelta`: 시간 간격 (days, weeks, hours 등)
- 날짜 연산 가능: `datetime - timedelta = datetime`

**문제 5: requests와 환경변수**
```python
api_key = os.getenv("SERPER_API_KEY")
response = requests.post(url, headers=headers, json=payload)
response.raise_for_status()
```
- `os.getenv()`: 환경변수 읽기 (API 키를 코드에 하드코딩하지 않기 위함)
- `requests.post()`: HTTP POST 요청
- `raise_for_status()`: 4xx/5xx 에러 시 예외 발생 (에러 핸들링 용이)

---

## 활동 3-6: 실습 구현 - 예시 정답

> **주의:** 아래는 **예시 구현**입니다. 정답은 여러 가지가 있을 수 있습니다.

---

### 활동 3: 감성 분석 도구 추가 - 예시 구현

**수정할 파일:** `agent/tools.py`

#### Step 1-3: 전체 구현 예시

```python
# agent/tools.py

# 1. Pydantic 스키마 정의 (파일 상단에 추가)
class SentimentAnalysisInput(BaseModel):
    """감성 분석 입력"""
    text: str = Field(description="분석할 텍스트 (기사 제목 또는 내용)")


class SentimentResult(BaseModel):
    """감성 분석 결과"""
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="감성 분류")
    score: float = Field(ge=-1.0, le=1.0, description="감성 점수 (-1.0 ~ 1.0)")
    keywords: List[str] = Field(description="감성을 결정한 핵심 키워드")
    reasoning: str = Field(description="분석 근거")


# 2. 도구 함수 구현 (Tool 3 아래에 추가)
@tool(args_schema=SentimentAnalysisInput)
def analyze_sentiment(text: str) -> str:
    """뉴스 기사의 감성을 분석합니다.

    긍정적(positive), 부정적(negative), 중립적(neutral) 감성을 판단하고
    -1.0(매우 부정)부터 1.0(매우 긍정) 사이의 점수를 반환합니다.

    Args:
        text: 분석할 텍스트 (기사 제목 또는 내용)

    Returns:
        감성 분석 결과 (감성, 점수, 키워드, 근거)
    """
    logger.info(f"[Tool Call] analyze_sentiment | text='{text[:50]}...'")

    from langchain_upstage import ChatUpstage
    from langchain_core.messages import SystemMessage

    # 감성 분석용 프롬프트
    prompt = f"""다음 텍스트의 감성을 분석하세요.

텍스트: {text}

분석 기준:
- positive: 긍정적 전망, 성장, 성공, 기회 등
- negative: 부정적 전망, 하락, 실패, 위험 등
- neutral: 사실 전달, 중립적 정보

점수 범위:
- 1.0 ~ 0.5: 매우 긍정적
- 0.5 ~ 0.1: 약간 긍정적
- 0.1 ~ -0.1: 중립적
- -0.1 ~ -0.5: 약간 부정적
- -0.5 ~ -1.0: 매우 부정적
"""

    try:
        llm = ChatUpstage(model="solar-pro2")
        structured_llm = llm.with_structured_output(SentimentResult)
        result = structured_llm.invoke([SystemMessage(content=prompt)])

        output = (
            f"감성 분석 결과:\n"
            f"- 감성: {result.sentiment}\n"
            f"- 점수: {result.score:.2f}\n"
            f"- 핵심 키워드: {', '.join(result.keywords)}\n"
            f"- 분석 근거: {result.reasoning}"
        )

        logger.info(f"[Tool Result] 감성={result.sentiment}, 점수={result.score}")
        return output

    except Exception as e:
        logger.error(f"[Tool Error] analyze_sentiment 실패: {e}")
        return f"감성 분석 중 오류가 발생했습니다: {e}"


# 3. TOOLS 리스트에 추가
TOOLS = [
    search_news_archive,
    calculate_date_range,
    search_recent_news,
    analyze_sentiment,  # 추가
]
```

#### 테스트 코드

```python
# 터미널에서 실행
python -c "
from agent.tools import analyze_sentiment

# 긍정 테스트
print(analyze_sentiment.invoke({'text': '엔비디아 주가 사상 최고치 경신, AI 칩 수요 폭발'}))
print('\n---\n')

# 부정 테스트
print(analyze_sentiment.invoke({'text': '경기 침체 우려로 증시 하락, 투자자 불안 확산'}))
print('\n---\n')

# 중립 테스트
print(analyze_sentiment.invoke({'text': '삼성전자, 2024년 4분기 실적 발표 예정'}))
"
```

#### 검증 체크리스트

- [x] `analyze_sentiment("엔비디아 주가 사상 최고치 경신")` → 긍정 감성
- [x] `analyze_sentiment("경기 침체 우려로 증시 하락")` → 부정 감성
- [x] 감성 점수와 키워드가 함께 출력됨
- [x] `main.py` 실행 시 에러 없음

---

### 활동 4: 입력 검증 노드 추가 - 예시 구현

**수정할 파일:** `agent/state.py`, `agent/nodes.py`, `agent/graph.py`

#### Step 1: state.py에 새 필드 추가

```python
# agent/state.py

class NewsChatbotState(TypedDict):
    """뉴스 챗봇 에이전트 상태"""
    # ... 기존 필드들 ...

    # 입력 검증 (새로 추가)
    is_valid_input: bool
    """입력 유효성 검사 결과"""

    validation_message: str
    """유효성 검사 메시지 (실패 시)"""
```

#### Step 2: nodes.py에 검증 노드 함수 작성

```python
# agent/nodes.py

def validate_input_node(state: NewsChatbotState) -> dict:
    """사용자 입력을 검증합니다.

    검증 규칙:
    1. 최소 길이: 2자 이상
    2. 뉴스 관련 키워드 포함 여부
    3. 금지어 확인 (욕설, 스팸 등)

    Args:
        state: 현재 상태

    Returns:
        is_valid_input, validation_message 업데이트
    """
    logger.info("[Node] validate_input_node 시작")
    query = state.get("user_input", "")

    # 규칙 1: 최소 길이 검사
    if len(query.strip()) < 2:
        logger.warning(f"[Node] 입력이 너무 짧음: '{query}'")
        return {
            "is_valid_input": False,
            "validation_message": "질문이 너무 짧습니다. 최소 2자 이상 입력해주세요."
        }

    # 규칙 2: 뉴스 관련 키워드 확인
    NEWS_KEYWORDS = [
        "뉴스", "기사", "소식", "속보", "발표", "트렌드", "이슈",
        "최신", "검색", "알려줘", "알려주세요", "찾아줘", "찾아주세요",
        # 도메인 키워드
        "AI", "반도체", "주가", "증시", "경제", "IT", "정치", "사회",
        "엔비디아", "삼성", "테슬라", "애플", "카카오", "네이버"
    ]

    query_lower = query.lower()
    contains_news_keyword = any(keyword.lower() in query_lower for keyword in NEWS_KEYWORDS)

    if not contains_news_keyword:
        logger.warning(f"[Node] 뉴스 관련 키워드 없음: '{query}'")
        return {
            "is_valid_input": False,
            "validation_message": (
                "뉴스 검색 챗봇입니다. 뉴스 관련 질문을 입력해주세요.\n"
                "예: '엔비디아 뉴스 알려줘', '최신 AI 트렌드 검색'"
            )
        }

    # 규칙 3: 금지어 확인 (선택 사항)
    BANNED_WORDS = ["욕설1", "스팸2"]  # 실제로는 더 많은 단어 추가
    if any(banned in query_lower for banned in BANNED_WORDS):
        logger.warning(f"[Node] 금지어 감지: '{query}'")
        return {
            "is_valid_input": False,
            "validation_message": "부적절한 단어가 포함되어 있습니다."
        }

    # 모든 검증 통과
    logger.info(f"[Node] 입력 검증 통과")
    return {
        "is_valid_input": True,
        "validation_message": ""
    }
```

#### Step 3: graph.py 수정

```python
# agent/graph.py

from agent.nodes import (
    # ... 기존 import ...
    validate_input_node,  # 새로 추가
)

def create_news_chatbot_graph(with_memory: bool = True):
    """뉴스 챗봇 그래프 생성"""
    # ... 기존 코드 ...

    # 노드 추가 (맨 앞에)
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("classify_intent", classify_intent_node)
    # ... 나머지 노드들 ...

    # 엣지 수정
    builder.set_entry_point("validate_input")  # START → validate_input

    # 조건부 엣지: 유효하면 classify로, 아니면 바로 종료
    builder.add_conditional_edges(
        "validate_input",
        lambda s: "valid" if s.get("is_valid_input", True) else "invalid",
        {
            "valid": "classify_intent",
            "invalid": "save_memory"  # 검증 실패 시 바로 종료
        }
    )

    # 기존 classify_intent 엣지는 그대로 유지
    builder.add_conditional_edges(
        "classify_intent",
        lambda s: "skip" if s.get("intent") == "general" else "continue",
        {"continue": "extract_preferences", "skip": "save_memory"}
    )

    # ... 나머지 엣지들 ...
```

#### 검증 체크리스트

- [x] 빈 입력 "" → `validation_message` 출력 후 종료
- [x] "피자 맛집 추천해줘" → 뉴스 관련 아님 안내
- [x] "엔비디아 뉴스 알려줘" → 정상적으로 파이프라인 진행
- [x] `main.py` 실행 시 정상 동작

---

### 활동 5: 평가 시스템 개선 - 예시 구현

**수정할 파일:** `agent/state.py`, `agent/nodes.py`, `agent/prompts.py`

#### Step 1: state.py - 세분화된 평가 스키마 추가

```python
# agent/state.py

class DetailedQualityEvaluation(BaseModel):
    """세분화된 응답 품질 평가 스키마"""

    accuracy_score: int = Field(description="정보 정확성 (1-10)", ge=1, le=10)
    relevance_score: int = Field(description="질문 관련성 (1-10)", ge=1, le=10)
    completeness_score: int = Field(description="정보 완성도 (1-10)", ge=1, le=10)
    readability_score: int = Field(description="가독성/구성 (1-10)", ge=1, le=10)

    overall_score: int = Field(description="종합 점수 (1-10)", ge=1, le=10)
    weakest_area: Literal["accuracy", "relevance", "completeness", "readability"] = Field(
        description="가장 약한 영역"
    )
    improvement_suggestion: str = Field(description="구체적 개선 제안")
    passed: bool = Field(description="통과 여부 (종합 점수 7점 이상)")


# State에 새 필드 추가
class NewsChatbotState(TypedDict):
    # ... 기존 필드들 ...

    # 세분화된 평가 결과 (선택 사항, 기존 quality_score와 병행)
    detailed_evaluation: Optional[dict]
```

#### Step 2: prompts.py - 세분화된 평가 프롬프트

```python
# agent/prompts.py

DETAILED_EVALUATION_PROMPT = """당신은 뉴스 챗봇 응답의 품질을 평가하는 전문가입니다.

사용자 질문:
{user_input}

챗봇 응답:
{response}

다음 4가지 기준으로 평가하세요 (각 1-10점):

1. **정확성 (accuracy_score)**
   - 제공된 정보가 사실인가?
   - 출처가 명확한가?
   - 오해의 소지가 없는가?

2. **관련성 (relevance_score)**
   - 사용자 질문에 직접적으로 답변하는가?
   - 불필요한 정보가 포함되지 않았는가?

3. **완성도 (completeness_score)**
   - 질문의 모든 측면을 다루었는가?
   - 추가 정보가 필요한가?

4. **가독성 (readability_score)**
   - 문장 구조가 명확한가?
   - 적절한 포맷팅(마크다운, 리스트 등)을 사용했는가?
   - 핵심이 잘 전달되는가?

**종합 점수 (overall_score)**: 4가지 점수의 가중 평균
**가장 약한 영역 (weakest_area)**: 4가지 중 가장 낮은 점수 영역
**개선 제안 (improvement_suggestion)**: 구체적이고 실행 가능한 제안
**통과 여부 (passed)**: 종합 점수 7점 이상이면 True
"""
```

#### Step 3: nodes.py - 세분화된 평가 노드 구현

```python
# agent/nodes.py

def evaluate_response_detailed_node(state: NewsChatbotState) -> dict:
    """세분화된 품질 평가를 수행합니다."""
    logger.info("[Node] evaluate_response_detailed_node 시작")

    from .prompts import DETAILED_EVALUATION_PROMPT
    from .state import DetailedQualityEvaluation

    prompt = DETAILED_EVALUATION_PROMPT.format(
        user_input=state["user_input"],
        response=state["final_response"],
    )

    structured_llm = llm.with_structured_output(DetailedQualityEvaluation)
    result = structured_llm.invoke([SystemMessage(content=prompt)])

    logger.info(
        f"[Node] 세분화 평가 결과: "
        f"정확성={result.accuracy_score}, "
        f"관련성={result.relevance_score}, "
        f"완성도={result.completeness_score}, "
        f"가독성={result.readability_score}, "
        f"종합={result.overall_score} "
        f"(통과: {result.passed})"
    )
    logger.info(f"[Node] 가장 약한 영역: {result.weakest_area}")
    logger.info(f"[Node] 개선 제안: {result.improvement_suggestion[:100]}...")

    return {
        "quality_score": result.overall_score,
        "quality_feedback": result.improvement_suggestion,
        "evaluation_passed": result.passed,
        "detailed_evaluation": {
            "accuracy": result.accuracy_score,
            "relevance": result.relevance_score,
            "completeness": result.completeness_score,
            "readability": result.readability_score,
            "weakest_area": result.weakest_area,
        },
        "iteration": state["iteration"] + 1,
    }
```

#### Step 4: graph.py에서 노드 교체 (선택 사항)

```python
# agent/graph.py

# 기존 evaluate_response_node를 evaluate_response_detailed_node로 교체
builder.add_node("evaluate", evaluate_response_detailed_node)
```

#### 검증 체크리스트

- [x] 4가지 점수가 모두 출력됨
- [x] `weakest_area`가 정확히 식별됨
- [x] 개선 제안이 구체적임
- [x] 기존 기능이 깨지지 않음

---

## 추가 학습 팁

### 디버깅 팁

**1. 로깅 레벨 조정**
```python
# main.py
import logging
logging.basicConfig(level=logging.DEBUG)  # INFO → DEBUG
```

**2. 특정 노드만 테스트**
```python
from agent.nodes import classify_intent_node
from agent.state import create_initial_state

state = create_initial_state("엔비디아 뉴스 알려줘")
result = classify_intent_node(state)
print(result)
```

**3. LangGraph Studio 활용**
- 그래프 실행 과정 시각화
- 각 노드의 입출력 확인
- 특정 노드부터 재실행

### 성능 최적화 팁

**1. 캐싱 활용**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(query):
    # 비싼 연산 (임베딩, API 호출 등)
    pass
```

**2. 병렬 도구 호출**
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(tool.invoke, args) for tool in tools]
    results = [f.result() for f in futures]
```

**3. 모델 선택 최적화**
- 간단한 작업: `solar-mini` (빠르고 저렴)
- 표준 작업: `solar-pro2` (균형)
- 복잡한 작업: `solar-pro3` (고성능)

---

## 제출 전 최종 체크리스트

### 코드 품질

- [ ] 모든 함수에 docstring 작성
- [ ] 타입 힌트 추가 (`def func(x: str) -> dict:`)
- [ ] 로깅 추가 (`logger.info`, `logger.error`)
- [ ] 예외 처리 (`try-except`)
- [ ] 매직 넘버 제거 (상수로 정의)

### 기능 검증

- [ ] `main.py` 실행 시 에러 없음
- [ ] 모든 테스트 케이스 통과
- [ ] 기존 기능이 깨지지 않음
- [ ] LangGraph Studio에서 그래프 시각화 확인

### 문서화

- [ ] README에 새 기능 설명 추가
- [ ] 코드 주석 충분히 작성
- [ ] 사용 예시 제공

---

*최종 수정일: 2026-02-02*
