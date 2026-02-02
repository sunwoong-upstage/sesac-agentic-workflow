"""뉴스 챗봇 상태 정의"""
import operator
from typing import Annotated, List, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


# Pydantic Schemas for Structured Output
class IntentClassification(BaseModel):
    """사용자 의도 분류"""
    intent: Literal["news_search", "trending", "summary", "general"] = Field(
        description="사용자 의도 유형"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="분류 신뢰도")
    reasoning: str = Field(description="분류 근거")


class DateRange(BaseModel):
    """날짜 범위"""
    start_date: str = Field(description="시작 날짜 (YYYY-MM-DD)")
    end_date: str = Field(description="종료 날짜 (YYYY-MM-DD)")
    description: str = Field(description="날짜 범위 설명")


class NewsPreferences(BaseModel):
    """뉴스 선호도"""
    topics: List[str] = Field(default_factory=list, description="관심 주제")
    keywords: List[str] = Field(default_factory=list, description="검색 키워드")
    date_range: Optional[DateRange] = Field(default=None, description="날짜 범위")


class QualityEvaluation(BaseModel):
    """응답 품질 평가"""
    score: int = Field(ge=1, le=10, description="품질 점수 (1-10)")
    passed: bool = Field(description="통과 여부 (7점 이상)")
    feedback: str = Field(description="개선 피드백")


# Main State TypedDict
class NewsChatbotState(TypedDict):
    """뉴스 챗봇 에이전트 상태"""
    # 입력
    user_input: str
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 의도 분류
    intent: str
    intent_confidence: float
    
    # 뉴스 선호도
    topics: List[str]
    keywords: List[str]
    date_range: Optional[dict]
    
    # 실행 계획
    execution_plan: str
    
    # 도구 결과
    tool_results: Annotated[List[dict], operator.add]
    
    # 최종 응답
    final_response: str
    
    # 품질 평가
    quality_score: int
    quality_feedback: str
    evaluation_passed: bool
    
    # 반복 제어
    iteration: int
    max_iterations: int
    
    # 사용자 프로필 (장기 기억)
    user_profile: dict
    
    # 에러 로그
    error_log: Annotated[List[str], operator.add]


def create_initial_state(user_input: str) -> NewsChatbotState:
    """초기 상태 생성"""
    return NewsChatbotState(
        user_input=user_input,
        messages=[],
        intent="",
        intent_confidence=0.0,
        topics=[],
        keywords=[],
        date_range=None,
        execution_plan="",
        tool_results=[],
        final_response="",
        quality_score=0,
        quality_feedback="",
        evaluation_passed=False,
        iteration=0,
        max_iterations=2,
        user_profile={},
        error_log=[],
    )
