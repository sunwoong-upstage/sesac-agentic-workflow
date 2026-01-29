"""여행 계획 에이전트 - 상태(State) 정의"""

import operator
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from dataclasses import dataclass


@dataclass
class TravelContext:
    """Runtime context for travel planning agent"""
    user_id: str = "anonymous"


class TravelPlanningState(TypedDict):
    """여행 계획 에이전트의 워크플로우 상태"""

    messages: Annotated[List[BaseMessage], operator.add]
    """대화 메시지 히스토리"""

    user_input: str
    """사용자 입력"""

    final_response: str
    """최종 응답"""

    travel_plan: str
    """Plan-and-Solve 계획 텍스트"""

    plan_steps: List[str]
    """분해된 계획 단계"""

    intent: str
    """분류된 의도"""

    tool_results: Annotated[List[dict], operator.add]
    """도구 호출 결과"""

    retrieved_context: str
    """RAG 검색 결과"""

    budget_info: str
    """예산 도구 결과"""

    web_search_info: str
    """웹 검색 결과"""

    quality_score: int
    """응답 품질 점수 (1-10)"""

    quality_feedback: str
    """품질 평가 피드백"""

    evaluation_passed: bool
    """품질 기준 통과 여부"""

    iteration: int
    """현재 반복 횟수"""

    max_iterations: int
    """최대 반복 횟수"""

    extracted_preferences: dict
    """대화에서 추출한 선호도"""

    user_profile: dict
    """사용자 프로필 (장기 메모리)"""

    skip_to_end: bool
    """파이프라인 건너뛰기 플래그"""

    error_log: Annotated[List[str], operator.add]
    """에러 로그"""


class IntentClassification(BaseModel):
    """의도 분류 결과 스키마"""
    intent: Literal[
        "destination_research",
        "itinerary_planning",
        "budget_estimation",
        "general_travel"
    ] = Field(
        description="문의 유형"
    )


class TravelPlan(BaseModel):
    """여행 계획 스키마 (Plan-and-Solve)"""
    destination: str = Field(description="주요 여행지")
    duration_days: int = Field(description="여행 기간 (일수)", ge=1)
    steps: List[str] = Field(description="실행 계획 단계")


class QualityEvaluation(BaseModel):
    """응답 품질 평가 스키마"""
    score: int = Field(description="품질 점수 (1-10)", ge=1, le=10)
    feedback: str = Field(description="개선 피드백")
    passed: bool = Field(description="품질 기준 통과 여부")


class ExtractedPreferences(BaseModel):
    """대화에서 추출한 여행 선호도 스키마"""
    destination: str | None = Field(default=None, description="여행지")
    duration_days: int | None = Field(default=None, description="여행 기간 (일수)", ge=1)
    budget: int | None = Field(default=None, description="예산 (원화)", ge=0)
    travel_style: str | None = Field(default=None, description="여행 스타일")
    companions: int | None = Field(default=None, description="동행자 수", ge=1)


def create_initial_state(
    user_input: str,
    max_iterations: int = 3
) -> TravelPlanningState:
    """초기 상태 생성"""
    return {
        "messages": [],
        "user_input": user_input,
        "final_response": "",
        "travel_plan": "",
        "plan_steps": [],
        "intent": "",
        "tool_results": [],
        "retrieved_context": "",
        "budget_info": "",
        "web_search_info": "",
        "quality_score": 0,
        "quality_feedback": "",
        "evaluation_passed": False,
        "iteration": 0,
        "max_iterations": max_iterations,
        "extracted_preferences": {},
        "user_profile": {},
        "skip_to_end": False,
        "error_log": [],
    }
