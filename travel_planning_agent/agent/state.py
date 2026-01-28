"""
여행 계획 에이전트 - 상태(State) 정의

이 모듈은 워크플로우 전체에서 공유되는 상태를 정의합니다.

통합된 기능:
- Plan-and-Solve 계획 상태 관리
- 도구 호출 결과 누적
- RAG 검색 결과 저장
- 이중 메모리 시스템 (단기/장기)
- 응답 품질 평가 루프 제어
"""

import operator
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


# =============================================================================
# 메인 에이전트 상태
# =============================================================================

class TravelPlanningState(TypedDict):
    """
    여행 계획 에이전트의 통합 상태

    워크플로우의 모든 노드가 이 상태를 공유합니다.
    Annotated[..., operator.add]는 값이 누적되는 필드입니다.
    """

    # =========================================================================
    # 대화 관련 (단기 메모리 - MemorySaver가 자동 관리)
    # =========================================================================
    messages: Annotated[List[BaseMessage], operator.add]
    """대화 메시지 히스토리 - LangChain 메시지 객체 리스트"""

    # =========================================================================
    # 입력/출력
    # =========================================================================
    user_input: str
    """사용자의 원본 입력"""

    final_response: str
    """최종 응답 (사용자에게 전달될 메시지)"""

    # =========================================================================
    # Plan-and-Solve 관련
    # =========================================================================
    travel_plan: str
    """Plan-and-Solve에서 생성된 계획 텍스트"""

    plan_steps: List[str]
    """분해된 계획 단계들 - research_node가 이를 읽어 도구 호출을 결정합니다"""

    # =========================================================================
    # 의도 분류 관련
    # =========================================================================
    intent: str
    """분류된 의도: destination_research, itinerary_planning, budget_estimation, general_travel"""

    # =========================================================================
    # 도구 호출 결과
    # =========================================================================
    tool_results: Annotated[List[dict], operator.add]
    """도구 호출 결과 누적 - 각 항목: {"tool": str, "args": dict, "result": str}"""

    # =========================================================================
    # RAG 및 조사 결과
    # =========================================================================
    retrieved_context: str
    """RAG 벡터 검색으로 가져온 여행 지식"""

    budget_info: str
    """예산 도구 조회 결과"""

    web_search_info: str
    """웹 검색(Serper API) 결과"""

    # =========================================================================
    # 응답 품질 평가 (Evaluator-Optimizer 패턴)
    # =========================================================================
    quality_score: int
    """응답 품질 점수 (1-10)"""

    quality_feedback: str
    """품질 평가 피드백"""

    evaluation_passed: bool
    """품질 기준 통과 여부 (7점 이상)"""

    # =========================================================================
    # 루프 제어
    # =========================================================================
    iteration: int
    """현재 반복 횟수 (평가-개선 루프)"""

    max_iterations: int
    """최대 반복 횟수"""

    # =========================================================================
    # 대화에서 추출된 선호도 (다중 턴 지원)
    # =========================================================================
    extracted_preferences: dict
    """
    대화 히스토리에서 추출한 구조화된 선호도
    예: {"destination": "도쿄", "budget": 300000, "duration_days": 3}
    """

    # =========================================================================
    # 장기 메모리 (사용자 프로필)
    # =========================================================================
    user_profile: dict
    """장기 메모리: 사용자 여행 선호도 및 이력"""

    # =========================================================================
    # 파이프라인 제어
    # =========================================================================
    skip_to_end: bool
    """빈 입력 시 파이프라인을 건너뛰기 위한 플래그"""

    # =========================================================================
    # 에러 로그
    # =========================================================================
    error_log: Annotated[List[str], operator.add]
    """에러 로그 누적 - 폴백 발생 시 기록"""


# =============================================================================
# 구조화된 출력 스키마 (Pydantic 모델)
# =============================================================================

class IntentClassification(BaseModel):
    """
    의도 분류 결과 스키마

    LLM이 사용자 쿼리를 분류할 때 반환하는 구조화된 출력입니다.
    """
    intent: Literal[
        "destination_research",
        "itinerary_planning",
        "budget_estimation",
        "general_travel"
    ] = Field(
        description="문의 유형: destination_research(여행지 조사), itinerary_planning(일정 계획), budget_estimation(예산 추정), general_travel(일반 여행 문의)"
    )


class TravelPlan(BaseModel):
    """
    여행 계획 스키마 (Plan-and-Solve의 Plan 단계 출력)

    LLM이 사용자 요청을 분석하여 생성하는 실행 계획입니다.
    steps 필드가 research_node에서 도구 호출 결정에 사용됩니다.
    """
    destination: str = Field(
        description="주요 여행지"
    )
    duration_days: int = Field(
        description="여행 기간 (일수)",
        ge=1
    )
    steps: List[str] = Field(
        description="실행 계획 단계 리스트 (예: ['여행지 정보 검색', '날씨 확인', '관광지 검색', '예산 추정'])"
    )


class QualityEvaluation(BaseModel):
    """
    응답 품질 평가 스키마
    """
    score: int = Field(
        description="품질 점수 (1-10)",
        ge=1,
        le=10
    )
    feedback: str = Field(
        description="개선 피드백"
    )
    passed: bool = Field(
        description="품질 기준 통과 여부 (7점 이상)"
    )


class ExtractedPreferences(BaseModel):
    """
    대화에서 추출한 여행 선호도 스키마
    
    다중 턴 대화에서 사용자가 언급한 정보를 누적 추출합니다.
    각 필드는 Optional이며, 언급되지 않은 정보는 None으로 유지됩니다.
    """
    destination: str | None = Field(
        default=None,
        description="여행지 (예: '제주도', '도쿄', '파리'). 언급되지 않으면 None"
    )
    duration_days: int | None = Field(
        default=None,
        description="여행 기간 (일수). 예: 2박3일 → 3. 언급되지 않으면 None",
        ge=1
    )
    budget: int | None = Field(
        default=None,
        description="예산 (원화). 예: 30만원 → 300000. 언급되지 않으면 None",
        ge=0
    )
    travel_style: str | None = Field(
        default=None,
        description="여행 스타일 (예: 'budget', 'moderate', 'luxury'). 언급되지 않으면 None"
    )
    companions: int | None = Field(
        default=None,
        description="동행자 수 (본인 포함). 예: 혼자 → 1, 가족 4명 → 4. 언급되지 않으면 None",
        ge=1
    )


# =============================================================================
# 상태 초기화 헬퍼
# =============================================================================

def create_initial_state(
    user_input: str,
    max_iterations: int = 3
) -> TravelPlanningState:
    """
    초기 상태를 생성합니다.

    Args:
        user_input: 사용자 입력
        max_iterations: 최대 반복 횟수 (평가-개선 루프)

    Returns:
        초기화된 TravelPlanningState
    """
    return {
        # 대화
        "messages": [],
        # 입출력
        "user_input": user_input,
        "final_response": "",
        # Plan-and-Solve
        "travel_plan": "",
        "plan_steps": [],
        # 의도 분류
        "intent": "",
        # 도구 결과
        "tool_results": [],
        # RAG 및 조사 결과
        "retrieved_context": "",
        "budget_info": "",
        "web_search_info": "",
        # 품질 평가
        "quality_score": 0,
        "quality_feedback": "",
        "evaluation_passed": False,
        # 루프 제어
        "iteration": 0,
        "max_iterations": max_iterations,
        # 추출된 선호도
        "extracted_preferences": {},
        # 장기 메모리
        "user_profile": {},
        # 파이프라인 제어
        "skip_to_end": False,
        # 에러
        "error_log": [],
    }
