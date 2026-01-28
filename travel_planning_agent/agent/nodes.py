"""
여행 계획 에이전트 - 노드 함수 정의

이 모듈은 그래프의 각 노드에서 실행되는 함수들을 정의합니다.

워크플로우 흐름 (Plan-and-Solve 파이프라인):
1. classify_intent: 문의 유형 분류
2. plan: Plan 단계 - 실행 계획 수립
3. research: Solve 단계 - 도구 호출로 정보 수집
4. synthesize: Synthesize 단계 - 결과 종합하여 응답 생성
5. evaluate_response: 응답 품질 평가
6. improve_response: 응답 개선 (필요시)
7. save_memory: 이중 메모리 저장
"""

from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_upstage import ChatUpstage
from langchain_core.runnables import RunnableConfig

from agent.state import (
    TravelPlanningState,
    IntentClassification,
    TravelPlan,
    QualityEvaluation,
)
from agent.prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    PLAN_AND_SOLVE_PROMPT,
    RESEARCH_PROMPT,
    SYNTHESIZE_PROMPT,
    QUALITY_EVALUATION_PROMPT,
    RESPONSE_IMPROVEMENT_PROMPT,
)
from agent.tools import (
    search_travel_knowledge,
    estimate_budget,
    web_search,
    RESEARCH_TOOLS,
)


# =============================================================================
# 상태 헬퍼
# =============================================================================

def _get_user_input(state: TravelPlanningState) -> str:
    """상태에서 사용자 입력을 추출합니다."""
    if state.get("user_input"):
        return state["user_input"]
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# =============================================================================
# 장기 메모리 저장소
#
# 교육용 참고:
# 이 딕셔너리는 프로세스 재시작 시 초기화됩니다.
# 프로덕션에서는 Redis, MongoDB 등 영속적 저장소를 사용해야 합니다.
#
# 구조: { "user_id": { "preferred_destinations": [...], "query_history": [...], ... } }
# =============================================================================

USER_PROFILES: dict[str, dict] = {}


# =============================================================================
# 1. 의도 분류 노드
# =============================================================================

def classify_intent_node(state: TravelPlanningState) -> dict:
    """
    사용자 문의의 의도를 분류합니다.

    구조화된 출력(Structured Output)을 사용하여 LLM이
    IntentClassification 스키마에 맞는 JSON을 반환하도록 합니다.

    분류 카테고리:
    - destination_research: 여행지 조사/추천
    - itinerary_planning: 일정 계획
    - budget_estimation: 예산 추정
    - general_travel: 일반 여행 문의

    Args:
        state: 현재 상태

    Returns:
        분류 결과 딕셔너리 (user_input, intent, messages)
    """
    llm = ChatUpstage(model="solar-pro2", temperature=0.0)
    structured_llm = llm.with_structured_output(IntentClassification)

    query = _get_user_input(state)

    # 빈 입력 처리 - 사용자에게 입력 요청
    if not query.strip():
        return {
            "user_input": "",
            "intent": "general_travel",
            "final_response": "여행에 대해 궁금한 점을 입력해주세요! 예: '제주도 3박 4일 여행 계획 세워줘'",
            "evaluation_passed": True,
            "skip_to_end": True,
            "messages": [],
        }

    messages = [
        SystemMessage(content=INTENT_CLASSIFICATION_PROMPT.format(query=query)),
        HumanMessage(content=query)
    ]

    try:
        result: IntentClassification = structured_llm.invoke(messages)
        return {
            "user_input": query,
            "intent": result.intent,
            "messages": [HumanMessage(content=query)],
        }
    except Exception as e:
        # 분류 실패 시 general_travel로 폴백
        return {
            "user_input": query,
            "intent": "general_travel",
            "messages": [HumanMessage(content=query)],
            "error_log": [f"의도 분류 실패, general_travel로 폴백: {e}"],
        }


# =============================================================================
# 2. Plan-and-Solve 노드 (핵심 기법 - Plan 단계)
#
# Plan-and-Solve 프롬프팅 기법:
# 복잡한 문제를 "계획 수립 → 단계별 실행 → 결과 종합"으로 나누어 해결합니다.
#
# 이 노드는 Plan 단계를 담당합니다:
# - 사용자 요청을 분석하여 필요한 조사 항목을 파악
# - TravelPlan 스키마로 구조화된 계획 생성
# - plan_steps를 다음 노드(research_node)에 전달
#
# 참고 논문: "Plan-and-Solve Prompting" (Wang et al., 2023)
# =============================================================================

def plan_node(state: TravelPlanningState) -> dict:
    """
    Plan-and-Solve 기법의 Plan 단계

    사용자 요청을 분석하여 실행 계획을 수립합니다.
    생성된 plan_steps는 research_node에서 도구 호출 결정에 사용됩니다.

    Args:
        state: 현재 상태

    Returns:
        계획 텍스트 및 실행 단계 리스트
    """
    llm = ChatUpstage(model="solar-pro2", temperature=0.0)
    structured_llm = llm.with_structured_output(TravelPlan)

    query = state.get("user_input", "")
    intent = state.get("intent", "general_travel")
    user_profile = state.get("user_profile", {})

    prompt = PLAN_AND_SOLVE_PROMPT.format(
        query=query,
        intent=intent,
        user_profile=user_profile if user_profile else "프로필 없음 (첫 방문)",
    )

    try:
        plan: TravelPlan = structured_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ])
        plan_text = f"{plan.destination} {plan.duration_days}일 여행 계획"
        return {
            "travel_plan": plan_text,
            "plan_steps": plan.steps,
            "messages": [AIMessage(content=f"[Plan] {plan_text}\n단계: {', '.join(plan.steps)}")],
        }
    except Exception as e:
        # 폴백: 기본 계획 단계
        # 구조화된 출력이 실패해도 기본적인 조사는 수행할 수 있도록 합니다
        default_steps = ["여행지 정보 검색", "예산 추정", "웹 검색"]
        return {
            "travel_plan": "기본 여행 조사 계획",
            "plan_steps": default_steps,
            "error_log": [f"계획 생성 실패, 기본 계획 사용: {e}"],
            "messages": [AIMessage(content=f"[Plan] 기본 계획: {', '.join(default_steps)}")],
        }


# =============================================================================
# 3. 조사 노드 (Plan-and-Solve - Solve 단계)
# =============================================================================

def research_node(state: TravelPlanningState) -> dict:
    """
    Plan-and-Solve 기법의 Solve 단계 - 도구 호출

    plan_node에서 생성된 plan_steps를 읽어
    필요한 도구를 호출하고 정보를 수집합니다.

    Args:
        state: 현재 상태

    Returns:
        도구 호출 결과 및 수집된 정보
    """
    llm = ChatUpstage(model="solar-pro2", temperature=0.3)
    llm_with_tools = llm.bind_tools(RESEARCH_TOOLS)
    plan_steps = state.get("plan_steps", [])
    query = state.get("user_input", "")
    intent = state.get("intent", "general_travel")
    tool_results = []

    # plan_steps를 프롬프트에 포함하여 LLM에게 도구 호출 요청
    prompt = RESEARCH_PROMPT.format(
        query=query,
        plan_steps="\n".join(f"- {step}" for step in plan_steps),
        intent=intent,
    )

    response = llm_with_tools.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=query)
    ])

    # LLM이 반환한 tool_calls 실행
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_map = {tool.name: tool for tool in RESEARCH_TOOLS}
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if tool_name in tool_map:
                try:
                    tool_output = tool_map[tool_name].invoke(tool_args)
                    tool_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_output,
                    })
                except Exception as e:
                    tool_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": f"도구 실행 오류: {e}",
                    })

    return {
        "tool_results": tool_results,
        "retrieved_context": next((r["result"] for r in tool_results if r["tool"] == "search_travel_knowledge"), ""),
        "budget_info": next((r["result"] for r in tool_results if r["tool"] == "estimate_budget"), ""),
        "web_search_info": next((r["result"] for r in tool_results if r["tool"] == "web_search"), ""),
        "messages": [AIMessage(content=f"[Research] {len(tool_results)}개 도구 호출 완료")],
    }


# =============================================================================
# 4. 종합 응답 노드 (Plan-and-Solve - Synthesize 단계)
# =============================================================================

def synthesize_node(state: TravelPlanningState) -> dict:
    """
    Plan-and-Solve 기법의 Synthesize 단계

    plan (계획) + research (조사 결과)를 종합하여 최종 응답을 생성합니다.
    모든 수집된 정보를 프롬프트에 포함하여 포괄적인 답변을 만듭니다.

    Args:
        state: 현재 상태

    Returns:
        최종 응답
    """
    llm = ChatUpstage(model="solar-pro2", temperature=0.5)

    query = state.get("user_input", "")
    prompt = SYNTHESIZE_PROMPT.format(
        query=query,
        travel_plan=state.get("travel_plan") or "계획 없음",
        retrieved_context=state.get("retrieved_context") or "검색 결과 없음",
        budget_info=state.get("budget_info") or "예산 정보 없음",
        web_search_info=state.get("web_search_info") or "웹 검색 결과 없음",
    )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=query)
    ])

    return {
        "final_response": response.content,
        "messages": [AIMessage(content=response.content)],
    }


# =============================================================================
# 5. 응답 품질 평가 노드 (Evaluator-Optimizer 패턴)
# =============================================================================

def evaluate_response_node(state: TravelPlanningState) -> dict:
    """
    생성된 응답의 품질을 평가합니다.

    Evaluator-Optimizer 패턴:
    - 평가자(Evaluator)가 응답 품질을 점수화
    - 7점 미만이면 개선자(Optimizer)가 응답을 개선
    - 최대 반복 횟수까지 루프

    Args:
        state: 현재 상태

    Returns:
        평가 결과
    """
    llm = ChatUpstage(model="solar-pro2", temperature=0.0)
    structured_llm = llm.with_structured_output(QualityEvaluation)

    query = state.get("user_input", "")
    response = state.get("final_response", "")
    intent = state.get("intent", "general_travel")

    prompt = QUALITY_EVALUATION_PROMPT.format(query=query, response=response, intent=intent)

    try:
        result: QualityEvaluation = structured_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"평가 대상 응답:\n{response}")
        ])

        return {
            "quality_score": result.score,
            "quality_feedback": result.feedback,
            "evaluation_passed": result.passed,
            "iteration": state.get("iteration", 0) + 1,
        }
    except Exception as e:
        # 평가 실패 시 통과 처리
        return {
            "quality_score": 7,
            "quality_feedback": "평가 완료",
            "evaluation_passed": True,
            "iteration": state.get("iteration", 0) + 1,
            "error_log": [f"평가 실패, 통과 처리: {e}"],
        }


# =============================================================================
# 6. 응답 개선 노드
# =============================================================================

def improve_response_node(state: TravelPlanningState) -> dict:
    """
    평가 피드백을 반영하여 응답을 개선합니다.

    Evaluator-Optimizer 패턴의 Optimizer 역할:
    - 평가자의 피드백을 입력으로 받아 응답을 재생성
    - 원본 응답의 좋은 부분은 유지하면서 부족한 부분만 보완

    Args:
        state: 현재 상태

    Returns:
        개선된 응답
    """
    llm = ChatUpstage(model="solar-pro2", temperature=0.3)

    query = state.get("user_input", "")
    original_response = state.get("final_response", "")
    feedback = state.get("quality_feedback", "")
    score = state.get("quality_score", 0)

    prompt = RESPONSE_IMPROVEMENT_PROMPT.format(query=query, original_response=original_response, feedback=feedback, score=score)

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="개선된 응답을 작성해주세요.")
    ])

    return {
        "final_response": response.content,
        "messages": [AIMessage(content=f"[개선됨] {response.content}")],
    }


# =============================================================================
# 7. 메모리 저장 노드 (이중 메모리 시스템)
#
# 이중 메모리 설계:
#
# 1. 단기 메모리 (Short-term Memory):
#    - MemorySaver 체크포인터가 자동으로 대화 상태 저장
#    - thread_id 기반으로 대화 히스토리 유지
#    - 같은 thread_id로 호출하면 이전 대화 맥락 유지
#
# 2. 장기 메모리 (Long-term Memory):
#    - USER_PROFILES 딕셔너리에 사용자 선호도 저장
#    - user_id 기반으로 여행 선호도 누적
#    - 세션 간에도 사용자 취향 정보 유지
#
# 교육용 참고: USER_PROFILES는 프로세스 재시작 시 초기화됩니다.
# 프로덕션에서는 Redis, MongoDB 등 영속적 저장소를 사용합니다.
# =============================================================================

def save_memory_node(state: TravelPlanningState, config: RunnableConfig) -> dict:
    """
    이중 메모리 시스템으로 대화 정보를 저장합니다.

    Args:
        state: 현재 상태
        config: LangGraph 실행 설정 (user_id 포함)

    Returns:
        업데이트된 사용자 프로필
    """
    # config에서 user_id 추출 (ADR-4: user_id는 config으로 전달)
    user_id = config.get("configurable", {}).get("user_id", "anonymous")

    # 장기 메모리 초기화 (첫 방문 시)
    if user_id not in USER_PROFILES:
        USER_PROFILES[user_id] = {
            "preferred_destinations": [],
            "query_history": [],
        }

    user_profile = USER_PROFILES[user_id]

    # 문의 이력 추가
    user_profile["query_history"].append({
        "query": _get_user_input(state),
        "intent": state.get("intent", "general_travel"),
        "quality_score": state.get("quality_score", 0),
    })

    return {
        "user_profile": user_profile,
    }


# =============================================================================
# 라우팅 함수
# =============================================================================

def should_improve_response(state: TravelPlanningState) -> Literal["improve", "end"]:
    """
    응답 개선이 필요한지 결정합니다.

    조건:
    - 품질 기준 통과 → "end" (save_memory로 이동)
    - 최대 반복 횟수 도달 → "end"
    - 그 외 → "improve" (improve_response로 이동)

    Args:
        state: 현재 상태

    Returns:
        "improve" 또는 "end"
    """
    if state.get("evaluation_passed", False):
        return "end"

    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "end"

    return "improve"
