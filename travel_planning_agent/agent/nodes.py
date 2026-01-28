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

import logging
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_upstage import ChatUpstage
from langchain_core.runnables import RunnableConfig

# 로거 설정
logger = logging.getLogger(__name__)

from agent.state import (
    TravelPlanningState,
    IntentClassification,
    TravelPlan,
    QualityEvaluation,
    ExtractedPreferences,
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


USER_PROFILES: dict[str, dict] = {}


def classify_intent_node(state: TravelPlanningState) -> dict:
    """
    Structured Output: 의도 분류
    사용자 문의를 4가지 카테고리로 분류합니다.
    """
    logger.info("[Node] classify_intent 시작")
    llm = ChatUpstage(model="solar-pro2", temperature=0.0)
    structured_llm = llm.with_structured_output(IntentClassification)

    query = _get_user_input(state)
    logger.debug(f"사용자 입력: '{query}'")

    # 빈 입력 처리 - 사용자에게 입력 요청
    if not query.strip():
        logger.warning("빈 입력 감지, skip_to_end=True")
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
        logger.info(f"의도 분류 결과: {result.intent}")
        return {
            "user_input": query,
            "intent": result.intent,
            "messages": [HumanMessage(content=query)],
        }
    except Exception as e:
        # 분류 실패 시 general_travel로 폴백
        logger.error(f"의도 분류 실패, general_travel로 폴백: {e}")
        return {
            "user_input": query,
            "intent": "general_travel",
            "messages": [HumanMessage(content=query)],
            "error_log": [f"의도 분류 실패, general_travel로 폴백: {e}"],
        }


def extract_preferences_node(state: TravelPlanningState) -> dict:
    """
    다중 턴 대화: 선호도 추출 및 누적
    대화 히스토리에서 여행지, 예산, 기간 등을 추출합니다.
    """
    logger.info("[Node] extract_preferences 시작")
    llm = ChatUpstage(model="solar-pro2", temperature=0.0)
    structured_llm = llm.with_structured_output(ExtractedPreferences)
    
    # 전체 대화 히스토리 가져오기
    messages = state.get("messages", [])
    if not messages:
        logger.debug("대화 메시지 없음, 선호도 추출 스킵")
        return {"extracted_preferences": {}}
    
    # 추출 프롬프트
    extraction_prompt = """당신은 대화에서 여행 선호도를 추출하는 전문가입니다.

아래 대화 내용을 분석하여 사용자가 **명시적으로 언급한** 여행 정보만 추출하세요.

추출 규칙:
1. 명확히 언급된 정보만 추출합니다.
2. 추론이나 가정을 하지 마세요.
3. 언급되지 않은 정보는 None으로 유지합니다.
4. 이전 대화와 최신 대화를 모두 고려하여 누적 정보를 파악합니다.

예시:
- "도쿄 가고 싶어" → destination: "도쿄"
- "예산 30만원" → budget: 300000
- "2박3일" → duration_days: 3
- "가족 4명이서" → companions: 4

대화 내용을 분석하여 추출하세요."""
    
    try:
        result: ExtractedPreferences = structured_llm.invoke([
            SystemMessage(content=extraction_prompt),
            *messages  # 전체 대화 히스토리 전달
        ])
        
        # 기존 선호도와 병합 (새로운 정보가 우선)
        current_prefs = state.get("extracted_preferences", {})
        new_prefs = result.model_dump(exclude_none=True)  # None 값은 제외
        
        merged_prefs = {**current_prefs, **new_prefs}
        logger.info(f"선호도 추출 완료: {list(new_prefs.keys())}")
        logger.debug(f"누적 선호도: {merged_prefs}")
        
        return {
            "extracted_preferences": merged_prefs,
            "messages": [AIMessage(content=f"[선호도 추출] {list(new_prefs.keys())} 정보 갱신")],
        }
        
    except Exception as e:
        logger.error(f"선호도 추출 실패: {e}")
        return {
            "extracted_preferences": state.get("extracted_preferences", {}),
            "error_log": [f"선호도 추출 실패: {e}"],
        }


def plan_node(state: TravelPlanningState) -> dict:
    """
    Plan-and-Solve: Plan 단계
    사용자 요청을 분석하여 실행 계획을 수립합니다.
    """
    logger.info("[Node] plan 시작")
    llm = ChatUpstage(model="solar-pro2", temperature=0.0)
    structured_llm = llm.with_structured_output(TravelPlan)

    query = state.get("user_input", "")
    intent = state.get("intent", "general_travel")
    user_profile = state.get("user_profile", {})
    preferences = state.get("extracted_preferences", {})
    logger.debug(f"의도: {intent}, 선호도: {preferences}")

    # 추출된 선호도 정보를 프롬프트에 포함
    prefs_text = "없음"
    if preferences:
        prefs_text = ", ".join(f"{k}: {v}" for k, v in preferences.items() if v is not None)

    prompt = PLAN_AND_SOLVE_PROMPT.format(
        query=query,
        intent=intent,
        user_profile=f"선호도: {prefs_text}\n프로필: {user_profile if user_profile else '첫 방문'}",
    )

    try:
        # 전체 대화 히스토리를 포함하여 LLM에 전달
        messages = [SystemMessage(content=prompt)] + state.get("messages", [])
        plan: TravelPlan = structured_llm.invoke(messages)
        plan_text = f"{plan.destination} {plan.duration_days}일 여행 계획"
        logger.info(f"계획 수립 완료: {plan_text}")
        logger.debug(f"계획 단계: {plan.steps}")
        return {
            "travel_plan": plan_text,
            "plan_steps": plan.steps,
            "messages": [AIMessage(content=f"[Plan] {plan_text}\n단계: {', '.join(plan.steps)}")],
        }
    except Exception as e:
        # 폴백: 기본 계획 단계
        # 구조화된 출력이 실패해도 기본적인 조사는 수행할 수 있도록 합니다
        logger.error(f"계획 생성 실패, 기본 계획 사용: {e}")
        default_steps = ["여행지 정보 검색", "예산 추정", "웹 검색"]
        return {
            "travel_plan": "기본 여행 조사 계획",
            "plan_steps": default_steps,
            "error_log": [f"계획 생성 실패, 기본 계획 사용: {e}"],
            "messages": [AIMessage(content=f"[Plan] 기본 계획: {', '.join(default_steps)}")],
        }


def research_node(state: TravelPlanningState) -> dict:
    """
    Plan-and-Solve: Solve 단계
    plan_steps에 따라 도구를 호출하고 정보를 수집합니다.
    """
    logger.info("[Node] research 시작")
    llm = ChatUpstage(model="solar-pro2", temperature=0.3)
    llm_with_tools = llm.bind_tools(RESEARCH_TOOLS)
    plan_steps = state.get("plan_steps", [])
    query = state.get("user_input", "")
    intent = state.get("intent", "general_travel")
    preferences = state.get("extracted_preferences", {})
    tool_results = []
    logger.debug(f"계획 단계: {plan_steps}")

    # plan_steps와 추출된 선호도를 프롬프트에 포함
    from agent.tools import BUDGET_DB
    budget_destinations = ", ".join(BUDGET_DB.keys())
    
    prefs_text = "없음"
    if preferences:
        prefs_text = "\n".join(f"  - {k}: {v}" for k, v in preferences.items() if v is not None)
    
    prompt = RESEARCH_PROMPT.format(
        query=query,
        plan_steps="\n".join(f"- {step}" for step in plan_steps),
        intent=intent,
        budget_destinations=budget_destinations,
    )
    
    prompt += f"\n\n### 추출된 사용자 선호도\n{prefs_text}\n\n도구 호출 시 이 정보를 활용하세요."

    # 전체 대화 히스토리를 포함
    messages = [SystemMessage(content=prompt)] + state.get("messages", [])
    response = llm_with_tools.invoke(messages)

    # LLM이 반환한 tool_calls 실행
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"LLM이 {len(response.tool_calls)}개 도구 호출 요청")
        tool_map = {tool.name: tool for tool in RESEARCH_TOOLS}
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            logger.info(f"도구 실행 시작: {tool_name} | args={tool_args}")
            if tool_name in tool_map:
                try:
                    tool_output = tool_map[tool_name].invoke(tool_args)
                    tool_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_output,
                    })
                    logger.debug(f"도구 실행 성공: {tool_name}")
                except Exception as e:
                    logger.error(f"도구 실행 실패: {tool_name} - {e}")
                    tool_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": f"도구 실행 오류: {e}",
                    })
    else:
        logger.warning("LLM이 도구 호출을 요청하지 않음")

    logger.info(f"조사 완료: {len(tool_results)}개 도구 실행됨")
    return {
        "tool_results": tool_results,
        "retrieved_context": next((r["result"] for r in tool_results if r["tool"] == "search_travel_knowledge"), ""),
        "budget_info": next((r["result"] for r in tool_results if r["tool"] == "estimate_budget"), ""),
        "web_search_info": next((r["result"] for r in tool_results if r["tool"] == "web_search"), ""),
        "messages": [AIMessage(content=f"[Research] {len(tool_results)}개 도구 호출 완료")],
    }


def synthesize_node(state: TravelPlanningState) -> dict:
    """
    Plan-and-Solve: Synthesize 단계
    계획과 조사 결과를 종합하여 최종 응답을 생성합니다.
    """
    logger.info("[Node] synthesize 시작")
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

    logger.info(f"응답 생성 완료: {len(response.content)}자")
    return {
        "final_response": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def evaluate_response_node(state: TravelPlanningState) -> dict:
    """
    Evaluator-Optimizer: 응답 품질 평가
    7점 미만이면 improve_response_node로 이동합니다.
    """
    logger.info("[Node] evaluate_response 시작")
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

        logger.info(f"품질 평가 완료: {result.score}/10점, 통과={result.passed}")
        logger.debug(f"피드백: {result.feedback}")
        return {
            "quality_score": result.score,
            "quality_feedback": result.feedback,
            "evaluation_passed": result.passed,
            "iteration": state.get("iteration", 0) + 1,
        }
    except Exception as e:
        # 평가 실패 시 통과 처리
        logger.error(f"평가 실패, 통과 처리: {e}")
        return {
            "quality_score": 7,
            "quality_feedback": "평가 완료",
            "evaluation_passed": True,
            "iteration": state.get("iteration", 0) + 1,
            "error_log": [f"평가 실패, 통과 처리: {e}"],
        }


def improve_response_node(state: TravelPlanningState) -> dict:
    """
    Evaluator-Optimizer: 응답 개선
    평가 피드백을 반영하여 응답을 재생성합니다.
    """
    logger.info("[Node] improve_response 시작")
    llm = ChatUpstage(model="solar-pro2", temperature=0.3)

    query = state.get("user_input", "")
    original_response = state.get("final_response", "")
    feedback = state.get("quality_feedback", "")
    score = state.get("quality_score", 0)
    logger.debug(f"개선 피드백 (점수={score}): {feedback}")

    prompt = RESPONSE_IMPROVEMENT_PROMPT.format(query=query, original_response=original_response, feedback=feedback, score=score)

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="개선된 응답을 작성해주세요.")
    ])

    logger.info(f"응답 개선 완료: {len(response.content)}자")
    return {
        "final_response": response.content,
        "messages": [AIMessage(content=f"[개선됨] {response.content}")],
    }


def save_memory_node(state: TravelPlanningState, config: RunnableConfig) -> dict:
    """
    이중 메모리: 단기(MemorySaver) + 장기(USER_PROFILES)
    사용자 선호도와 문의 이력을 저장합니다.
    """
    logger.info("[Node] save_memory 시작")
    # config에서 user_id 추출 (ADR-4: user_id는 config으로 전달)
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    logger.debug(f"사용자 ID: {user_id}")

    # 장기 메모리 초기화 (첫 방문 시)
    if user_id not in USER_PROFILES:
        logger.info(f"신규 사용자 프로필 생성: {user_id}")
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
    logger.debug(f"문의 이력 추가: 총 {len(user_profile['query_history'])}건")

    # 선호 여행지 업데이트 (extracted_preferences에서 추출)
    extracted_prefs = state.get("extracted_preferences", {})
    destination = extracted_prefs.get("destination")
    if destination and destination not in user_profile["preferred_destinations"]:
        user_profile["preferred_destinations"].append(destination)
        logger.info(f"선호 여행지 추가: {destination}")

    logger.info(f"메모리 저장 완료 | 선호지: {user_profile['preferred_destinations']}, 이력: {len(user_profile['query_history'])}건")
    return {
        "user_profile": user_profile,
    }


def should_improve_response(state: TravelPlanningState) -> Literal["improve", "end"]:
    """
    품질 기준(7점) 통과 또는 최대 반복 도달 시 "end", 아니면 "improve"
    """
    if state.get("evaluation_passed", False):
        return "end"

    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "end"

    return "improve"
