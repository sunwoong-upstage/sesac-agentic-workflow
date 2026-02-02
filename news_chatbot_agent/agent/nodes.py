"""뉴스 챗봇 워크플로우 노드"""
import logging
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_upstage import ChatUpstage

from .state import (
    NewsChatbotState,
    IntentClassification,
    NewsPreferences,
    QualityEvaluation,
)
from .prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    PREFERENCE_EXTRACTION_PROMPT,
    PLAN_PROMPT,
    RESEARCH_PROMPT,
    SYNTHESIS_PROMPT,
    EVALUATION_PROMPT,
    IMPROVEMENT_PROMPT,
)
from .tools import TOOLS

logger = logging.getLogger(__name__)

# LLM 초기화
llm = ChatUpstage(model="solar-pro2")


def classify_intent_node(state: NewsChatbotState) -> dict:
    """사용자 의도 분류 노드"""
    logger.info("[Node] classify_intent_node 시작")
    
    user_input = state["user_input"]
    messages = state.get("messages", [])
    
    # 대화 이력 포맷팅
    conversation_history = ""
    for msg in messages[-6:]:  # 최근 6개 메시지
        role = "사용자" if isinstance(msg, HumanMessage) else "어시스턴트"
        conversation_history += f"{role}: {msg.content}\n"
    
    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        user_input=user_input,
        conversation_history=conversation_history or "없음"
    )
    
    structured_llm = llm.with_structured_output(IntentClassification)
    result = structured_llm.invoke([SystemMessage(content=prompt)])
    
    logger.info(f"[Node] 의도 분류 결과: {result.intent} (신뢰도: {result.confidence})")
    
    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "messages": [HumanMessage(content=user_input)],
    }


def extract_preferences_node(state: NewsChatbotState) -> dict:
    """뉴스 선호도 추출 노드"""
    logger.info("[Node] extract_preferences_node 시작")
    
    user_input = state["user_input"]
    previous_topics = state.get("topics", [])
    previous_keywords = state.get("keywords", [])
    
    prompt = PREFERENCE_EXTRACTION_PROMPT.format(
        user_input=user_input,
        previous_topics=", ".join(previous_topics) or "없음",
        previous_keywords=", ".join(previous_keywords) or "없음",
    )
    
    structured_llm = llm.with_structured_output(NewsPreferences)
    result = structured_llm.invoke([SystemMessage(content=prompt)])
    
    # 이전 선호도와 병합
    new_topics = list(set(previous_topics + result.topics))
    new_keywords = list(set(previous_keywords + result.keywords))
    
    logger.info(f"[Node] 추출된 주제: {result.topics}, 키워드: {result.keywords}")
    
    return {
        "topics": new_topics,
        "keywords": new_keywords,
        "date_range": result.date_range.model_dump() if result.date_range else None,
    }


def plan_node(state: NewsChatbotState) -> dict:
    """실행 계획 수립 노드 (Plan-and-Solve: Plan)"""
    logger.info("[Node] plan_node 시작")
    
    prompt = PLAN_PROMPT.format(
        user_input=state["user_input"],
        intent=state["intent"],
        topics=", ".join(state.get("topics", [])) or "없음",
        keywords=", ".join(state.get("keywords", [])) or "없음",
        date_range=state.get("date_range") or "지정 안됨",
    )
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    logger.info(f"[Node] 실행 계획 수립 완료")
    logger.debug(f"[Node] 계획 내용: {response.content[:200]}...")
    
    return {"execution_plan": response.content}


def research_node(state: NewsChatbotState) -> dict:
    """정보 수집 노드 (Plan-and-Solve: Solve)"""
    logger.info("[Node] research_node 시작")
    
    prompt = RESEARCH_PROMPT.format(execution_plan=state["execution_plan"])
    
    # 도구 바인딩
    llm_with_tools = llm.bind_tools(TOOLS)
    
    messages = [SystemMessage(content=prompt)]
    tool_results = []
    
    # 도구 호출 루프 (최대 3회)
    for i in range(3):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            break
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"[Node] 도구 호출: {tool_name}")
            
            # 도구 실행
            tool_fn = next((t for t in TOOLS if t.name == tool_name), None)
            if tool_fn:
                result = tool_fn.invoke(tool_args)
                tool_results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                })
                
                # 도구 결과를 메시지에 추가
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    
    logger.info(f"[Node] 정보 수집 완료 (도구 호출 {len(tool_results)}회)")
    
    return {"tool_results": tool_results}


def synthesize_node(state: NewsChatbotState) -> dict:
    """응답 합성 노드 (Plan-and-Solve: Synthesize)"""
    logger.info("[Node] synthesize_node 시작")
    
    # 도구 결과 포맷팅
    tool_results_text = ""
    for result in state.get("tool_results", []):
        tool_results_text += f"\n[{result['tool']}]\n{result['result']}\n"
    
    prompt = SYNTHESIS_PROMPT.format(
        user_input=state["user_input"],
        tool_results=tool_results_text or "수집된 정보 없음",
    )
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    logger.info("[Node] 응답 합성 완료")
    
    return {
        "final_response": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def evaluate_response_node(state: NewsChatbotState) -> dict:
    """응답 품질 평가 노드"""
    logger.info("[Node] evaluate_response_node 시작")
    
    prompt = EVALUATION_PROMPT.format(
        user_input=state["user_input"],
        response=state["final_response"],
    )
    
    structured_llm = llm.with_structured_output(QualityEvaluation)
    result = structured_llm.invoke([SystemMessage(content=prompt)])
    
    logger.info(f"[Node] 품질 평가: {result.score}/10 (통과: {result.passed})")
    
    return {
        "quality_score": result.score,
        "quality_feedback": result.feedback,
        "evaluation_passed": result.passed,
        "iteration": state["iteration"] + 1,
    }


def improve_response_node(state: NewsChatbotState) -> dict:
    """응답 개선 노드"""
    logger.info("[Node] improve_response_node 시작")
    
    prompt = IMPROVEMENT_PROMPT.format(
        user_input=state["user_input"],
        previous_response=state["final_response"],
        feedback=state["quality_feedback"],
    )
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    logger.info("[Node] 응답 개선 완료")
    
    return {
        "final_response": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def save_memory_node(state: NewsChatbotState) -> dict:
    """사용자 프로필 저장 노드"""
    logger.info("[Node] save_memory_node 시작")
    
    # 기존 프로필 가져오기
    profile = state.get("user_profile", {})
    
    # 검색 이력 업데이트
    query_history = profile.get("query_history", [])
    query_history.append({
        "query": state["user_input"],
        "intent": state["intent"],
        "topics": state.get("topics", []),
        "keywords": state.get("keywords", []),
    })
    
    # 최근 10개만 유지
    profile["query_history"] = query_history[-10:]
    
    # 관심 주제 누적
    all_topics = profile.get("interests", [])
    all_topics.extend(state.get("topics", []))
    profile["interests"] = list(set(all_topics))[-20:]  # 최대 20개
    
    logger.info(f"[Node] 프로필 업데이트 완료 (검색 이력: {len(profile['query_history'])}개)")
    
    return {"user_profile": profile}


# ============================================
# 조건부 라우팅 함수
# ============================================

def should_improve_response(state: NewsChatbotState) -> Literal["improve", "end"]:
    """응답 개선 필요 여부 판단"""
    if state["evaluation_passed"]:
        logger.info("[Router] 평가 통과 → 종료")
        return "end"
    
    if state["iteration"] >= state["max_iterations"]:
        logger.info("[Router] 최대 반복 도달 → 종료")
        return "end"
    
    logger.info("[Router] 평가 미통과 → 개선 필요")
    return "improve"
