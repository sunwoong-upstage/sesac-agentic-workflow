"""
여행 계획 에이전트 - 통합 워크플로우 그래프

이 모듈은 모든 노드를 연결하여 완성된 워크플로우를 구성합니다.

워크플로우 구조 (Plan-and-Solve 파이프라인):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   START                                                         │
│     ↓                                                           │
│   [classify_intent] ─── 문의 유형 분류                          │
│     ↓ (skip_to_end?)                                           │
│   [extract_preferences] ─── 대화에서 선호도 추출 (다중 턴)       │
│     ↓                                                           │
│   [plan] ─── Plan 단계: 실행 계획 수립                          │
│     ↓                                                           │
│   [research] ─── Solve 단계: 도구 호출로 정보 수집              │
│     ↓                                                           │
│   [synthesize] ─── Synthesize 단계: 결과 종합                   │
│     ↓                                                           │
│   [evaluate_response] ─── 품질 평가                              │
│     ↓                                                           │
│   [improve?] ──→ [improve_response] ──→ [evaluate_response]     │
│     ↓                                                           │
│   [save_memory] ─── 이중 메모리 저장                            │
│     ↓                                                           │
│   END                                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

설계 결정 (ADR-1):
이 그래프는 의도적으로 선형 구조입니다. 의도(intent)에 따른 조건부 라우팅
대신, 모든 요청이 동일한 Plan-and-Solve 파이프라인을 거칩니다.
intent는 노드 내부에서 프롬프트/도구 선택에만 사용됩니다.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from typing import Literal

from agent.state import TravelPlanningState, create_initial_state
from agent.nodes import (
    classify_intent_node,
    extract_preferences_node,
    plan_node,
    research_node,
    synthesize_node,
    evaluate_response_node,
    improve_response_node,
    save_memory_node,
    should_improve_response,
    USER_PROFILES,
)


def create_travel_planning_graph(with_memory: bool = True):
    """
    여행 계획 에이전트의 통합 워크플로우 그래프를 생성합니다.

    이 그래프는 Plan-and-Solve 파이프라인을 구현합니다:
    1. 의도 분류
    2. 선호도 추출 (다중 턴 대화 지원)
    3. Plan: 실행 계획 수립
    4. Solve: 도구 호출로 정보 수집
    5. Synthesize: 결과 종합하여 응답 생성
    6. 응답 품질 평가 및 개선 루프
    7. 이중 메모리 저장

    Args:
        with_memory: 메모리(체크포인터) 사용 여부

    Returns:
        컴파일된 그래프 인스턴스

    Usage:
        graph = create_travel_planning_graph()
        result = graph.invoke(
            create_initial_state("제주도 3박 4일 여행 계획 세워줘"),
            {"configurable": {"thread_id": "user-123", "user_id": "user-001"}}
        )
        print(result["final_response"])
    """
    # =========================================================================
    # 1. StateGraph 생성
    # =========================================================================
    builder = StateGraph(TravelPlanningState)

    # =========================================================================
    # 2. 노드 추가
    # =========================================================================

    # Plan-and-Solve 파이프라인 노드
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("extract_preferences", extract_preferences_node)
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)

    # 평가-개선 루프 노드
    builder.add_node("evaluate_response", evaluate_response_node)
    builder.add_node("improve_response", improve_response_node)

    # 메모리 저장 노드
    builder.add_node("save_memory", save_memory_node)

    # =========================================================================
    # 3. 엣지 추가 (선형 파이프라인 + 평가 루프)
    # =========================================================================

    # START -> 의도 분류
    builder.add_edge(START, "classify_intent")

    # classify_intent 다음에 조건부 라우팅
    builder.add_conditional_edges(
        "classify_intent",
        lambda s: "skip" if s.get("skip_to_end") else "continue",
        {"continue": "extract_preferences", "skip": "save_memory"}
    )

    # Plan-and-Solve 파이프라인 (선형)
    builder.add_edge("extract_preferences", "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "synthesize")

    # 종합 -> 평가
    builder.add_edge("synthesize", "evaluate_response")

    # 평가 -> 개선 또는 저장 (조건부 분기)
    builder.add_conditional_edges(
        "evaluate_response",
        should_improve_response,
        {
            "improve": "improve_response",
            "end": "save_memory",
        }
    )

    # 응답 개선 -> 재평가 (루프)
    builder.add_edge("improve_response", "evaluate_response")

    # 메모리 저장 -> END
    builder.add_edge("save_memory", END)

    # =========================================================================
    # 4. 컴파일
    #
    # MemorySaver는 단기 메모리(Short-term Memory)를 담당합니다:
    # - thread_id 기반으로 대화 상태를 자동 저장/복원
    # - 같은 thread_id로 재호출하면 이전 대화 맥락 유지
    # =========================================================================
    if with_memory:
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)
    else:
        graph = builder.compile()

    return graph


# =============================================================================
# 그래프 시각화
# =============================================================================

def get_graph_mermaid(graph) -> str:
    """
    그래프를 Mermaid 다이어그램 코드로 변환합니다.

    Args:
        graph: 컴파일된 그래프

    Returns:
        Mermaid 다이어그램 코드
    """
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        return f"시각화 실패: {e}"


# =============================================================================
# 실행 헬퍼 함수
# =============================================================================

def run_travel_planning(
    query: str,
    thread_id: str = "default",
    user_id: str = "anonymous",
) -> dict:
    """
    여행 계획 에이전트를 실행합니다.

    Args:
        query: 사용자 문의 내용
        thread_id: 대화 스레드 ID (단기 메모리용)
        user_id: 사용자 ID (장기 메모리용)
        with_evaluation: 응답 품질 평가 사용 여부

    Returns:
        실행 결과 (상태 딕셔너리)

    Usage:
        result = run_travel_planning(
            "제주도 3박 4일 여행 계획 세워줘",
            thread_id="session-123",
            user_id="user-001"
        )
        print(result["final_response"])
    """
    # 그래프 생성
    graph = create_travel_planning_graph(with_memory=True)

    # 초기 상태 생성
    initial_state = create_initial_state(
        user_input=query,
        max_iterations=3,
    )

    # 장기 메모리에서 사용자 프로필 불러오기
    if user_id in USER_PROFILES:
        initial_state["user_profile"] = USER_PROFILES[user_id]

    # 설정 (단기 메모리용 thread_id + 장기 메모리용 user_id)
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    # 실행
    result = graph.invoke(initial_state, config)

    return result


# =============================================================================
# 테스트 코드
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("여행 계획 에이전트 워크플로우")
    print("=" * 60)

    graph = create_travel_planning_graph(with_memory=False)

    print("\n[워크플로우 구조 (Mermaid)]")
    print(get_graph_mermaid(graph))


# =============================================================================
# langgraph dev 용 그래프 인스턴스
#
# langgraph dev 명령어로 LangGraph Studio에서 대화형으로 테스트할 수 있습니다.
# 실행: langgraph dev
#
# 주의: with_memory=False로 생성합니다.
# langgraph dev는 자체 persistence(체크포인터)를 제공하므로,
# 여기서 MemorySaver를 포함하면 중복 에러가 발생합니다.
# main.py에서 직접 실행할 때는 run_travel_planning()이
# with_memory=True로 별도 그래프를 생성합니다.
# =============================================================================

graph = create_travel_planning_graph(with_memory=False)
