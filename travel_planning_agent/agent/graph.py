"""
여행 계획 에이전트 - 통합 워크플로우 그래프

Plan-and-Solve 파이프라인: classify_intent -> extract_preferences -> plan -> research -> synthesize -> evaluate -> improve (optional) -> save_memory
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
    """여행 계획 에이전트 그래프 생성 (Plan-and-Solve 파이프라인 + 평가/개선 루프)"""
    builder = StateGraph(TravelPlanningState)

    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("extract_preferences", extract_preferences_node)
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("evaluate_response", evaluate_response_node)
    builder.add_node("improve_response", improve_response_node)
    builder.add_node("save_memory", save_memory_node)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        lambda s: "skip" if s.get("skip_to_end") else "continue",
        {"continue": "extract_preferences", "skip": "save_memory"}
    )

    builder.add_edge("extract_preferences", "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "synthesize")
    builder.add_edge("synthesize", "evaluate_response")

    builder.add_conditional_edges(
        "evaluate_response",
        should_improve_response,
        {
            "improve": "improve_response",
            "end": "save_memory",
        }
    )

    builder.add_edge("improve_response", "evaluate_response")
    builder.add_edge("save_memory", END)

    if with_memory:
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)
    else:
        graph = builder.compile()

    return graph


def get_graph_mermaid(graph) -> str:
    """그래프를 Mermaid 다이어그램 코드로 변환"""
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        return f"시각화 실패: {e}"


def run_travel_planning(
    query: str,
    thread_id: str = "default",
    user_id: str = "anonymous",
) -> dict:
    """여행 계획 에이전트 실행"""
    graph = create_travel_planning_graph(with_memory=True)

    initial_state = create_initial_state(
        user_input=query,
        max_iterations=3,
    )

    if user_id in USER_PROFILES:
        initial_state["user_profile"] = USER_PROFILES[user_id]

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    result = graph.invoke(initial_state, config)

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("여행 계획 에이전트 워크플로우")
    print("=" * 60)

    graph = create_travel_planning_graph(with_memory=False)

    print("\n[워크플로우 구조 (Mermaid)]")
    print(get_graph_mermaid(graph))


# langgraph dev용: with_memory=False (LangGraph Studio 자체 persistence 사용)
graph = create_travel_planning_graph(with_memory=False)
