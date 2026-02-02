"""뉴스 챗봇 LangGraph 워크플로우"""
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from .state import NewsChatbotState, create_initial_state
from .nodes import (
    classify_intent_node,
    extract_preferences_node,
    plan_node,
    research_node,
    synthesize_node,
    evaluate_response_node,
    improve_response_node,
    save_memory_node,
    should_improve_response,
)

logger = logging.getLogger(__name__)

# 장기 메모리 저장소 (모듈 레벨)
user_store = InMemoryStore()


def create_news_chatbot_graph(with_memory: bool = True):
    """뉴스 챗봇 그래프 생성
    
    Args:
        with_memory: 메모리 사용 여부 (False: LangGraph Studio용)
        
    Returns:
        컴파일된 LangGraph
    """
    logger.info(f"[Graph] 그래프 생성 시작 (with_memory={with_memory})")
    
    # 그래프 빌더 생성
    builder = StateGraph(NewsChatbotState)
    
    # 노드 추가
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("extract_preferences", extract_preferences_node)
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("evaluate", evaluate_response_node)
    builder.add_node("improve", improve_response_node)
    builder.add_node("save_memory", save_memory_node)
    
    # 엣지 추가 (순차 실행)
    builder.set_entry_point("classify_intent")
    builder.add_edge("classify_intent", "extract_preferences")
    builder.add_edge("extract_preferences", "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "synthesize")
    builder.add_edge("synthesize", "evaluate")
    
    # 조건부 엣지 (평가 → 개선 or 종료)
    builder.add_conditional_edges(
        "evaluate",
        should_improve_response,
        {
            "improve": "improve",
            "end": "save_memory",
        }
    )
    builder.add_edge("improve", "evaluate")
    builder.add_edge("save_memory", END)
    
    # 컴파일
    if with_memory:
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory, store=user_store)
        logger.info("[Graph] 메모리 포함 그래프 컴파일 완료")
    else:
        graph = builder.compile()
        logger.info("[Graph] 메모리 없이 그래프 컴파일 완료")
    
    return graph


def run_news_chatbot(
    query: str,
    thread_id: str = "default",
    user_id: str = "anonymous",
) -> dict:
    """뉴스 챗봇 실행
    
    Args:
        query: 사용자 질문
        thread_id: 대화 스레드 ID
        user_id: 사용자 ID
        
    Returns:
        최종 상태
    """
    logger.info(f"[Run] 챗봇 실행 시작 | thread={thread_id}, user={user_id}")
    logger.info(f"[Run] 질문: {query}")
    
    graph = create_news_chatbot_graph(with_memory=True)
    
    # 초기 상태 생성
    initial_state = create_initial_state(query)
    
    # 장기 메모리에서 사용자 프로필 로드
    profile_item = user_store.get(("users",), user_id)
    if profile_item:
        initial_state["user_profile"] = profile_item.value
        logger.info(f"[Run] 기존 사용자 프로필 로드됨")
    
    # 설정
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }
    
    # 실행
    result = graph.invoke(initial_state, config)
    
    # 장기 메모리에 사용자 프로필 저장
    if result.get("user_profile"):
        user_store.put(("users",), user_id, result["user_profile"])
        logger.info(f"[Run] 사용자 프로필 저장됨")
    
    logger.info(f"[Run] 챗봇 실행 완료")
    return result


# LangGraph Studio용 그래프 (메모리 없이)
graph = create_news_chatbot_graph(with_memory=False)
