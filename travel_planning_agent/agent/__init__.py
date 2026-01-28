"""
여행 계획 에이전트 패키지

이 패키지는 LangGraph 기반의 여행 계획 AI 에이전트를 제공합니다.

적용된 기술:
- Plan-and-Solve 프롬프팅 기법
- FAISS 벡터 스토어 기반 Agentic RAG
- 도구 호출 (RAG, 예산 추정, 웹 검색 + 방어적 폴백)
- 이중 메모리 시스템 (단기: MemorySaver / 장기: USER_PROFILES)
- Evaluator-Optimizer 패턴 (응답 품질 평가 및 개선)

사용 예시:
    from agent import run_travel_planning

    result = run_travel_planning(
        query="제주도 3박 4일 여행 계획 세워줘",
        user_id="user-001"
    )
    print(result["final_response"])
"""

from agent.state import (
    TravelPlanningState,
    create_initial_state,
    IntentClassification,
    TravelPlan,
    QualityEvaluation,
    ExtractedPreferences,
)
from agent.graph import (
    create_travel_planning_graph,
    run_travel_planning,
    get_graph_mermaid,
)
from agent.tools import (
    search_travel_knowledge,
    estimate_budget,
    web_search,
)

__all__ = [
    # State
    "TravelPlanningState",
    "create_initial_state",
    "IntentClassification",
    "TravelPlan",
    "QualityEvaluation",
    "ExtractedPreferences",
    # Graph
    "create_travel_planning_graph",
    "run_travel_planning",
    "get_graph_mermaid",
    # Tools
    "search_travel_knowledge",
    "estimate_budget",
    "web_search",
]

__version__ = "1.0.0"
