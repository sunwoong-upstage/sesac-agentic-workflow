"""여행 계획 에이전트 데모 실행 스크립트"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def print_section(title: str, char: str = "=", width: int = 60):
    """섹션 헤더를 출력합니다."""
    print("\n" + char * width)
    print(title)
    print(char * width)


def main():
    """여행 계획 에이전트를 실행합니다."""

    # API 키 확인
    if not os.getenv("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY") == "your-upstage-api-key-here":
        print("UPSTAGE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return

    from agent.graph import create_travel_planning_graph
    from agent.nodes import USER_PROFILES
    from agent.state import create_initial_state

    # 그래프 생성 (with_memory=True로 단기 메모리 활성화)
    graph = create_travel_planning_graph(with_memory=True)

    # 설정: 같은 thread_id와 user_id로 다중 쿼리 실행
    config = {
        "configurable": {
            "thread_id": "demo-thread-001",
            "user_id": "user-001",
        }
    }

    # ==========================================
    # 쿼리 1: 여행 계획 요청
    # ==========================================
    print_section("쿼리 1: 제주도 3박 4일 여행 계획 세워줘")

    initial_state = create_initial_state("제주도 3박 4일 여행 계획 세워줘")

    # graph.stream()으로 각 노드 실행 과정을 확인
    print("\n[워크플로우 실행 과정]")
    node_count = 0

    for step in graph.stream(initial_state, config):
        # step은 {node_name: output_dict} 형태
        node_name = list(step.keys())[0]
        node_count += 1
        print(f"  [{node_count}] {node_name} 완료")

    # 최종 상태 확인
    final_state = graph.get_state(config)
    print("\n[에이전트 응답]")
    print(final_state.values.get("final_response", "응답 없음"))

    # ==========================================
    # 쿼리 2: 후속 질문 (같은 thread - 단기 메모리 시연)
    # ==========================================
    print_section("쿼리 2: 거기 맛집도 추천해줘 (같은 대화 - 단기 메모리)")

    # 같은 thread_id를 사용하면 MemorySaver가 이전 대화 맥락을 자동 복원
    followup_state = create_initial_state("거기 맛집도 추천해줘")

    print("\n[워크플로우 실행 과정]")
    node_count = 0

    for step in graph.stream(followup_state, config):
        node_name = list(step.keys())[0]
        node_count += 1
        print(f"  [{node_count}] {node_name} 완료")

    # 최종 상태 확인
    final_state = graph.get_state(config)
    print("\n[에이전트 응답]")
    print(final_state.values.get("final_response", "응답 없음"))

    # ==========================================
    # 장기 메모리 확인
    # ==========================================
    print_section("장기 메모리 (USER_PROFILES)")

    if USER_PROFILES:
        for uid, profile in USER_PROFILES.items():
            print(f"\n사용자 ID: {uid}")
            print(f"  선호 여행지: {profile.get('preferred_destinations', [])}")
            print(f"  문의 이력: {len(profile.get('query_history', []))}건")

            # 문의 이력 상세 (최근 5건만)
            query_history = profile.get('query_history', [])
            if query_history:
                print("\n  최근 문의:")
                for i, entry in enumerate(query_history[-5:], 1):
                    query_text = entry.get("query", "N/A")
                    intent = entry.get("intent", "N/A")
                    score = entry.get("quality_score", 0)
                    print(f"    {i}. [{intent}] {query_text} (품질: {score}/10)")
    else:
        print("  (저장된 프로필 없음)")

    print_section("실행 완료")


if __name__ == "__main__":
    main()
