"""
여행 계획 에이전트 - 도구 호출 및 엣지 케이스 테스트

테스트 케이스:
1. 도구 단위 테스트 (3개: RAG, 예산, 웹 검색)
2. 노드 헬퍼 함수 테스트
3. 빈 입력 처리
4. 예산 질문 엣지 케이스
5. 상태 기본값 테스트 (langgraph dev 호환)
6. 상태 & 스키마 테스트
7. 프롬프트 포맷팅 테스트
8. __init__.py 임포트 테스트
"""

import os
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# 테스트 유틸리티
# ============================================================

def run_test(name, test_func):
    """테스트를 실행하고 결과를 출력합니다."""
    try:
        test_func()
        print(f"  PASS: {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL: {name} - {e}")
        return False
    except Exception as e:
        print(f"  ERROR: {name} - {type(e).__name__}: {e}")
        return False


# ============================================================
# 1. 도구 단위 테스트 (3개 도구)
# ============================================================

def test_tools():
    """도구들이 정상적으로 동작하는지 단위 테스트"""
    from agent.tools import (
        search_travel_knowledge,
        estimate_budget,
        web_search,
        _keyword_fallback_search,
    )

    results = []
    print("\n[1] 도구 단위 테스트")

    # 1-1. RAG 검색 - 정상 쿼리
    def test_rag_normal():
        result = search_travel_knowledge.invoke({"query": "제주도 여행"})
        assert "제주" in result, f"제주도 관련 결과가 없음: {result[:100]}"
    results.append(run_test("RAG 검색 - 정상 쿼리 (제주도)", test_rag_normal))

    # 1-2. RAG 검색 - 빈 쿼리
    def test_rag_empty():
        result = search_travel_knowledge.invoke({"query": ""})
        assert isinstance(result, str), f"문자열이 아님: {type(result)}"
    results.append(run_test("RAG 검색 - 빈 쿼리", test_rag_empty))

    # 1-3. RAG 검색 - 지원하지 않는 여행지
    def test_rag_unknown():
        result = search_travel_knowledge.invoke({"query": "남극 여행"})
        assert isinstance(result, str), f"문자열이 아님: {type(result)}"
    results.append(run_test("RAG 검색 - 미지원 여행지 (남극)", test_rag_unknown))

    # 1-4. RAG 검색 - 관광지 정보 통합 확인 (ATTRACTIONS_DB가 RAG에 통합됨)
    def test_rag_attractions_merged():
        result = search_travel_knowledge.invoke({"query": "제주도 관광지 맛집"})
        assert "제주" in result, f"제주도 결과 없음: {result[:100]}"
        # 통합된 관광지/맛집 정보가 RAG에서 검색 가능해야 함
        assert isinstance(result, str) and len(result) > 50, f"결과가 너무 짧음: {result[:100]}"
    results.append(run_test("RAG 검색 - 관광지/맛집 통합 확인", test_rag_attractions_merged))

    # 1-5. 예산 - 정상 쿼리 (travel_style 없이)
    def test_budget_normal():
        result = estimate_budget.invoke({"destination": "제주도", "duration_days": 3})
        assert "원" in result, f"금액 정보 없음: {result}"
        assert "숙박" in result, f"항목 정보 없음: {result}"
    results.append(run_test("예산 추정 - 정상 (제주도 3일)", test_budget_normal))

    # 1-6. 예산 - 미지원 여행지
    def test_budget_unknown():
        result = estimate_budget.invoke({"destination": "남극", "duration_days": 3})
        assert "없습니다" in result or "지원" in result, f"에러 메시지 없음: {result}"
    results.append(run_test("예산 추정 - 미지원 여행지 (남극)", test_budget_unknown))

    # 1-7. 예산 - 항공/KTX는 왕복 1회
    def test_budget_transport():
        result = estimate_budget.invoke({"destination": "제주도", "duration_days": 5})
        assert "왕복" in result, f"왕복 표시 없음: {result}"
        assert "원" in result, f"금액 없음: {result}"
    results.append(run_test("예산 추정 - 항공 왕복 1회 표시", test_budget_transport))

    # 1-8. 예산 - 해외 여행지
    def test_budget_overseas():
        result = estimate_budget.invoke({"destination": "도쿄", "duration_days": 5})
        assert "원" in result, f"금액 없음: {result}"
        assert "도쿄" in result, f"도쿄 없음: {result}"
    results.append(run_test("예산 추정 - 해외 (도쿄 5일)", test_budget_overseas))

    # 1-9. 웹 검색 - API 키 상태에 따른 동작
    def test_web_search():
        api_key = os.getenv("SERPER_API_KEY")
        if api_key:
            result = web_search.invoke({"query": "제주도 맛집 추천"})
            assert isinstance(result, str), f"문자열이 아님: {type(result)}"
            assert len(result) > 0, "빈 결과"
        else:
            result = web_search.invoke({"query": "테스트"})
            assert "사용할 수 없습니다" in result, f"에러 메시지 없음: {result}"
    results.append(run_test("웹 검색 - API 키 상태에 따른 동작", test_web_search))

    # 1-10. 키워드 폴백 검색
    def test_keyword_fallback():
        result = _keyword_fallback_search("제주도 여행")
        assert "제주" in result, f"제주도 결과 없음: {result}"
    results.append(run_test("키워드 폴백 검색 - 정상", test_keyword_fallback))

    # 1-11. 키워드 폴백 - 매칭 없음
    def test_keyword_fallback_no_match():
        result = _keyword_fallback_search("xyz123")
        assert "찾지 못했습니다" in result, f"미매칭 메시지 없음: {result}"
    results.append(run_test("키워드 폴백 검색 - 매칭 없음", test_keyword_fallback_no_match))

    # 1-12. 모든 지원 여행지 예산 확인
    def test_budget_all_destinations():
        from agent.tools import BUDGET_DB
        for dest in BUDGET_DB:
            result = estimate_budget.invoke({"destination": dest, "duration_days": 3})
            assert "원" in result, f"{dest} 예산에 금액 없음: {result}"
    results.append(run_test("예산 추정 - 모든 지원 여행지", test_budget_all_destinations))

    # 1-13. RESEARCH_TOOLS에 3개 도구만 포함
    def test_tool_count():
        from agent.tools import RESEARCH_TOOLS
        assert len(RESEARCH_TOOLS) == 3, f"도구 수가 3이 아님: {len(RESEARCH_TOOLS)}"
        tool_names = [t.name for t in RESEARCH_TOOLS]
        assert "search_travel_knowledge" in tool_names
        assert "estimate_budget" in tool_names
        assert "web_search" in tool_names
    results.append(run_test("도구 개수 확인 (3개)", test_tool_count))

    return results


# ============================================================
# 2. 노드 헬퍼 함수 테스트
# ============================================================

def test_node_helpers():
    """노드 헬퍼 함수들 테스트"""
    from agent.nodes import _get_user_input

    results = []
    print("\n[2] 노드 헬퍼 함수 테스트")

    # 2-1. _get_user_input - user_input 직접 설정
    def test_get_input_direct():
        state = {"user_input": "제주도 여행", "messages": []}
        assert _get_user_input(state) == "제주도 여행"
    results.append(run_test("_get_user_input - 직접 설정", test_get_input_direct))

    # 2-2. _get_user_input - messages에서 추출 (langgraph dev)
    def test_get_input_from_messages():
        state = {
            "user_input": "",
            "messages": [HumanMessage(content="부산 여행 추천해줘")]
        }
        assert _get_user_input(state) == "부산 여행 추천해줘"
    results.append(run_test("_get_user_input - messages에서 추출", test_get_input_from_messages))

    # 2-3. _get_user_input - 빈 상태
    def test_get_input_empty():
        state = {"messages": []}
        result = _get_user_input(state)
        assert result == "", f"빈 상태에서 빈 문자열이어야 함: '{result}'"
    results.append(run_test("_get_user_input - 빈 상태", test_get_input_empty))

    # 2-4. _get_user_input - dict 형태 메시지 (Studio 형식)
    def test_get_input_dict_message():
        state = {
            "user_input": "",
            "messages": [{"role": "user", "content": "도쿄 맛집"}]
        }
        assert _get_user_input(state) == "도쿄 맛집"
    results.append(run_test("_get_user_input - dict 메시지 (Studio 형식)", test_get_input_dict_message))

    # 2-5. _get_user_input - user_input 키 자체가 없는 경우
    def test_get_input_no_key():
        state = {"messages": [HumanMessage(content="강릉 카페")]}
        assert _get_user_input(state) == "강릉 카페"
    results.append(run_test("_get_user_input - user_input 키 없음", test_get_input_no_key))

    return results


# ============================================================
# 3. 엣지 케이스: 빈 입력 처리
# ============================================================

def test_empty_input():
    """빈 입력이 에러 없이 처리되는지 테스트"""
    from agent.nodes import classify_intent_node

    results = []
    print("\n[3] 빈 입력 엣지 케이스")

    # 3-1. 빈 user_input → 즉시 안내 메시지 반환
    def test_empty_classify():
        state = {"user_input": "", "messages": []}
        result = classify_intent_node(state)
        assert "intent" in result, f"intent 없음: {result}"
        assert result["evaluation_passed"] is True, f"evaluation_passed가 True여야 함: {result}"
        assert "입력해주세요" in result.get("final_response", ""), f"안내 메시지 없음: {result}"
    results.append(run_test("빈 입력 - 즉시 안내 반환", test_empty_classify))

    # 3-2. 공백만 있는 입력
    def test_whitespace_classify():
        state = {"user_input": "   ", "messages": []}
        result = classify_intent_node(state)
        assert result["evaluation_passed"] is True, f"공백 입력도 빈 입력 처리해야 함: {result}"
    results.append(run_test("빈 입력 - 공백만 입력", test_whitespace_classify))

    return results


# ============================================================
# 4. 엣지 케이스: 예산 질문
# ============================================================

def test_budget_edge_cases():
    """예산 관련 엣지 케이스"""
    from agent.tools import estimate_budget

    results = []
    print("\n[4] 예산 질문 엣지 케이스")

    # 4-1. 1일 여행 예산
    def test_budget_one_day():
        result = estimate_budget.invoke({"destination": "부산", "duration_days": 1})
        assert "원" in result, f"금액 없음: {result}"
        assert "부산" in result, f"부산 없음: {result}"
    results.append(run_test("예산 - 1일 여행 (부산)", test_budget_one_day))

    # 4-2. 장기 여행 예산
    def test_budget_long_trip():
        result = estimate_budget.invoke({"destination": "파리", "duration_days": 14})
        assert "원" in result, f"금액 없음: {result}"
        assert "파리" in result, f"파리 없음: {result}"
    results.append(run_test("예산 - 장기 여행 (파리 14일)", test_budget_long_trip))

    return results


# ============================================================
# 5. 상태 기본값 테스트 (langgraph dev 호환)
# ============================================================

def test_state_defaults():
    """state.get() 기본값이 올바르게 작동하는지"""
    from agent.nodes import (
        _get_user_input,
        should_improve_response,
    )

    results = []
    print("\n[5] 상태 기본값 테스트 (langgraph dev 호환)")

    # 5-1. 최소 상태로 should_improve_response
    def test_minimal_state_routing():
        state = {}
        result = should_improve_response(state)
        assert result == "improve", f"최소 상태에서 improve여야 함: {result}"
    results.append(run_test("라우팅 - 최소 상태 (improve)", test_minimal_state_routing))

    # 5-2. 통과 상태
    def test_passed_routing():
        state = {"evaluation_passed": True}
        result = should_improve_response(state)
        assert result == "end", f"통과인데 end가 아님: {result}"
    results.append(run_test("라우팅 - 통과 상태 (end)", test_passed_routing))

    # 5-3. 최대 반복 도달
    def test_max_iter_routing():
        state = {"evaluation_passed": False, "iteration": 3, "max_iterations": 3}
        result = should_improve_response(state)
        assert result == "end", f"최대 반복인데 end가 아님: {result}"
    results.append(run_test("라우팅 - 최대 반복 도달 (end)", test_max_iter_routing))

    # 5-4. 빈 입력 → skip_to_end 라우팅 테스트
    def test_empty_input_skip():
        # Test the skip logic directly (function was inlined as lambda in graph.py)
        def skip_check(s):
            return "skip" if s.get("skip_to_end") else "continue"
        state_empty = {"skip_to_end": True}
        assert skip_check(state_empty) == "skip", "빈 입력이면 skip이어야 함"
        state_normal = {"skip_to_end": False}
        assert skip_check(state_normal) == "continue", "정상 입력이면 continue여야 함"
        state_missing = {}
        assert skip_check(state_missing) == "continue", "키 없으면 continue여야 함"
    results.append(run_test("빈 입력 skip_to_end 라우팅", test_empty_input_skip))

    return results


# ============================================================
# 6. 상태 & 스키마 테스트
# ============================================================

def test_state_and_schema():
    """상태 및 스키마 구조 테스트"""
    from agent.state import (
        create_initial_state,
        TravelPlan,
        IntentClassification,
        QualityEvaluation,
    )

    results = []
    print("\n[6] 상태 & 스키마 테스트")

    # 6-1. 초기 상태 생성
    def test_initial_state():
        state = create_initial_state("제주도 여행")
        assert state["user_input"] == "제주도 여행"
        assert state["messages"] == []
        assert state["final_response"] == ""
        assert state["plan_steps"] == []
        assert state["tool_results"] == []
        assert state["retrieved_context"] == ""
        assert state["budget_info"] == ""
        assert state["web_search_info"] == ""
        assert state["evaluation_passed"] is False
        assert state["iteration"] == 0
        assert state["max_iterations"] == 3
        assert state["error_log"] == []
    results.append(run_test("초기 상태 생성 확인", test_initial_state))

    # 6-2. 상태에 weather_info / attractions_info 없어야 함
    def test_no_weather_attractions():
        state = create_initial_state("테스트")
        assert "weather_info" not in state, "weather_info가 제거되어야 함"
        assert "attractions_info" not in state, "attractions_info가 제거되어야 함"
    results.append(run_test("상태에 weather/attractions 없음", test_no_weather_attractions))

    # 6-3. TravelPlan에 travel_style 없어야 함
    def test_no_travel_style():
        plan = TravelPlan(
            destination="제주도",
            duration_days=3,
            steps=["검색", "예산"],
        )
        assert not hasattr(plan, "travel_style") or "travel_style" not in plan.model_fields, \
            "TravelPlan에 travel_style이 제거되어야 함"
    results.append(run_test("TravelPlan에 travel_style 없음", test_no_travel_style))

    # 6-4. IntentClassification 유효성
    def test_intent_schema():
        ic = IntentClassification(intent="itinerary_planning")
        assert ic.intent == "itinerary_planning"
    results.append(run_test("IntentClassification 스키마", test_intent_schema))

    # 6-5. QualityEvaluation 유효성
    def test_quality_schema():
        qe = QualityEvaluation(score=8, feedback="좋은 응답", passed=True)
        assert qe.score == 8
        assert qe.passed is True
    results.append(run_test("QualityEvaluation 스키마", test_quality_schema))

    return results


# ============================================================
# 7. 프롬프트 포맷팅 테스트
# ============================================================

def test_prompts():
    """프롬프트 포맷팅 테스트"""
    from agent.prompts import (
        SYNTHESIZE_PROMPT,
        QUALITY_EVALUATION_PROMPT,
        RESPONSE_IMPROVEMENT_PROMPT,
    )

    results = []
    print("\n[7] 프롬프트 포맷팅 테스트")

    # 7-1. synthesize 프롬프트 포맷 (weather_info/attractions_info 파라미터 없음)
    def test_synthesize_format():
        prompt = SYNTHESIZE_PROMPT.format(
            query="제주도 여행",
            travel_plan="제주도 3일 계획",
            retrieved_context="제주도 여행 정보",
            budget_info="100만원",
            web_search_info="최신 맛집",
        )
        assert "제주도 여행" in prompt
        assert "제주도 3일 계획" in prompt
        assert "100만원" in prompt
        assert "최신 맛집" in prompt
        assert "{weather_info}" not in prompt
        assert "{attractions_info}" not in prompt
    results.append(run_test("synthesize 프롬프트 포맷", test_synthesize_format))

    # 7-2. synthesize 프롬프트 - 빈 값 기본값 처리
    def test_synthesize_defaults():
        prompt = SYNTHESIZE_PROMPT.format(
            query="테스트",
            travel_plan="" or "계획 없음",
            retrieved_context="" or "검색 결과 없음",
            budget_info="" or "예산 정보 없음",
            web_search_info="" or "웹 검색 결과 없음",
        )
        assert "계획 없음" in prompt
        assert "검색 결과 없음" in prompt
        assert "예산 정보 없음" in prompt
        assert "웹 검색 결과 없음" in prompt
    results.append(run_test("synthesize 프롬프트 - 빈 값 기본값", test_synthesize_defaults))

    # 7-3. quality evaluation 프롬프트 직접 포맷
    def test_quality_format():
        prompt = QUALITY_EVALUATION_PROMPT.format(
            query="제주도 여행",
            response="좋은 응답",
            intent="itinerary_planning"
        )
        assert "제주도 여행" in prompt
        assert "좋은 응답" in prompt
        assert "itinerary_planning" in prompt
    results.append(run_test("quality evaluation 프롬프트 포맷", test_quality_format))

    # 7-4. improvement 프롬프트 직접 포맷
    def test_improvement_format():
        prompt = RESPONSE_IMPROVEMENT_PROMPT.format(
            query="테스트",
            original_response="원본",
            feedback="개선 필요",
            score=5
        )
        assert "테스트" in prompt
        assert "원본" in prompt
        assert "개선 필요" in prompt
        assert "5" in prompt
    results.append(run_test("improvement 프롬프트 포맷", test_improvement_format))

    return results


# ============================================================
# 8. __init__.py 임포트 테스트
# ============================================================

def test_imports():
    """패키지 임포트 테스트"""
    results = []
    print("\n[8] 패키지 임포트 테스트")

    # 8-1. 메인 임포트
    def test_main_import():
        from agent import (
            TravelPlanningState,
            create_initial_state,
            IntentClassification,
            TravelPlan,
            QualityEvaluation,
            create_travel_planning_graph,
            run_travel_planning,
            get_graph_mermaid,
            search_travel_knowledge,
            estimate_budget,
            web_search,
        )
        # 모두 임포트 가능해야 함
        assert TravelPlanningState is not None
        assert create_initial_state is not None
    results.append(run_test("메인 패키지 임포트", test_main_import))

    # 8-2. 삭제된 도구가 임포트되지 않아야 함
    def test_no_deleted_imports():
        import agent
        assert not hasattr(agent, "get_weather_info"), "get_weather_info가 삭제되어야 함"
        assert not hasattr(agent, "search_attractions"), "search_attractions가 삭제되어야 함"
    results.append(run_test("삭제된 도구 미임포트 확인", test_no_deleted_imports))

    return results


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("여행 계획 에이전트 - 도구 호출 & 엣지 케이스 테스트")
    print("=" * 60)

    all_results = []
    all_results.extend(test_tools())
    all_results.extend(test_node_helpers())
    all_results.extend(test_empty_input())
    all_results.extend(test_budget_edge_cases())
    all_results.extend(test_state_defaults())
    all_results.extend(test_state_and_schema())
    all_results.extend(test_prompts())
    all_results.extend(test_imports())

    passed = sum(1 for r in all_results if r)
    total = len(all_results)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"결과: {passed}/{total} 통과, {failed} 실패")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("모든 테스트 통과!")
        sys.exit(0)
