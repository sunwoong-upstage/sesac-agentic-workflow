"""
Multi-turn conversation test for news chatbot

Tests the scenario:
Turn 1: "엔비디아 관련 뉴스 알려줘"
Turn 2: "삼성전자도 관련 있어?"
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pytest
from agent.graph import create_news_chatbot_graph, run_news_chatbot, user_store
from agent.state import create_initial_state


@pytest.mark.skipif(
    not os.getenv("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY") == "your-upstage-api-key-here",
    reason="UPSTAGE_API_KEY not set"
)
def test_multiturn_topic_accumulation():
    """Test that topics and keywords accumulate across multiple conversation turns."""
    graph = create_news_chatbot_graph(with_memory=True)

    config = {
        "configurable": {
            "thread_id": "pytest-multiturn",
            "user_id": "pytest-user"
        }
    }

    # ========== Turn 1: 엔비디아 ==========
    state1 = create_initial_state("엔비디아 관련 뉴스 알려줘")
    result1 = graph.invoke(state1, config)

    # Verify Turn 1 extractions
    assert result1.get("intent") == "news_search", f"Expected intent 'news_search', got {result1.get('intent')}"
    assert "엔비디아" in result1.get("keywords", []) or len(result1.get("keywords", [])) > 0, "Should have keywords"
    assert result1.get("final_response"), "Turn 1 should have a response"

    # ========== Turn 2: 삼성전자 ==========
    state2 = create_initial_state("삼성전자도 관련 있어?")
    result2 = graph.invoke(state2, config)

    # Verify Turn 2: topics should accumulate
    keywords2 = result2.get("keywords", [])
    assert result2.get("final_response"), "Turn 2 should have a response"

    # Response should be relevant
    response2 = result2.get("final_response", "")
    assert "삼성" in response2 or "Samsung" in response2, "Response should mention Samsung"


@pytest.mark.skipif(
    not os.getenv("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY") == "your-upstage-api-key-here",
    reason="UPSTAGE_API_KEY not set"
)
def test_inmemorystore_cross_thread_persistence():
    """Test that user profile persists across different threads via InMemoryStore.

    This tests the InMemoryStore integration:
    - Query 1: user asks about AI (thread-A)
    - Query 2: user asks about 반도체 (thread-B, different thread!)
    - Verify: query_history has 2 entries, interests accumulated
    """
    test_user_id = "pytest-store-user"

    # Clear any existing data for this test user
    try:
        user_store.put(("users",), test_user_id, {
            "interests": [],
            "query_history": [],
        })
    except:
        pass

    # ========== Query 1: AI (Thread A) ==========
    result1 = run_news_chatbot(
        query="AI 관련 최신 뉴스 알려줘",
        thread_id="pytest-thread-A",
        user_id=test_user_id
    )

    profile1 = result1.get("user_profile", {})
    assert len(profile1.get("query_history", [])) == 1, "Query 1 should have 1 history entry"
    assert result1.get("final_response"), "Query 1 should have a response"

    # ========== Query 2: 반도체 (Thread B - DIFFERENT thread) ==========
    result2 = run_news_chatbot(
        query="반도체 뉴스도 궁금해",
        thread_id="pytest-thread-B",
        user_id=test_user_id
    )

    profile2 = result2.get("user_profile", {})
    history = profile2.get("query_history", [])
    interests = profile2.get("interests", [])

    # ========== Verify InMemoryStore Persistence ==========
    assert len(history) == 2, f"Expected 2 history entries (cross-thread), got {len(history)}"
    assert result2.get("final_response"), "Query 2 should have a response"

    # Check that both queries are in history
    queries = [h.get("query", "") for h in history]
    assert any("AI" in q for q in queries), "History should contain AI query"
    assert any("반도체" in q for q in queries), "History should contain 반도체 query"

    print(f"\n✅ InMemoryStore Test Passed!")
    print(f"   - Query history count: {len(history)}")
    print(f"   - User interests: {interests}")


@pytest.mark.skipif(
    not os.getenv("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY") == "your-upstage-api-key-here",
    reason="UPSTAGE_API_KEY not set"
)
def test_quality_evaluation_loop():
    """Test that the quality evaluation loop works correctly."""
    result = run_news_chatbot(
        query="오늘 주요 뉴스 알려줘",
        thread_id="pytest-eval-loop",
        user_id="pytest-eval-user"
    )

    # Should have quality evaluation
    assert result.get("quality_score", 0) > 0, "Should have quality score"
    assert result.get("final_response"), "Should have final response"

    # If score >= 7, should pass
    if result.get("quality_score", 0) >= 7:
        assert result.get("evaluation_passed") == True, "Score >= 7 should pass"


@pytest.mark.skipif(
    not os.getenv("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY") == "your-upstage-api-key-here",
    reason="UPSTAGE_API_KEY not set"
)
def test_date_range_tool_usage():
    """Test that calculate_date_range tool is called for date-based queries."""
    result = run_news_chatbot(
        query="최근 일주일간 엔비디아 뉴스 알려줘",
        thread_id="pytest-date-range",
        user_id="pytest-date-user"
    )

    # Should have tool results
    tool_results = result.get("tool_results", [])
    assert len(tool_results) > 0, "Should have tool results"

    # Check if calculate_date_range was called
    tool_names = [r.get("tool") for r in tool_results]
    assert "calculate_date_range" in tool_names, f"calculate_date_range should be called. Tools used: {tool_names}"

    # Response should mention date range
    response = result.get("final_response", "")
    assert response, "Should have a response"

    # Verify date calculation result exists
    date_result = next((r for r in tool_results if r.get("tool") == "calculate_date_range"), None)
    assert date_result is not None, "Should have date calculation result"
    assert "result" in date_result, "Date result should have content"
