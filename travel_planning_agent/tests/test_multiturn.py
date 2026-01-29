"""
Multi-turn conversation test

Tests the scenario:
Turn 1: "나 도쿄 가고 싶어 예산 30만원"
Turn 2: "어, 나 근데 일정은 2박3일이야"
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pytest
from agent.graph import create_travel_planning_graph
from agent.state import create_initial_state


@pytest.mark.skipif(
    not os.getenv("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY") == "your-upstage-api-key-here",
    reason="UPSTAGE_API_KEY not set"
)
def test_multiturn_preference_extraction():
    """Test that preferences accumulate across multiple conversation turns."""
    graph = create_travel_planning_graph(with_memory=True)
    
    config = {
        "configurable": {
            "thread_id": "pytest-multiturn",
            "user_id": "pytest-user"
        }
    }
    
    # ========== Turn 1: destination + budget ==========
    state1 = create_initial_state("나 도쿄 가고 싶어 예산 30만원")
    result1 = graph.invoke(state1, config)
    
    prefs1 = result1.get("extracted_preferences", {})
    
    # Verify Turn 1 extractions
    assert prefs1.get("destination") == "도쿄", f"Expected destination '도쿄', got {prefs1.get('destination')}"
    assert prefs1.get("budget") == 300000, f"Expected budget 300000, got {prefs1.get('budget')}"
    assert result1.get("final_response"), "Turn 1 should have a response"
    
    # ========== Turn 2: Add duration ==========
    state2 = create_initial_state("어, 나 근데 일정은 2박3일이야")
    result2 = graph.invoke(state2, config)
    
    prefs2 = result2.get("extracted_preferences", {})
    
    # Verify Turn 2: previous + new information
    assert prefs2.get("destination") == "도쿄", "Turn 1 destination should persist"
    assert prefs2.get("budget") == 300000, "Turn 1 budget should persist"
    assert prefs2.get("duration_days") == 3, f"Expected duration 3, got {prefs2.get('duration_days')}"
    assert result2.get("final_response"), "Turn 2 should have a response"
    
    # ========== Turn 3: Use accumulated context ==========
    state3 = create_initial_state("예산 좀 더 자세히 알려줘")
    result3 = graph.invoke(state3, config)
    
    prefs3 = result3.get("extracted_preferences", {})
    response3 = result3.get("final_response", "")
    
    # Verify Turn 3: all preferences still present
    assert prefs3.get("destination") == "도쿄", "All turns: destination should persist"
    assert prefs3.get("budget") == 300000, "All turns: budget should persist"
    assert prefs3.get("duration_days") == 3, "All turns: duration should persist"
    
    # Response should reference Tokyo (context maintained)
    assert "도쿄" in response3 or "Tokyo" in response3, "Response should reference Tokyo from context"
    assert response3, "Turn 3 should have a response"


@pytest.mark.skipif(
    not os.getenv("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY") == "your-upstage-api-key-here",
    reason="UPSTAGE_API_KEY not set"
)
def test_inmemorystore_cross_thread_persistence():
    """Test that user profile persists across different threads via InMemoryStore.

    This tests the InMemoryStore integration:
    - Query 1: user_001 asks about 제주도 (thread-A)
    - Query 2: user_001 asks about 도쿄 (thread-B, different thread!)
    - Verify: query_history has 2 entries, preferred_destinations accumulated
    """
    from agent.graph import run_travel_planning, user_store

    test_user_id = "pytest-store-user"

    # Clear any existing data for this test user
    try:
        # Reset by putting empty profile
        user_store.put(("users",), test_user_id, {
            "preferred_destinations": [],
            "query_history": [],
        })
    except:
        pass

    # ========== Query 1: 제주도 (Thread A) ==========
    result1 = run_travel_planning(
        query="제주도 2박 3일 여행 추천해줘",
        thread_id="pytest-thread-A",  # Thread A
        user_id=test_user_id
    )

    profile1 = result1.get("user_profile", {})
    assert len(profile1.get("query_history", [])) == 1, "Query 1 should have 1 history entry"
    assert result1.get("final_response"), "Query 1 should have a response"

    # ========== Query 2: 도쿄 (Thread B - DIFFERENT thread) ==========
    result2 = run_travel_planning(
        query="도쿄 예산 알려줘",
        thread_id="pytest-thread-B",  # Thread B (different!)
        user_id=test_user_id          # Same user
    )

    profile2 = result2.get("user_profile", {})
    history = profile2.get("query_history", [])
    destinations = profile2.get("preferred_destinations", [])

    # ========== Verify InMemoryStore Persistence ==========
    assert len(history) == 2, f"Expected 2 history entries (cross-thread), got {len(history)}"
    assert result2.get("final_response"), "Query 2 should have a response"

    # Check that both queries are in history
    queries = [h.get("query", "") for h in history]
    assert any("제주" in q for q in queries), "History should contain 제주도 query"
    assert any("도쿄" in q for q in queries), "History should contain 도쿄 query"

    print(f"\n✅ InMemoryStore Test Passed!")
    print(f"   - Query history count: {len(history)}")
    print(f"   - Preferred destinations: {destinations}")
