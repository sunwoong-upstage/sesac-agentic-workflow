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
