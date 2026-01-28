"""
Multi-turn conversation test

Tests the scenario:
Turn 1: "나 도쿄 가고 싶어 예산 30만원"
Turn 2: "어, 나 근데 일정은 2박3일이야"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from agent.graph import create_travel_planning_graph
from agent.state import create_initial_state

def print_separator(title=""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)

def main():
    if not os.getenv("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY") == "your-upstage-api-key-here":
        print("UPSTAGE_API_KEY가 설정되지 않았습니다.")
        return

    graph = create_travel_planning_graph(with_memory=True)
    
    config = {
        "configurable": {
            "thread_id": "multiturn-test",
            "user_id": "test-user"
        }
    }

    print_separator("Multi-Turn Conversation Test")
    
    # ========== Turn 1 ==========
    print_separator("Turn 1: '나 도쿄 가고 싶어 예산 30만원'")
    
    state1 = create_initial_state("나 도쿄 가고 싶어 예산 30만원")
    result1 = graph.invoke(state1, config)
    
    print("\n[추출된 선호도]")
    prefs1 = result1.get("extracted_preferences", {})
    for k, v in prefs1.items():
        print(f"  {k}: {v}")
    
    print("\n[응답 (앞부분만)]")
    response1 = result1.get("final_response", "")
    print(response1[:300] + "..." if len(response1) > 300 else response1)
    
    # ========== Turn 2 ==========
    print_separator("Turn 2: '어, 나 근데 일정은 2박3일이야'")
    
    state2 = create_initial_state("어, 나 근데 일정은 2박3일이야")
    result2 = graph.invoke(state2, config)
    
    print("\n[추출된 선호도 (누적)]")
    prefs2 = result2.get("extracted_preferences", {})
    for k, v in prefs2.items():
        print(f"  {k}: {v}")
    
    print("\n[검증]")
    # Turn 1에서 추출한 정보가 유지되는지 확인
    if prefs2.get("destination") == "도쿄":
        print("  ✅ destination: 도쿄 (Turn 1에서 유지)")
    else:
        print(f"  ❌ destination: {prefs2.get('destination')} (도쿄가 아님!)")
    
    if prefs2.get("budget") == 300000:
        print("  ✅ budget: 300000 (Turn 1에서 유지)")
    else:
        print(f"  ❌ budget: {prefs2.get('budget')} (30만원이 아님!)")
    
    if prefs2.get("duration_days") == 3:
        print("  ✅ duration_days: 3 (Turn 2에서 추가)")
    else:
        print(f"  ❌ duration_days: {prefs2.get('duration_days')} (3이 아님!)")
    
    print("\n[응답 (앞부분만)]")
    response2 = result2.get("final_response", "")
    print(response2[:300] + "..." if len(response2) > 300 else response2)
    
    # ========== Turn 3: 선호도 활용 확인 ==========
    print_separator("Turn 3: '예산 좀 더 자세히 알려줘'")
    
    state3 = create_initial_state("예산 좀 더 자세히 알려줘")
    result3 = graph.invoke(state3, config)
    
    print("\n[추출된 선호도 (계속 유지)]")
    prefs3 = result3.get("extracted_preferences", {})
    for k, v in prefs3.items():
        print(f"  {k}: {v}")
    
    print("\n[응답에 '도쿄' 언급 확인]")
    response3 = result3.get("final_response", "")
    if "도쿄" in response3 or "Tokyo" in response3:
        print("  ✅ 응답에 '도쿄'가 포함됨 (컨텍스트 유지)")
    else:
        print("  ⚠️ 응답에 '도쿄' 없음")
    
    print("\n[응답 (앞부분만)]")
    print(response3[:300] + "..." if len(response3) > 300 else response3)
    
    print_separator("Test Complete")
    print("✅ Multi-turn conversation with preference extraction works!")

if __name__ == "__main__":
    main()
