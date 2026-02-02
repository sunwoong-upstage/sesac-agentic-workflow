"""뉴스 챗봇 에이전트 데모"""
import logging
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    # API 키 확인
    required_keys = ["UPSTAGE_API_KEY"]
    optional_keys = ["SERPER_API_KEY"]
    
    for key in required_keys:
        if not os.getenv(key):
            logger.error(f"필수 환경 변수가 설정되지 않았습니다: {key}")
            return
    
    for key in optional_keys:
        if not os.getenv(key):
            logger.warning(f"선택 환경 변수가 설정되지 않았습니다: {key} (일부 기능 제한)")
    
    from agent import run_news_chatbot
    from agent.graph import user_store
    
    print("\n" + "=" * 60)
    print("🗞️  뉴스 챗봇 에이전트 데모")
    print("=" * 60)
    
    # 사용자/스레드 ID
    thread_id = "demo-thread-001"
    user_id = "demo-user-001"
    
    # 첫 번째 질문
    print("\n[질문 1] 엔비디아 관련 최근 뉴스 알려줘")
    print("-" * 40)
    
    result1 = run_news_chatbot(
        query="엔비디아 관련 최근 뉴스 알려줘",
        thread_id=thread_id,
        user_id=user_id,
    )
    
    print(f"\n📰 응답:\n{result1['final_response']}")
    print(f"\n📊 품질 점수: {result1['quality_score']}/10")
    
    # 두 번째 질문 (멀티턴)
    print("\n" + "=" * 60)
    print("\n[질문 2] 삼성전자도 관련 있어?")
    print("-" * 40)
    
    result2 = run_news_chatbot(
        query="삼성전자도 관련 있어?",
        thread_id=thread_id,
        user_id=user_id,
    )
    
    print(f"\n📰 응답:\n{result2['final_response']}")
    print(f"\n📊 품질 점수: {result2['quality_score']}/10")
    
    # 메모리 상태 출력
    print("\n" + "=" * 60)
    print("💾 사용자 프로필 (장기 메모리)")
    print("-" * 40)
    
    profile = user_store.get(("users",), user_id)
    if profile:
        print(f"관심 주제: {profile.value.get('interests', [])}")
        print(f"검색 이력: {len(profile.value.get('query_history', []))}개")
    
    print("\n" + "=" * 60)
    print("✅ 데모 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
