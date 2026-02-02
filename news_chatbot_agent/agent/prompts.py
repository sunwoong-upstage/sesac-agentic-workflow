"""뉴스 챗봇 프롬프트 템플릿"""

# 의도 분류 프롬프트
INTENT_CLASSIFICATION_PROMPT = """당신은 뉴스 챗봇의 의도 분류기입니다.

사용자 메시지를 분석하여 다음 중 하나로 분류하세요:
- news_search: 특정 주제/키워드로 뉴스 검색 요청
- trending: 현재 인기 뉴스/트렌드 요청
- summary: 특정 뉴스 요약 요청
- general: 일반 대화 또는 인사

사용자 메시지: {user_input}

이전 대화 맥락:
{conversation_history}
"""

# 선호도 추출 프롬프트
PREFERENCE_EXTRACTION_PROMPT = """당신은 뉴스 검색 요청에서 선호도를 추출합니다.

사용자 요청에서 다음을 추출하세요:
1. 관심 주제 (예: 경제, IT, 스포츠)
2. 검색 키워드 (예: 엔비디아, 삼성전자, AI)
3. 날짜 범위 (예: "최근 7일", "이번 주", "3일 전")

사용자 요청: {user_input}

이전 선호도:
- 주제: {previous_topics}
- 키워드: {previous_keywords}

누적하여 새로운 선호도를 추출하세요.
"""

# 실행 계획 프롬프트
PLAN_PROMPT = """당신은 뉴스 검색 계획을 수립합니다.

사용자 요청: {user_input}
분류된 의도: {intent}
추출된 선호도:
- 주제: {topics}
- 키워드: {keywords}
- 날짜 범위: {date_range}

다음 도구를 사용할 수 있습니다:
1. search_news_archive: 저장된 뉴스 아카이브에서 검색 (과거 기사, 배경 정보)
2. calculate_date_range: 상대적 날짜를 실제 날짜로 변환 (예: "7일 전" → 2025-01-26)
3. search_recent_news: SERPER API로 최신 뉴스 검색 (실시간 뉴스)

효과적인 뉴스 검색을 위한 단계별 계획을 수립하세요.
"""

# 연구 실행 프롬프트
RESEARCH_PROMPT = """당신은 뉴스 연구 에이전트입니다.

실행 계획:
{execution_plan}

사용 가능한 도구:
1. search_news_archive(query: str) - 뉴스 아카이브 검색
2. calculate_date_range(time_value: int, time_unit: str) - 날짜 계산
3. search_recent_news(query: str, date_from: str) - 최신 뉴스 검색

계획에 따라 도구를 호출하여 정보를 수집하세요.
필요한 경우 여러 도구를 순차적으로 호출할 수 있습니다.
"""

# 응답 합성 프롬프트
SYNTHESIS_PROMPT = """당신은 뉴스 정보를 종합하여 사용자에게 응답합니다.

사용자 요청: {user_input}

수집된 정보:
{tool_results}

다음 지침을 따르세요:
1. **중요**: 사용자가 요청한 주제와 관련된 뉴스만 응답에 포함하세요
   - 예: "Sandisk 뉴스"를 요청했는데 "삼성전자" 뉴스가 검색되면, 삼성전자 뉴스는 무시하세요
2. search_recent_news (웹 검색) 결과를 우선적으로 활용하세요 (최신 뉴스)
3. search_news_archive (아카이브) 결과는 관련성이 있을 때만 보조 정보로 사용하세요
4. 핵심 내용을 먼저 요약하고, 출처와 날짜를 명시하세요
5. 관련 뉴스가 여러 개면 중요도/최신순으로 정리
6. 한국어로 자연스럽게 답변
7. 웹 검색에서 관련 뉴스를 찾았다면 반드시 해당 뉴스를 포함하세요
8. 정말로 어떤 도구에서도 관련 정보를 찾지 못한 경우에만 "관련 뉴스를 찾지 못했습니다"라고 답변
"""

# 품질 평가 프롬프트
EVALUATION_PROMPT = """당신은 뉴스 챗봇 응답의 품질을 평가합니다.

사용자 요청: {user_input}
생성된 응답: {response}

다음 기준으로 1-10점 평가하세요:
1. 정확성: 요청에 맞는 정보 제공 (3점)
2. 완전성: 충분한 정보 제공 (3점)
3. 명확성: 이해하기 쉬운 표현 (2점)
4. 출처: 출처/날짜 명시 (2점)

7점 이상이면 통과입니다.
"""

# 응답 개선 프롬프트
IMPROVEMENT_PROMPT = """당신은 뉴스 챗봇 응답을 개선합니다.

원래 요청: {user_input}
이전 응답: {previous_response}
평가 피드백: {feedback}

피드백을 반영하여 개선된 응답을 생성하세요.
"""
