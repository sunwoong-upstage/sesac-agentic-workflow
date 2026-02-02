"""
평가용 Q&A 데이터셋

5개 카테고리, 총 23개 예제:
- News Search (5)
- Trending Topics (5)
- News Summary (5)
- General Info (5)
- Date Range (3)
"""

DATASET_NAME = "News Chatbot Agent Q&A"

# Category 1: News Search (5 examples)
NEWS_SEARCH_EXAMPLES = [
    {
        "inputs": {"question": "엔비디아 관련 최신 뉴스 알려줘"},
        "outputs": {
            "answer": "엔비디아 관련 최신 뉴스: 1) 4분기 실적 발표 - 매출 220억 달러로 전년 대비 120% 증가, 2) AI 칩 수요 폭발로 GPU 공급 부족, 3) 삼성전자 HBM3E 납품 계약 체결"
        }
    },
    {
        "inputs": {"question": "삼성전자 반도체 뉴스 검색해줘"},
        "outputs": {
            "answer": "삼성전자 반도체 뉴스: 1) HBM3E 양산 본격화로 엔비디아 납품 확대, 2) 4분기 영업이익 20조원 기록, 3) AI 반도체 R&D 투자 확대 발표"
        }
    },
    {
        "inputs": {"question": "오늘 IT 뉴스 있어?"},
        "outputs": {
            "answer": "오늘 IT 뉴스: 1) OpenAI GPT-5 개발 막바지, 연내 출시 예정, 2) 카카오 AI 검색 서비스 '카나' 출시, 3) SK텔레콤 에이닷 2.0 멀티모달 AI 적용"
        }
    },
    {
        "inputs": {"question": "테슬라 자율주행 소식 알려줘"},
        "outputs": {
            "answer": "테슬라 자율주행 뉴스: 로보택시 6월 출시 확정. 운전대와 페달 없는 완전 자율주행 차량으로 텍사스와 캘리포니아에서 서비스 시작 예정."
        }
    },
    {
        "inputs": {"question": "현대차 전기차 뉴스"},
        "outputs": {
            "answer": "현대차 전기차 뉴스: 전고체 배터리 탑재 EV 2027년 출시 계획. 삼성SDI와 협력하여 충전 10분에 주행거리 600km 목표."
        }
    },
]

# Category 2: Trending Topics (5 examples)
TRENDING_EXAMPLES = [
    {
        "inputs": {"question": "요즘 핫한 뉴스 뭐야?"},
        "outputs": {
            "answer": "최근 인기 뉴스: 1) AI 반도체 경쟁 심화 (엔비디아, 삼성, SK하이닉스), 2) GPT-5 출시 임박, 3) 테슬라 로보택시 발표, 4) 애플 비전프로2 개발"
        }
    },
    {
        "inputs": {"question": "이번 주 핫토픽 알려줘"},
        "outputs": {
            "answer": "이번 주 핫토픽: 1) 정부 AI 반도체 10조원 투자 발표, 2) 네이버 하이퍼클로바X 일본 진출, 3) 빅테크 실적 발표 시즌"
        }
    },
    {
        "inputs": {"question": "AI 관련 트렌드는?"},
        "outputs": {
            "answer": "AI 트렌드: 1) 멀티모달 AI 확산 (텍스트+이미지+음성), 2) AI 에이전트 기술 발전, 3) 온디바이스 AI 경쟁, 4) AI 규제 논의 활발"
        }
    },
    {
        "inputs": {"question": "경제 뉴스 트렌드"},
        "outputs": {
            "answer": "경제 뉴스 트렌드: 1) 반도체 슈퍼사이클 전망, 2) 미중 기술 패권 경쟁, 3) 금리 인하 기대감, 4) ESG 투자 확대"
        }
    },
    {
        "inputs": {"question": "기술 트렌드 뉴스"},
        "outputs": {
            "answer": "기술 트렌드: 1) 생성형 AI 고도화, 2) 전고체 배터리 상용화, 3) 6G 기술 개발, 4) 양자 컴퓨팅 진전"
        }
    },
]

# Category 3: News Summary (5 examples)
SUMMARY_EXAMPLES = [
    {
        "inputs": {"question": "엔비디아 실적 요약해줘"},
        "outputs": {
            "answer": "엔비디아 2024년 4분기 실적 요약: 매출 220억 달러 (전년 대비 120% 증가), AI 칩 수요 폭발이 주요 원인. 젠슨 황 CEO는 AI 시대 본격 시작이라고 평가."
        }
    },
    {
        "inputs": {"question": "삼성전자 HBM 관련 내용 정리해줘"},
        "outputs": {
            "answer": "삼성전자 HBM 현황: HBM3E 양산 본격화, 엔비디아 차세대 GPU에 탑재 예정. SK하이닉스와의 기술 격차 축소 중. 품질 문제 해결 및 수율 개선 완료."
        }
    },
    {
        "inputs": {"question": "OpenAI GPT-5 뉴스 요약"},
        "outputs": {
            "answer": "OpenAI GPT-5 요약: 2025년 내 출시 예정. 멀티모달 기능 대폭 강화, 추론 능력 향상. 샘 알트만 CEO는 AGI 향한 이정표가 될 것이라고 언급."
        }
    },
    {
        "inputs": {"question": "정부 AI 반도체 정책 요약"},
        "outputs": {
            "answer": "정부 AI 반도체 정책: 5년간 10조원 투자 계획. 팹리스 기업 육성, 인력 양성, R&D 지원에 집중. 미중 반도체 패권 경쟁 속 한국 경쟁력 강화 목표."
        }
    },
    {
        "inputs": {"question": "테슬라 로보택시 발표 내용"},
        "outputs": {
            "answer": "테슬라 로보택시: 2025년 6월 출시 확정. 운전대·페달 없는 완전 자율주행. 텍사스, 캘리포니아에서 서비스 시작. 머스크는 테슬라 역사상 가장 중요한 제품이라고 강조."
        }
    },
]

# Category 4: General Info (5 examples)
GENERAL_EXAMPLES = [
    {
        "inputs": {"question": "뉴스 어떻게 검색해?"},
        "outputs": {
            "answer": "뉴스 검색 방법: 1) 키워드로 검색 (예: '엔비디아 뉴스'), 2) 날짜 범위 지정 (예: '최근 7일'), 3) 카테고리 검색 (예: 'IT 뉴스'). 질문하시면 관련 뉴스를 찾아드립니다."
        }
    },
    {
        "inputs": {"question": "어떤 뉴스를 볼 수 있어?"},
        "outputs": {
            "answer": "제공 가능한 뉴스: IT/기술, 경제, 정책, 자동차 등 다양한 분야. 저장된 뉴스 아카이브와 실시간 뉴스 검색을 통해 최신 정보를 제공합니다."
        }
    },
    {
        "inputs": {"question": "뉴스 출처는 어디야?"},
        "outputs": {
            "answer": "뉴스 출처: 한국경제, 매일경제, 연합뉴스, 전자신문, ZDNet Korea 등 국내 주요 언론사. 실시간 검색 시 구글 뉴스 기반 결과도 제공."
        }
    },
    {
        "inputs": {"question": "날짜별로 뉴스 볼 수 있어?"},
        "outputs": {
            "answer": "날짜별 검색 가능: '최근 7일', '이번 주', '3일 전' 등 상대적 날짜로 검색 가능. 예: '엔비디아 최근 2주 뉴스 알려줘'"
        }
    },
    {
        "inputs": {"question": "안녕하세요"},
        "outputs": {
            "answer": "안녕하세요! 뉴스 챗봇입니다. 궁금한 뉴스나 최신 소식을 물어보세요. 예: '오늘 IT 뉴스', '엔비디아 관련 뉴스', '이번 주 핫토픽'"
        }
    },
]

# Category 5: Date Range (3 examples)
DATE_RANGE_EXAMPLES = [
    {
        "inputs": {"question": "최근 일주일간 엔비디아 뉴스 알려줘"},
        "outputs": {
            "answer": "최근 7일간 엔비디아 뉴스: [날짜 범위와 함께 뉴스 제공]. 검색 기간: YYYY-MM-DD ~ YYYY-MM-DD"
        }
    },
    {
        "inputs": {"question": "지난 2주 동안 AI 관련 뉴스"},
        "outputs": {
            "answer": "최근 2주간 AI 뉴스: [날짜 범위와 함께 뉴스 제공]. 검색 기간: YYYY-MM-DD ~ YYYY-MM-DD"
        }
    },
    {
        "inputs": {"question": "3일 전부터 오늘까지 삼성전자 뉴스 검색해줘"},
        "outputs": {
            "answer": "최근 3일간 삼성전자 뉴스: [날짜 범위와 함께 뉴스 제공]. 검색 기간: YYYY-MM-DD ~ YYYY-MM-DD"
        }
    },
]

# 전체 예제 통합
ALL_EXAMPLES = (
    NEWS_SEARCH_EXAMPLES +
    TRENDING_EXAMPLES +
    SUMMARY_EXAMPLES +
    GENERAL_EXAMPLES +
    DATE_RANGE_EXAMPLES
)
