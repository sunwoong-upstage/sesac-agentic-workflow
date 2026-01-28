# SESAC Agentic Workflow

새싹 에이전틱 워크플로우 학습 프로젝트 저장소입니다.

## 프로젝트 구조

```
sesac-agentic-workflow/
├── travel_planning_agent/        # 메인 프로젝트: 여행 계획 AI 에이전트
└── README.md                     # 이 문서
```

## 메인 프로젝트: Travel Planning Agent

LangGraph 기반 Plan-and-Solve 여행 계획 AI 에이전트

### 주요 기능
- 의도 분류 및 Plan-and-Solve 프롬프팅
- RAG 기반 여행 지식 검색 (FAISS)
- 도구 호출 (날씨, 관광지, 예산)
- 이중 메모리 시스템 (단기/장기)
- 품질 평가 및 개선 루프

자세한 내용은 [`travel_planning_agent/README.md`](./travel_planning_agent/README.md)를 참고하세요.

## 환경 설정

이 프로젝트는 **uv**를 사용한 Python 패키지 관리를 사용합니다.

### uv 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 프로젝트 실행

```bash
# 1. travel_planning_agent로 이동
cd travel_planning_agent

# 2. 의존성 설치 (자동으로 가상환경 생성)
uv sync

# 3. .env 파일 설정
# .env 파일을 편집하여 UPSTAGE_API_KEY를 설정하세요

# 4. 실행
uv run python main.py
```

### uv 주요 명령어

```bash
# 의존성 설치
uv sync

# 패키지 추가
uv add langchain langgraph

# Python 스크립트 실행
uv run python main.py

# 특정 Python 버전 사용
uv python install 3.11
uv venv --python 3.11
```
