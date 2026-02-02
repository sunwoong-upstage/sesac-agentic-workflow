# 뉴스 챗봇 에이전트 평가

LangSmith를 사용한 에이전트 평가 모듈입니다.

## 실행 방법

```bash
# 프로젝트 루트에서
python -m evaluate.run

# 또는
python evaluate/run.py
```

## 필요 환경 변수

```
UPSTAGE_API_KEY=...
LANGSMITH_API_KEY=...
```

## 평가 기준

1. **Correctness**: 정답과 의미적으로 일치하는지
2. **Groundedness**: 검색된 문서에 기반하는지 (환각 검사)
3. **Concision**: 답변 길이가 적절한지

## 데이터셋

4개 카테고리, 총 20개 예제:
- News Search (5)
- Trending Topics (5)
- News Summary (5)
- General Info (5)
