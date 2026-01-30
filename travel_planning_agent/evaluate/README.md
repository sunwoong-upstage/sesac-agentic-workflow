# 여행 계획 에이전트 평가 (Evaluation)

LangSmith를 사용한 에이전트 품질 평가 프레임워크입니다.

## 실행 방법

```bash
cd travel_planning_agent
python -m evaluate.run
```

## 구성 파일

| 파일 | 설명 |
|------|------|
| `dataset.py` | 20개 Q&A 예제 (예산, 목적지, 일정, 일반) |
| `evaluators.py` | LLM-as-Judge 평가 함수 (correctness, groundedness, concision) |
| `run.py` | 평가 실행 스크립트 |

## 데이터셋 (20 예제)

| 카테고리 | 예제 수 | 예시 |
|----------|---------|------|
| Budget Estimation | 5 | 제주도 예산, 도쿄 비용 |
| Destination Research | 5 | 맛집 추천, 벚꽃 명소 |
| Itinerary Planning | 5 | 2박3일 일정, 1주일 계획 |
| General Travel | 5 | 준비물, 환전, 영어 |

## 평가 기준 (Evaluators)

### 1. Correctness (정확성)
- LLM-as-Judge가 학생 답변과 정답(Ground Truth)을 비교
- 핵심 정보(예산, 장소, 일정)가 일치하면 통과

### 2. Groundedness (근거성)
- 답변이 검색된 문서에 기반하는지 확인
- 환각(hallucination) 검출

### 3. Concision (간결성)
- 답변 길이가 정답의 3배 이하인지 확인

## 요구사항

`.env` 파일에 다음 API 키 필요:
```
LANGSMITH_API_KEY=lsv2_...
UPSTAGE_API_KEY=up_...
```

## 결과 확인

평가 완료 후 LangSmith 대시보드에서 상세 결과 확인:
https://smith.langchain.com/
