"""
여행 계획 에이전트 평가 모듈

LangSmith를 사용한 에이전트 품질 평가 프레임워크입니다.

사용법:
    python -m evaluate.run

모듈 구성:
    - dataset.py: 평가용 Q&A 데이터셋 (20개 예제)
    - evaluators.py: LLM-as-Judge 평가 함수들
    - run.py: 평가 실행 스크립트
"""

from evaluate.dataset import DATASET_NAME, ALL_EXAMPLES
from evaluate.evaluators import correctness, groundedness, concision

__all__ = [
    "DATASET_NAME",
    "ALL_EXAMPLES",
    "correctness",
    "groundedness",
    "concision",
]
