"""
LLM-as-Judge 평가 함수들

세 가지 평가 기준:
- correctness: 정답과 의미적으로 일치하는지
- groundedness: 검색된 문서에 기반하는지 (환각 검사)
- concision: 답변 길이가 적절한지
"""

from pydantic import BaseModel, Field
from langchain_upstage import ChatUpstage


class CorrectnessGrade(BaseModel):
    """정답 일치 평가 결과"""
    explanation: str = Field(description="평가 근거")
    correct: bool = Field(description="학생 답변이 정답과 일치하는가?")
    score: float = Field(ge=0.0, le=1.0, description="정확도 점수 0.0~1.0")


class GroundednessGrade(BaseModel):
    """근거 기반 평가 결과 (환각 검사)"""
    explanation: str = Field(description="평가 근거")
    grounded: bool = Field(description="답변이 검색된 문서에 기반하는가?")
    score: float = Field(ge=0.0, le=1.0, description="Groundedness 점수 0.0~1.0")


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """
    LLM-as-Judge: 학생 답변이 정답과 일치하는지 평가

    Args:
        inputs: {"question": "..."}
        outputs: {"answer": "...", "documents": "..."}
        reference_outputs: {"answer": "ground truth"}

    Returns:
        bool: 정답 여부
    """
    llm = ChatUpstage(model="solar-pro")
    structured_llm = llm.with_structured_output(CorrectnessGrade)

    question = inputs.get("question", "")
    student_answer = outputs.get("answer", "")
    ground_truth = reference_outputs.get("answer", "")

    prompt = f"""당신은 여행 정보 답변을 평가하는 전문가입니다.

[질문]
{question}

[정답 (Ground Truth)]
{ground_truth}

[학생 답변]
{student_answer}

학생 답변이 정답과 **의미적으로 일치하는지** 평가하세요.
완전히 똑같을 필요는 없지만, 핵심 정보(예산, 장소, 일정 등)가 일치해야 합니다.

평가 기준:
- correct=True: 핵심 정보가 정답과 일치
- correct=False: 핵심 정보가 틀리거나 누락
- score: 0.0 (완전 틀림) ~ 1.0 (완전 일치)
"""

    result = structured_llm.invoke(prompt)
    print(f"[Correctness] Q: {question[:40]}... | Score: {result.score} | Pass: {result.correct}")

    return result.correct


def groundedness(inputs: dict, outputs: dict, reference_outputs: dict = None) -> bool:
    """
    LLM-as-Judge: 답변이 검색된 문서에 근거하는지 평가 (환각 검사)

    Args:
        inputs: {"question": "..."}
        outputs: {"answer": "...", "documents": "..."}
        reference_outputs: (사용 안 함)

    Returns:
        bool: 근거 기반 여부
    """
    llm = ChatUpstage(model="solar-pro")
    structured_llm = llm.with_structured_output(GroundednessGrade)

    question = inputs.get("question", "")
    answer = outputs.get("answer", "")
    documents = outputs.get("documents", "")

    if not documents:
        print(f"[Groundedness] No documents, skipping")
        return True  # 문서 없으면 평가 불가

    prompt = f"""당신은 답변의 근거를 평가하는 전문가입니다.

[질문]
{question}

[검색된 문서]
{documents}

[답변]
{answer}

답변의 모든 정보가 검색된 문서에서 확인 가능한지 평가하세요.
문서에 없는 정보를 지어낸 경우(환각) grounded=False로 평가하세요.

평가 기준:
- grounded=True: 모든 주장이 문서에서 확인 가능
- grounded=False: 문서에 없는 정보 포함
- score: 0.0 (완전 환각) ~ 1.0 (완전 근거 기반)
"""

    result = structured_llm.invoke(prompt)
    print(f"[Groundedness] Q: {question[:40]}... | Score: {result.score} | Pass: {result.grounded}")

    return result.grounded


def concision(outputs: dict, reference_outputs: dict) -> bool:
    """
    간결성 평가: 답변이 정답의 3배 이하 길이인가?

    Args:
        outputs: {"answer": "..."}
        reference_outputs: {"answer": "ground truth"}

    Returns:
        bool: 간결성 통과 여부
    """
    student_len = len(outputs.get("answer", ""))
    reference_len = len(reference_outputs.get("answer", ""))

    max_allowed_len = reference_len * 3
    is_concise = student_len <= max_allowed_len

    print(f"[Concision] Student: {student_len} | Reference: {reference_len} | Max: {max_allowed_len} | Pass: {is_concise}")

    return is_concise
