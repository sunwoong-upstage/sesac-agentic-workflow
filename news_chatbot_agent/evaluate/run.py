"""
평가 실행 스크립트

LangSmith 데이터셋을 생성하고 에이전트 평가를 실행합니다.

실행 방법:
    python -m evaluate.run

    또는

    cd news_chatbot_agent
    python evaluate/run.py
"""

import os
import sys
import uuid
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded .env from {env_path}")

# 환경 변수 확인
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

if not LANGSMITH_API_KEY:
    print("❌ LANGSMITH_API_KEY not found. Please set it in .env file.")
    sys.exit(1)

if not UPSTAGE_API_KEY:
    print("❌ UPSTAGE_API_KEY not found. Please set it in .env file.")
    sys.exit(1)

print(f"✓ LANGSMITH_API_KEY: {LANGSMITH_API_KEY[:15]}...")
print(f"✓ UPSTAGE_API_KEY: {UPSTAGE_API_KEY[:15]}...")

# LangSmith 및 프로젝트 imports
from langsmith import Client
from langsmith.evaluation import evaluate

from agent.graph import run_news_chatbot
from evaluate.dataset import DATASET_NAME, ALL_EXAMPLES
from evaluate.evaluators import correctness, groundedness, concision


# LangSmith 클라이언트
client = Client(api_key=LANGSMITH_API_KEY)
print(f"✓ LangSmith client initialized")


def target(inputs: dict) -> dict:
    """
    평가 대상 함수: 뉴스 챗봇 에이전트 실행

    Args:
        inputs: {"question": "..."}

    Returns:
        {"answer": "...", "documents": "..."}
    """
    question = inputs["question"]
    thread_id = f"eval-{uuid.uuid4().hex[:8]}"
    user_id = "eval-user"

    print(f"\n[Agent] Q: {question[:50]}...")

    result = run_news_chatbot(
        query=question,
        thread_id=thread_id,
        user_id=user_id,
    )

    answer = result.get("final_response", "")

    # 도구 결과에서 문서 추출
    tool_results = result.get("tool_results", [])
    documents = "\n\n".join([r.get("result", "") for r in tool_results])

    print(f"[Agent] Answer: {len(answer)} chars | Docs: {len(documents)} chars")

    return {
        "answer": answer,
        "documents": documents,
    }


def create_dataset():
    """LangSmith 데이터셋 생성"""
    print(f"\n{'='*60}")
    print(f"Creating dataset: {DATASET_NAME}")
    print(f"{'='*60}")

    # 기존 데이터셋 확인
    try:
        existing = list(client.list_datasets(dataset_name=DATASET_NAME))
        if existing:
            dataset = existing[0]
            print(f"✓ Dataset exists: {dataset.id}")
            return dataset
    except Exception as e:
        print(f"  Check failed: {e}")

    # 새 데이터셋 생성
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="뉴스 챗봇 에이전트 Q&A 평가 데이터셋",
    )
    print(f"✓ Created dataset: {dataset.id}")

    # 예제 추가
    print(f"\nAdding {len(ALL_EXAMPLES)} examples...")
    for i, ex in enumerate(ALL_EXAMPLES, 1):
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            dataset_id=dataset.id,
        )
        print(f"  [{i}/{len(ALL_EXAMPLES)}] {ex['inputs']['question'][:40]}...")

    print(f"\n✓ Dataset ready!")
    print(f"  URL: https://smith.langchain.com/datasets/{dataset.id}")

    return dataset


def run_evaluation():
    """평가 실행"""
    print(f"\n{'='*60}")
    print(f"Running Evaluation")
    print(f"{'='*60}")

    # 데이터셋 생성/가져오기
    create_dataset()

    # 평가 실행
    print(f"\nEvaluators: correctness, groundedness, concision")
    print(f"This may take several minutes...\n")

    experiment_results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness, groundedness, concision],
        experiment_prefix="news-agent-eval",
        metadata={
            "model": "solar-pro2",
            "version": "1.0",
        },
        max_concurrency=2,
    )

    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")

    # pandas 결과 출력 (선택)
    try:
        import pandas as pd
        df = experiment_results.to_pandas()
        print(f"\nResults DataFrame columns: {list(df.columns)}")
        print(df.head())

        # 통계 출력 (컬럼이 존재하면)
        print(f"\n--- Statistics ---")
        print(f"Total examples: {len(df)}")
        for col in ["correctness", "groundedness", "concision"]:
            if col in df.columns:
                print(f"{col}: {df[col].mean():.1%}")
            elif f"feedback.{col}" in df.columns:
                print(f"{col}: {df[f'feedback.{col}'].mean():.1%}")
    except ImportError:
        print("(Install pandas for detailed results: pip install pandas)")
    except Exception as e:
        print(f"Results display error: {e}")

    print(f"\n✓ View results at: https://smith.langchain.com/")

    return experiment_results


if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# News Chatbot Agent Evaluation")
    print(f"{'#'*60}")

    run_evaluation()

    print(f"\n{'#'*60}")
    print(f"# Done!")
    print(f"{'#'*60}\n")
