"""Pytest configuration for travel_planning_agent tests."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    """Load environment variables before tests run."""
    # Find .env file in project root
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        print(f"\n✅ Loaded environment from {env_file}")
    else:
        print(f"\n⚠️ No .env file found at {env_file}")


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Ensure environment is set up for all tests."""
    # Verify required keys
    required_keys = ["UPSTAGE_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]

    if missing:
        pytest.skip(f"Missing required environment variables: {missing}")

    yield
