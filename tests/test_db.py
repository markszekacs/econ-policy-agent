"""Tests for database persistence."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, RunRecord, AgentResult
from database.persistence import save_run_to_db, get_run_by_id

from config import DEFAULT_CONFIG
from datetime import datetime
import uuid


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _mock_state(query: str = "Test query") -> dict:
    run_id = str(uuid.uuid4())
    return {
        "query": query,
        "ablation_mode": "full",
        "final_output": "Test synthesis output.",
        "retrieved_docs": [],
        "macroeconomist_output": {
            "response": "Test macro response.",
            "self_confidence": 0.75,
            "confidence_reasoning": "Strong evidence.",
            "evidence_strength": 70,
            "prior_knowledge_rate": 0.2,
            "latency_ms": 3000,
            "status": "success",
            "error": None,
            "retry_count": 0,
            "supported_claims": [],
        },
        "labor_economist_output": None,
        "trade_unionist_output": None,
        "institutional_output": None,
        "fiscal_output": None,
        "critic_output": {
            "issues": [],
            "overall_quality": 0.8,
            "confidence_adjustments": {},
            "reasoning": "Good analysis.",
            "status": "success",
            "error": None,
            "latency_ms": 2000,
        },
        "metadata": {
            "run_id": run_id,
            "experiment_id": None,
            "run_type": "production",
            "experiment_type": None,
            "prompt_variant": "baseline",
            "config": DEFAULT_CONFIG,
            "ablation_mode": "full",
            "temperature": 0.3,
            "timestamp": datetime.utcnow().isoformat(),
            "retrieval_latency_ms": 500,
            "synthesizer_latency_ms": 2000,
            "total_latency_ms": 8000,
            "successful_agents": 1,
            "failed_agents": [],
            "skipped_agents": [
                "labor_economist",
                "trade_unionist",
                "institutional",
                "fiscal_expert"
            ],
            "fan_in_timestamp": datetime.utcnow().isoformat(),
        }
    }


def test_save_and_retrieve_run(db_session):
    state = _mock_state("Should Hungary implement UBI?")
    run = save_run_to_db(state, db_session)

    assert run.run_id == state["metadata"]["run_id"]
    assert run.query == "Should Hungary implement UBI?"
    assert run.successful_agents == 1

    retrieved = get_run_by_id(run.run_id, db_session)
    assert retrieved is not None
    assert retrieved.query == "Should Hungary implement UBI?"


def test_agent_results_saved(db_session):
    state = _mock_state()
    run = save_run_to_db(state, db_session)

    macro_result = db_session.query(AgentResult).filter(
        AgentResult.run_id == run.run_id,
        AgentResult.agent_name == "macroeconomist"
    ).first()

    assert macro_result is not None
    assert macro_result.self_confidence == 0.75
    assert macro_result.evidence_strength == 70
    assert macro_result.status == "success"
