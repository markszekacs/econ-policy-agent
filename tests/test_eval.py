"""Tests for eval metrics functions."""

import pytest
from eval.judges.metrics import (
    agent_confidence_variance,
    prior_knowledge_rates,
    critic_issue_taxonomy,
    latency_stats,
)

MOCK_RUNS = [
    {
        "macroeconomist": {
            "self_confidence": 0.7,
            "prior_knowledge_rate": 0.2,
            "status": "success",
        },
        "labor_economist": {
            "self_confidence": 0.6,
            "prior_knowledge_rate": 0.3,
            "status": "success",
        },
        "trade_unionist": {"status": "skipped"},
        "institutional": {"status": "skipped"},
        "fiscal_expert": {"status": "skipped"},
        "critic": {
            "issues": [
                {"issue_type": "unsupported_claim", "severity": 0.6},
                {"issue_type": "perspective_bias", "severity": 0.4},
            ]
        },
        "total_latency_ms": 12000,
    },
    {
        "macroeconomist": {
            "self_confidence": 0.8,
            "prior_knowledge_rate": 0.15,
            "status": "success",
        },
        "labor_economist": {
            "self_confidence": 0.55,
            "prior_knowledge_rate": 0.35,
            "status": "success",
        },
        "trade_unionist": {"status": "skipped"},
        "institutional": {"status": "skipped"},
        "fiscal_expert": {"status": "skipped"},
        "critic": {
            "issues": [
                {"issue_type": "unsupported_claim", "severity": 0.5},
            ]
        },
        "total_latency_ms": 14000,
    },
]


def test_confidence_variance():
    result = agent_confidence_variance(MOCK_RUNS)
    assert "macroeconomist" in result
    assert abs(result["macroeconomist"]["mean"] - 0.75) < 0.01
    assert result["macroeconomist"]["std"] > 0


def test_prior_knowledge_rates():
    result = prior_knowledge_rates(MOCK_RUNS)
    assert "macroeconomist" in result
    assert abs(result["macroeconomist"] - 0.175) < 0.01


def test_critic_taxonomy():
    result = critic_issue_taxonomy(MOCK_RUNS)
    assert result["unsupported_claim"] == 2
    assert result["perspective_bias"] == 1


def test_latency_stats():
    result = latency_stats(MOCK_RUNS)
    assert result["mean_ms"] == 13000
    assert result["min_ms"] == 12000
    assert result["max_ms"] == 14000
