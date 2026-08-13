from typing import TypedDict, Annotated
import operator


class AgentOutput(TypedDict):
    response: str | None
    self_confidence: float | None
    confidence_reasoning: str | None
    supported_claims: list[dict]
    latency_ms: int | None
    status: str                  # "success" | "failed" | "retry_success" | "skipped"
    error: str | None
    retry_count: int
    prior_knowledge_rate: float | None
    evidence_strength: int | None


class CriticOutput(TypedDict):
    issues: list[dict]           # {"claim": str, "source_agent": str, "issue_type": str, "severity": float}
    overall_quality: float | None
    confidence_adjustments: dict  # {"macroeconomist": 0.6, ...}
    reasoning: str | None
    status: str
    error: str | None
    latency_ms: int | None


class AgentState(TypedDict):
    # core input
    query: str
    retrieved_docs: Annotated[list[dict], operator.add]

    # five parallel agent outputs
    macroeconomist_output: AgentOutput
    labor_economist_output: AgentOutput
    trade_unionist_output: AgentOutput
    institutional_output: AgentOutput
    fiscal_output: AgentOutput

    # meta agents
    critic_output: CriticOutput
    final_output: str | None

    # graph control
    ablation_mode: str           # "single" | "multi" | "full"

    # eval metadata
    metadata: dict


SKIPPED_OUTPUT: AgentOutput = {
    "response": None,
    "self_confidence": None,
    "confidence_reasoning": None,
    "supported_claims": [],
    "latency_ms": None,
    "status": "skipped",
    "error": None,
    "retry_count": 0,
    "prior_knowledge_rate": None,
    "evidence_strength": None,
}

FAILED_OUTPUT: AgentOutput = {
    "response": None,
    "self_confidence": None,
    "confidence_reasoning": None,
    "supported_claims": [],
    "latency_ms": None,
    "status": "failed",
    "error": None,
    "retry_count": 0,
    "prior_knowledge_rate": None,
    "evidence_strength": None,
}
