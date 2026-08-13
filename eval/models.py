"""Pydantic schemas for eval run logs."""

import hashlib
from typing import Any
from pydantic import BaseModel, Field, model_validator


class SupportedClaimLog(BaseModel):
    claim: str
    is_prior_knowledge: bool
    prior_knowledge_reason: str | None = None
    faithfulness_score: float | None = None


class AgentLog(BaseModel):
    agent_name: str
    response: str | None
    self_confidence: float | None
    confidence_reasoning: str | None
    supported_claims: list[SupportedClaimLog]
    prior_knowledge_rate: float | None
    evidence_strength: int | None = None
    latency_ms: int | None
    status: str
    error: str | None
    retry_count: int
    judge_score: float | None = None


class CriticLog(BaseModel):
    issues: list[dict]
    overall_quality: float | None
    confidence_adjustments: dict[str, float]
    reasoning: str | None
    status: str
    error: str | None
    latency_ms: int | None


class RunLog(BaseModel):
    # Identity
    run_id: str
    experiment_id: str | None
    run_type: str                  # "production" | "experiment"
    experiment_type: str | None    # "temp_sweep" | "critic_ablation" | ...

    # Query
    query: str
    query_hash: str = Field(default="")  # MD5, set automatically

    # Config snapshot
    ablation_mode: str
    model_id: str
    temperature: float
    top_k: int
    chunk_size: int
    prompt_variant: str

    # Retrieval
    retrieved_docs: list[dict]

    # Agent outputs
    macroeconomist: AgentLog
    labor_economist: AgentLog
    trade_unionist: AgentLog
    institutional: AgentLog
    fiscal_expert: AgentLog
    critic: CriticLog

    # Synthesis
    final_output: str | None
    synthesizer_latency_ms: int | None

    # Timing
    total_latency_ms: int          # max(parallel agents) + sequential nodes
    timestamp: str
    completed_at: str

    # Aggregate metrics
    successful_agents: int
    failed_agents: list[str]
    skipped_agents: list[str]

    # Post-hoc scores (None at log time, filled by scorer.py)
    judge_scores: dict[str, Any] | None = None

    @model_validator(mode="after")
    def set_query_hash(self) -> "RunLog":
        if not self.query_hash:
            self.query_hash = hashlib.md5(self.query.encode()).hexdigest()
        return self
