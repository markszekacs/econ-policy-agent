"""LLM-as-judge scoring for agent outputs."""

import os
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class QualityScore(BaseModel):
    reasoning_quality: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How well-reasoned is this analysis? "
            "1.0 = rigorous, evidence-based, nuanced. "
            "0.0 = superficial, unsupported, vague."
        )
    )
    evidence_use: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How well does the response use available evidence? "
            "1.0 = every claim grounded in evidence. "
            "0.0 = pure assertion, no grounding."
        )
    )
    perspective_consistency: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How consistent is this with the agent's stated role? "
            "1.0 = clearly reflects the agent perspective. "
            "0.0 = could have been written by any agent."
        )
    )
    overall: float = Field(
        ge=0.0, le=1.0,
        description="Overall quality score."
    )
    brief_reasoning: str = Field(
        description="One sentence explaining the overall score."
    )


class AgentAgreementScore(BaseModel):
    agreement_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How much do these two analyses agree? "
            "1.0 = identical conclusions. "
            "0.0 = directly contradictory."
        )
    )
    disagreement_type: str = Field(
        description=(
            "One of: empirical, normative, emphasis, none. "
            "empirical = disagree on facts. "
            "normative = disagree on values. "
            "emphasis = same facts, different weight. "
            "none = no meaningful disagreement."
        )
    )
    brief_reasoning: str


_judge_client = None

def _get_judge_client():
    global _judge_client
    if _judge_client is None:
        _judge_client = instructor.from_anthropic(
            Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        )
    return _judge_client


def score_agent_quality(
    query: str,
    agent_name: str,
    agent_response: str,
    evidence_strength: int | None,
) -> QualityScore | None:
    """Score a single agent response for quality."""
    if not agent_response:
        return None

    client = _get_judge_client()

    prompt = f"""You are an expert evaluator assessing the quality of
economic policy analysis.

POLICY QUESTION: {query}

AGENT ROLE: {agent_name.replace("_", " ").title()}

AGENT RESPONSE:
{agent_response}

AGENT'S SELF-REPORTED EVIDENCE STRENGTH: {evidence_strength}/100

Evaluate this response on three dimensions and provide
an overall score. Be calibrated — most responses should
score 0.4-0.8, not extreme values."""

    try:
        result = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
            response_model=QualityScore,
            max_retries=2,
        )
        return result
    except Exception:
        return None


def score_agent_agreement(
    query: str,
    agent_a_name: str,
    agent_a_response: str,
    agent_b_name: str,
    agent_b_response: str,
) -> AgentAgreementScore | None:
    """Score agreement between two agent responses."""
    if not agent_a_response or not agent_b_response:
        return None

    client = _get_judge_client()

    prompt = f"""You are evaluating agreement between two economic analysts.

POLICY QUESTION: {query}

{agent_a_name.upper()} SAYS:
{agent_a_response}

{agent_b_name.upper()} SAYS:
{agent_b_response}

How much do these analyses agree? Identify the type
of disagreement if any exists."""

    try:
        result = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
            response_model=AgentAgreementScore,
            max_retries=2,
        )
        return result
    except Exception:
        return None
