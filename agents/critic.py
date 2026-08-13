import time
from typing import Literal

from pydantic import BaseModel, Field

from config import ModelConfig
from graph.state import AgentState
from agents.base import _get_instructor_client
from prompts.critic_prompt import CRITIC_PROMPT, format_analyses_for_critic


class CriticIssue(BaseModel):
    claim: str
    source_agent: str
    issue_type: Literal[
        "unsupported_claim",
        "weak_assumption",
        "perspective_bias",
        "internal_inconsistency",
        "cross_agent_conflict",
        "confidence_miscalibration",
    ]
    severity: float = Field(ge=0.0, le=1.0)


class CriticResponse(BaseModel):
    issues: list[CriticIssue] = Field(min_length=0, max_length=20)
    overall_quality: float = Field(ge=0.0, le=1.0)
    confidence_adjustments: dict[str, float] = Field(
        description="Adjusted confidence per agent, e.g. {'macroeconomist': 0.6}"
    )
    reasoning: str = Field(description="Brief summary of the critique")


def critic_node(state: AgentState) -> dict:
    start = time.time()
    config: ModelConfig = state["metadata"]["config"].critic

    try:
        client = _get_instructor_client(config.provider)

        formatted_analyses = format_analyses_for_critic(state)

        result: CriticResponse = client.messages.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{
                "role": "user",
                "content": CRITIC_PROMPT.format(
                    query=state["query"],
                    formatted_analyses=formatted_analyses,
                ),
            }],
            response_model=CriticResponse,
            max_retries=3,
        )

        return {
            "critic_output": {
                "issues": [i.model_dump() for i in result.issues],
                "overall_quality": result.overall_quality,
                "confidence_adjustments": result.confidence_adjustments,
                "reasoning": result.reasoning,
                "status": "success",
                "error": None,
                "latency_ms": int((time.time() - start) * 1000),
            }
        }

    except (Exception,) as e:
        error_msg = f"{type(e).__name__}: {e}"
        return {
            "critic_output": {
                "issues": [],
                "overall_quality": None,
                "confidence_adjustments": {},
                "reasoning": None,
                "status": "failed",
                "error": error_msg,
                "latency_ms": int((time.time() - start) * 1000),
            }
        }
