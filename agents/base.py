"""Generic agent runner shared by all specialist agents."""

import os
import time

import instructor
from anthropic import Anthropic
from openai import OpenAI

from config import ModelConfig
from graph.state import AgentState, SKIPPED_OUTPUT


_clients: dict = {}


def _get_instructor_client(provider: str):
    if provider not in _clients:
        if provider == "anthropic":
            _clients[provider] = instructor.from_anthropic(
                Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            )
        elif provider == "openai":
            _clients[provider] = instructor.from_openai(
                OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    return _clients[provider]


def _format_docs(docs: list[dict]) -> str:
    if not docs:
        return "[No relevant documents retrieved for this agent]"
    return "\n\n".join(doc["content"] for doc in docs)


def _run_agent_node(
    state: AgentState,
    prompt_template: str,
    output_schema: type,
    agent_name: str,
    output_key: str,
) -> dict:
    start = time.time()
    retry_count = 0

    config = getattr(state["metadata"]["config"], agent_name, None)
    if config is None:
        return {
            output_key: {
                **SKIPPED_OUTPUT,
                "status": "failed",
                "error": f"No config found for agent: {agent_name}",
                "latency_ms": 0,
            }
        }

    try:
        client = _get_instructor_client(config.provider)

        my_docs = [d for d in state["retrieved_docs"] if d["agent"] == agent_name]
        formatted_context = _format_docs(my_docs)

        result = client.messages.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=45.0,
            messages=[{
                "role": "user",
                "content": prompt_template.format(
                    query=state["query"],
                    formatted_docs=formatted_context,
                ),
            }],
            response_model=output_schema,
            max_retries=3,
        )

        prior_knowledge_rate = sum(
            1 for c in result.supported_claims if c.is_prior_knowledge
        ) / max(len(result.supported_claims), 1)

        return {
            output_key: {
                "response": result.response,
                "self_confidence": result.self_confidence,
                "confidence_reasoning": result.confidence_reasoning,
                "supported_claims": [
                    c.model_dump() for c in result.supported_claims
                ],
                "prior_knowledge_rate": prior_knowledge_rate,
                "evidence_strength": result.evidence_strength,
                "latency_ms": int((time.time() - start) * 1000),
                "status": "success",
                "error": None,
                "retry_count": retry_count,
            }
        }

    except (Exception,) as e:
        # Keep broad catch for now but log the exception type
        error_msg = f"{type(e).__name__}: {e}"
        return {
            output_key: {
                **SKIPPED_OUTPUT,
                "status": "failed",
                "error": error_msg,
                "latency_ms": int((time.time() - start) * 1000),
            }
        }
