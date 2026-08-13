import time

from pydantic import BaseModel, Field

from config import ModelConfig
from graph.state import AgentState
from agents.base import _get_instructor_client
from prompts.synthesizer_prompt import SYNTHESIZER_PROMPT, format_for_synthesizer


class SynthesisResponse(BaseModel):
    synthesis: str = Field(description="Full structured synthesis following the five-section format")


def synthesizer_node(state: AgentState) -> dict:
    start = time.time()
    config: ModelConfig = state["metadata"]["config"].synthesizer

    try:
        client = _get_instructor_client(config.provider)

        formatted_analyses, critic_section = format_for_synthesizer(state)

        result: SynthesisResponse = client.messages.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{
                "role": "user",
                "content": SYNTHESIZER_PROMPT.format(
                    query=state["query"],
                    formatted_analyses=formatted_analyses,
                    critic_section=critic_section,
                ),
            }],
            response_model=SynthesisResponse,
            max_retries=3,
        )

        latency = int((time.time() - start) * 1000)
        return {
            "final_output": result.synthesis,
            "metadata": {
                **state["metadata"],
                "synthesizer_latency_ms": latency,
            },
        }

    except (Exception,) as e:
        return {
            "final_output": None,
            "metadata": {
                **state["metadata"],
                "synthesizer_error": f"{type(e).__name__}: {e}",
                "synthesizer_latency_ms": int(
                    (time.time() - start) * 1000
                ),
            },
        }
