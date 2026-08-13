"""Convert internal AgentOutput dicts to display-safe AgentDisplayOutput."""

from graph.state import AgentOutput
from app.display.models import AgentDisplayOutput
from app.display.formatters import (
    confidence_label, strip_prior_knowledge_tags, grounding_pct,
    extract_core_analysis, strip_bullets, strip_chunk_refs, truncate_to_sentences,
    strip_filler_opener,
)

AGENT_DISPLAY_NAMES = {
    "macroeconomist": "Macroeconomist",
    "labor_economist": "Labor Economist",
    "trade_unionist": "Trade Unionist / People's Voice",
    "institutional": "Institutional Economist",
    "fiscal_expert": "Fiscal Policy Expert",
}


def to_display(agent_name: str, output: AgentOutput) -> AgentDisplayOutput:
    status = output.get("status", "skipped")
    response_raw = truncate_to_sentences(
        strip_filler_opener(
            strip_chunk_refs(
                strip_bullets(
                    strip_prior_knowledge_tags(
                        extract_core_analysis(output.get("response") or "")
                    )
                )
            )
        ),
        max_sentences=4,
    )
    confidence = output.get("self_confidence") or 0.0
    prior_rate = output.get("prior_knowledge_rate")

    return AgentDisplayOutput(
        agent_name=AGENT_DISPLAY_NAMES.get(agent_name, agent_name),
        response=response_raw,
        confidence=confidence,
        confidence_label=confidence_label(confidence),
        citations=[],
        rag_grounding_pct=grounding_pct(prior_rate),
        evidence_strength=output.get("evidence_strength") or 1,
        status=status,
    )


def to_display_from_state(state: dict) -> list[AgentDisplayOutput]:
    """Convert all agent outputs in a completed AgentState to display objects."""
    agent_keys = [
        ("macroeconomist", "macroeconomist_output"),
        ("labor_economist", "labor_economist_output"),
        ("trade_unionist", "trade_unionist_output"),
        ("institutional", "institutional_output"),
        ("fiscal_expert", "fiscal_output"),
    ]
    results = []
    for agent_name, key in agent_keys:
        output = state.get(key)
        if output:
            results.append(to_display(agent_name, output))
    return results


