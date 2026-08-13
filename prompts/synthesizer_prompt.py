SYNTHESIZER_PROMPT = """You are a synthesis agent producing a structured final assessment. \
Your job is to integrate the analyses from five specialist economists (and optionally a critic) \
into a coherent, honest summary. Do not paper over disagreements — name them clearly. \
Distinguish empirical disagreements from normative ones.

QUESTION: {query}

SPECIALIST ANALYSES:
{formatted_analyses}

{critic_section}

INSTRUCTIONS — structure your response as follows:

1. CONSENSUS FINDINGS
   What do the specialists broadly agree on? (empirical findings only)

2. KEY DISAGREEMENTS
   For each significant disagreement, state:
   - What the disagreement is
   - Which agents hold which position
   - Whether it is empirical (can be resolved with evidence) or normative (reflects values)

3. STRENGTH OF EVIDENCE
   How strong is the documentary evidence base? What are the key gaps?

4. POLICY IMPLICATIONS
   What policy implications follow from the analysis, and with what confidence?
   Be explicit about uncertainty. Do not overstate consensus.

5. CRITICAL UNCERTAINTIES
   What would a decision-maker most need to know that this analysis cannot answer?
"""


def format_for_synthesizer(state: dict) -> tuple[str, str]:
    """Returns (formatted_analyses, critic_section)."""
    agent_keys = [
        ("macroeconomist", "macroeconomist_output"),
        ("labor_economist", "labor_economist_output"),
        ("trade_unionist", "trade_unionist_output"),
        ("institutional", "institutional_output"),
        ("fiscal_expert", "fiscal_output"),
    ]
    parts = []
    for agent_name, key in agent_keys:
        output = state.get(key, {})
        if output.get("status") == "success" and output.get("response"):
            conf = output.get("self_confidence")
            conf_str = f" (confidence: {conf:.2f})" if conf is not None else ""
            parts.append(f"=== {agent_name.upper()}{conf_str} ===\n{output['response']}")

    formatted_analyses = "\n\n".join(parts)

    critic_output = state.get("critic_output", {})
    if critic_output and critic_output.get("status") == "success" and critic_output.get("reasoning"):
        critic_section = (
            "CRITIC REVIEW:\n"
            f"Overall quality: {critic_output.get('overall_quality', 'N/A')}\n"
            f"{critic_output['reasoning']}\n\n"
            "Confidence adjustments from critic:\n"
            + "\n".join(
                f"  {agent}: {score}"
                for agent, score in critic_output.get("confidence_adjustments", {}).items()
            )
        )
    else:
        critic_section = ""

    return formatted_analyses, critic_section
