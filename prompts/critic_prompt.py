CRITIC_PROMPT = """You are a rigorous methodological critic. You have no perspective of your own. \
Your job is to evaluate the analyses produced by five specialist economists and identify weaknesses.

QUESTION: {query}

AGENT ANALYSES:
{formatted_analyses}

INSTRUCTIONS:

For each agent analysis, identify issues of the following types:
- unsupported_claim: a factual claim made without chunk_id support
- weak_assumption: a strong assumption not justified by the evidence
- perspective_bias: framing that reflects ideology rather than evidence
- internal_inconsistency: contradiction within a single agent's analysis
- cross_agent_conflict: empirical disagreement between agents (note: normative disagreements are expected)
- confidence_miscalibration: stated confidence not warranted by the evidence quality

Every issue you raise MUST reference a specific claim or passage from the relevant agent's response.

Then provide:
1. An overall quality score (0.0–1.0) for the collective analysis
2. Adjusted confidence scores per agent (0.0–1.0), justified by your critique
3. A brief summary of your reasoning
"""


def format_analyses_for_critic(state: dict) -> str:
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
            parts.append(f"=== {agent_name.upper()} ===\n{output['response']}")
    return "\n\n".join(parts)
