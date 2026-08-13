from pydantic import BaseModel, Field
from graph.state import AgentState
from agents.base import _run_agent_node
from prompts.labor_economist_prompt import LABOR_ECONOMIST_PROMPT


class SupportedClaim(BaseModel):
    claim: str
    is_prior_knowledge: bool
    prior_knowledge_reason: str | None = None


class LaborEconomistOutput(BaseModel):
    response: str = Field(description="Exactly 4 sentences of continuous prose. No bullet points, no lists, no headers.")
    self_confidence: float = Field(ge=0.0, le=1.0)
    confidence_reasoning: str = Field(description="One sentence explaining confidence level")
    supported_claims: list[SupportedClaim] = Field(min_length=2, max_length=6)
    evidence_strength: int = Field(
        ge=1, le=100,
        description=(
            "How strong is the documentary evidence in the "
            "retrieved documents supporting your analysis? "
            "Score the quality of evidence you actually found: "
            "80-100 = multiple direct empirical studies in the "
            "documents strongly support your claims. "
            "50-79 = some relevant evidence but indirect or mixed. "
            "20-49 = weak evidence, mostly theoretical reasoning. "
            "1-19 = almost no relevant evidence found in documents. "
            "This is NOT about whether the policy is good — "
            "it is purely about how well the retrieved documents "
            "support the analysis you just wrote."
        )
    )


def labor_economist_node(state: AgentState) -> dict:
    return _run_agent_node(
        state=state,
        prompt_template=LABOR_ECONOMIST_PROMPT,
        output_schema=LaborEconomistOutput,
        agent_name="labor_economist",
        output_key="labor_economist_output",
    )
