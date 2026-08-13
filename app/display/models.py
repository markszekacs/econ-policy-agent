from pydantic import BaseModel


class CitationDisplay(BaseModel):
    chunk_id: str
    source: str
    page: int


class AgentDisplayOutput(BaseModel):
    agent_name: str
    response: str          # [PRIOR KNOWLEDGE] tags stripped
    confidence: float
    confidence_label: str  # "High" | "Moderate" | "Low"
    citations: list[CitationDisplay]
    rag_grounding_pct: float  # 1.0 - prior_knowledge_rate
    evidence_strength: int  # 1-100
    status: str
