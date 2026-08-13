from dataclasses import dataclass


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    temperature: float
    max_tokens: int


@dataclass
class RetrievalConfig:
    chunk_size: int
    parent_chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval_candidates: int


@dataclass
class AgentConfig:
    macroeconomist: ModelConfig
    labor_economist: ModelConfig
    trade_unionist: ModelConfig
    institutional: ModelConfig
    fiscal_expert: ModelConfig
    critic: ModelConfig
    synthesizer: ModelConfig
    retrieval: RetrievalConfig


DEFAULT_CONFIG = AgentConfig(
    macroeconomist=ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        temperature=0.3,
        max_tokens=2000,
    ),
    labor_economist=ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        temperature=0.3,
        max_tokens=2000,
    ),
    trade_unionist=ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        temperature=0.3,
        max_tokens=2000,
    ),
    institutional=ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        temperature=0.3,
        max_tokens=2000,
    ),
    fiscal_expert=ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        temperature=0.3,
        max_tokens=2000,
    ),
    critic=ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        temperature=0.2,
        max_tokens=3000,
    ),
    synthesizer=ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        temperature=0.3,
        max_tokens=3000,
    ),
    retrieval=RetrievalConfig(
        chunk_size=400,
        parent_chunk_size=1200,
        chunk_overlap=50,
        top_k=5,
        retrieval_candidates=20,
    ),
)

TEMPERATURE_SWEEP_CONFIGS = [0.0, 0.2, 0.5, 0.7]
