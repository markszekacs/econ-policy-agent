"""SQLAlchemy ORM models for run persistence."""

from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, JSON, Text, ForeignKey, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class RunRecord(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(Text, nullable=False)
    ablation_mode = Column(String, default="full")
    model_id = Column(String)
    temperature = Column(Float)
    top_k = Column(Integer)
    chunk_size = Column(Integer)
    prompt_variant = Column(String, default="baseline")
    run_type = Column(String, default="production")
    experiment_id = Column(String, nullable=True)
    experiment_type = Column(String, nullable=True)
    final_output = Column(Text, nullable=True)
    total_latency_ms = Column(Integer, nullable=True)
    successful_agents = Column(Integer, default=0)
    failed_agents = Column(JSON, default=list)
    skipped_agents = Column(JSON, default=list)
    timestamp = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    judge_scores = Column(JSON, nullable=True)

    # relationships
    agent_results = relationship(
        "AgentResult",
        back_populates="run",
        cascade="all, delete-orphan"
    )
    critic_result = relationship(
        "CriticResult",
        back_populates="run",
        uselist=False,
        cascade="all, delete-orphan"
    )


class AgentResult(Base):
    __tablename__ = "agent_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    agent_name = Column(String, nullable=False)
    response = Column(Text, nullable=True)
    self_confidence = Column(Float, nullable=True)
    confidence_reasoning = Column(Text, nullable=True)
    evidence_strength = Column(Integer, nullable=True)
    prior_knowledge_rate = Column(Float, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    status = Column(String, default="skipped")
    error = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    supported_claims = Column(JSON, default=list)

    run = relationship("RunRecord", back_populates="agent_results")


class CriticResult(Base):
    __tablename__ = "critic_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    issues = Column(JSON, default=list)
    overall_quality = Column(Float, nullable=True)
    confidence_adjustments = Column(JSON, default=dict)
    reasoning = Column(Text, nullable=True)
    status = Column(String, default="skipped")
    error = Column(Text, nullable=True)
    latency_ms = Column(Integer, nullable=True)

    run = relationship("RunRecord", back_populates="critic_result")
