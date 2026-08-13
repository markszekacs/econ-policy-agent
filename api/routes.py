"""API route definitions."""

import time
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.connection import get_db
from database.persistence import (
    save_run_to_db,
    get_run_by_id,
    get_recent_runs,
)
from graph.runner import run_query
from config import DEFAULT_CONFIG

router = APIRouter()


class AnalyzeRequest(BaseModel):
    question: str
    ablation_mode: str = "full"
    prompt_variant: str = "baseline"


class AgentSummary(BaseModel):
    agent_name: str
    response: str | None
    self_confidence: float | None
    evidence_strength: int | None
    status: str


class AnalyzeResponse(BaseModel):
    run_id: str
    question: str
    final_output: str | None
    agents: list[AgentSummary]
    critic_quality: float | None
    total_latency_ms: int
    successful_agents: int
    model_id: str
    prompt_variant: str


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "econ-policy-agent"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    db: Session = Depends(get_db),
):
    """
    Run a policy question through the full agent pipeline.
    Returns structured analysis with agent outputs and metadata.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    start = time.time()

    try:
        state = run_query(
            query=request.question.strip(),
            ablation_mode=request.ablation_mode,
            config=DEFAULT_CONFIG,
            prompt_variant=request.prompt_variant,
            run_type="production",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {e}"
        )

    # Persist to DB
    try:
        save_run_to_db(state, db)
    except Exception:
        pass  # DB failure should not break API response

    meta = state["metadata"]
    agent_keys = {
        "macroeconomist": "macroeconomist_output",
        "labor_economist": "labor_economist_output",
        "trade_unionist": "trade_unionist_output",
        "institutional": "institutional_output",
        "fiscal_expert": "fiscal_output",
    }

    agents = []
    for agent_name, output_key in agent_keys.items():
        output = state.get(output_key) or {}
        agents.append(AgentSummary(
            agent_name=agent_name,
            response=output.get("response"),
            self_confidence=output.get("self_confidence"),
            evidence_strength=output.get("evidence_strength"),
            status=output.get("status", "skipped"),
        ))

    critic_quality = (
        state.get("critic_output") or {}
    ).get("overall_quality")

    return AnalyzeResponse(
        run_id=meta["run_id"],
        question=request.question,
        final_output=state.get("final_output"),
        agents=agents,
        critic_quality=critic_quality,
        total_latency_ms=int((time.time() - start) * 1000),
        successful_agents=meta.get("successful_agents", 0),
        model_id=DEFAULT_CONFIG.macroeconomist.model_id,
        prompt_variant=request.prompt_variant,
    )


@router.get("/runs")
async def list_runs(
    limit: int = 20,
    run_type: str | None = None,
    db: Session = Depends(get_db),
):
    """List recent runs from the database."""
    runs = get_recent_runs(db, limit=limit, run_type=run_type)
    return [
        {
            "run_id": r.run_id,
            "query": r.query[:80],
            "ablation_mode": r.ablation_mode,
            "successful_agents": r.successful_agents,
            "total_latency_ms": r.total_latency_ms,
            "timestamp": r.timestamp.isoformat()
            if r.timestamp else None,
        }
        for r in runs
    ]


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve a specific run by ID."""
    run = get_run_by_id(run_id, db)
    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found."
        )
    return {
        "run_id": run.run_id,
        "query": run.query,
        "ablation_mode": run.ablation_mode,
        "final_output": run.final_output,
        "successful_agents": run.successful_agents,
        "total_latency_ms": run.total_latency_ms,
        "timestamp": run.timestamp.isoformat()
        if run.timestamp else None,
        "agent_results": [
            {
                "agent_name": a.agent_name,
                "response": a.response,
                "self_confidence": a.self_confidence,
                "evidence_strength": a.evidence_strength,
                "status": a.status,
                "latency_ms": a.latency_ms,
            }
            for a in run.agent_results
        ],
        "critic": {
            "overall_quality": run.critic_result.overall_quality
            if run.critic_result else None,
            "issues": run.critic_result.issues
            if run.critic_result else [],
            "reasoning": run.critic_result.reasoning
            if run.critic_result else None,
        }
    }
