"""LangGraph graph construction for the economic policy multi-agent system."""

import time
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END

from graph.state import AgentState, SKIPPED_OUTPUT, CriticOutput
from agents.macroeconomist import macroeconomist_node
from agents.labor_economist import labor_economist_node
from agents.trade_unionist import trade_unionist_node
from agents.institutional import institutional_node
from agents.fiscal_expert import fiscal_expert_node
from agents.critic import critic_node
from agents.synthesizer import synthesizer_node
from eval.logger import logger_node
from rag.retriever import retrieve_all_agents

AGENTS = {
    "macroeconomist": "macroeconomist_output",
    "labor_economist": "labor_economist_output",
    "trade_unionist": "trade_unionist_output",
    "institutional": "institutional_output",
    "fiscal_expert": "fiscal_output",
}
ALL_AGENT_NAMES = list(AGENTS.keys())
AGENT_OUTPUT_KEYS = list(AGENTS.values())

EMPTY_CRITIC: CriticOutput = {
    "issues": [],
    "overall_quality": None,
    "confidence_adjustments": {},
    "reasoning": None,
    "status": "skipped",
    "error": None,
    "latency_ms": None,
}


def rag_node(state: AgentState) -> dict:
    retrieval_start = time.time()
    ablation_mode = state.get("ablation_mode", "full")

    if ablation_mode == "single":
        agent_names = ["macroeconomist"]
    else:
        agent_names = ALL_AGENT_NAMES

    cohere_api_key = None  # pulled from env inside retriever
    docs = retrieve_all_agents(
        query=state["query"],
        retrieval_config=state["metadata"]["config"].retrieval,
        agent_names=agent_names,
        cohere_api_key=cohere_api_key,
    )

    retrieval_latency = int((time.time() - retrieval_start) * 1000)

    return {
        "retrieved_docs": docs,
        "metadata": {
            **state["metadata"],
            "retrieval_latency_ms": retrieval_latency,
        },
    }


def route_agents(state: AgentState) -> list[str]:
    mode = state.get("ablation_mode", "full")
    if mode == "single":
        return ["macroeconomist"]
    return ALL_AGENT_NAMES


def fan_in_node(state: AgentState) -> dict:
    successful = []
    failed = []
    skipped = []

    updates: dict = {}
    for agent_name, output_key in zip(ALL_AGENT_NAMES, AGENT_OUTPUT_KEYS):
        output = state.get(output_key)
        if output is None:
            updates[output_key] = {**SKIPPED_OUTPUT}
            skipped.append(agent_name)
        elif output.get("status") in ("success", "retry_success"):
            successful.append(agent_name)
        elif output.get("status") == "failed":
            failed.append(agent_name)
        else:
            skipped.append(agent_name)

    updates["metadata"] = {
        **state["metadata"],
        "successful_agents": len(successful),
        "failed_agents": failed,
        "skipped_agents": skipped,
        "fan_in_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return updates


def route_after_fanin(state: AgentState) -> str:
    mode = state.get("ablation_mode", "full")
    successful = state["metadata"].get("successful_agents", 0)
    if mode == "full" and successful >= 3:
        return "critic"
    return "synthesizer"


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("rag", rag_node)
    builder.add_node("macroeconomist", macroeconomist_node)
    builder.add_node("labor_economist", labor_economist_node)
    builder.add_node("trade_unionist", trade_unionist_node)
    builder.add_node("institutional", institutional_node)
    builder.add_node("fiscal_expert", fiscal_expert_node)
    builder.add_node("fan_in", fan_in_node)
    builder.add_node("critic", critic_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("logger", logger_node)

    builder.add_edge(START, "rag")
    builder.add_conditional_edges("rag", route_agents)

    for agent in ALL_AGENT_NAMES:
        builder.add_edge(agent, "fan_in")

    builder.add_conditional_edges(
        "fan_in",
        route_after_fanin,
        {"critic": "critic", "synthesizer": "synthesizer"},
    )

    builder.add_edge("critic", "synthesizer")
    builder.add_edge("synthesizer", "logger")
    builder.add_edge("logger", END)

    return builder.compile()


# Module-level compiled graph (lazy singleton pattern via function)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
