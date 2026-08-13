"""Eval results dashboard — visualizes experiment findings."""

import json
import statistics
from pathlib import Path

import streamlit as st

from eval.judges.metrics import (
    load_runs,
    agent_confidence_variance,
    evidence_strength_by_query,
    prior_knowledge_rates,
    critic_issue_taxonomy,
    latency_stats,
)

LOGS_DIR = Path("logs")
AGENT_NAMES = [
    "macroeconomist", "labor_economist",
    "trade_unionist", "institutional", "fiscal_expert"
]
AGENT_LABELS = {
    "macroeconomist": "Macroeconomist",
    "labor_economist": "Labor Economist",
    "trade_unionist": "Trade Unionist",
    "institutional": "Institutional",
    "fiscal_expert": "Fiscal Expert",
}
AGENT_COLORS = {
    "macroeconomist": "#4A9EFF",
    "labor_economist": "#34D399",
    "trade_unionist": "#F472B6",
    "institutional": "#A78BFA",
    "fiscal_expert": "#FB923C",
}

st.markdown("""
<div style="padding:2rem 0 1rem;">
  <h1 style="
    font-size:1.8rem;font-weight:800;letter-spacing:-0.03em;
    background:linear-gradient(135deg,#4A9EFF,#A78BFA,#F472B6);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;margin:0;">
    Eval Dashboard
  </h1>
  <p style="color:rgba(232,234,240,0.4);font-size:0.8rem;
            margin:0.25rem 0 0;letter-spacing:0.1em;">
    ROBUSTNESS · CALIBRATION · RETRIEVAL · CRITIC ABLATION
  </p>
</div>
""", unsafe_allow_html=True)

# ── Experiment selector ──────────────────────────────────────────────────
experiment_type = st.selectbox(
    "Experiment",
    ["production", "temp_sweep", "critic_ablation",
     "prompt_robustness", "calibration"],
    label_visibility="collapsed",
)

log_dir = (
    LOGS_DIR / "production"
    if experiment_type == "production"
    else LOGS_DIR / "experiments" / experiment_type
)

runs = load_runs(log_dir)

if not runs:
    st.warning(
        f"No logs found in {log_dir}. "
        f"Run an experiment first."
    )
    st.stop()

st.caption(f"{len(runs)} runs loaded from `{log_dir}`")
st.divider()

# ── Overview metrics ─────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

successful = sum(
    r.get("successful_agents", 0) for r in runs
)
total_agents = len(runs) * 5
success_rate = successful / max(total_agents, 1)

lat = latency_stats(runs)
mean_lat = lat.get("mean_ms", 0)

prior_rates = prior_knowledge_rates(runs)
avg_prior = (
    statistics.mean(prior_rates.values())
    if prior_rates else 0.0
)

col1.metric("Total Runs", len(runs))
col2.metric("Agent Success Rate", f"{success_rate:.0%}")
col3.metric("Avg Latency", f"{mean_lat/1000:.1f}s")
col4.metric("Avg Prior Knowledge", f"{avg_prior:.0%}")

st.divider()

# ── Confidence variance ──────────────────────────────────────────────────
st.subheader("Self-Confidence Distribution")
st.caption(
    "Mean and standard deviation of self_confidence per agent. "
    "High std = unstable confidence estimates."
)

conf_data = agent_confidence_variance(runs)
if conf_data:
    conf_col1, conf_col2 = st.columns(2)
    for i, (agent, stats) in enumerate(conf_data.items()):
        col = conf_col1 if i % 2 == 0 else conf_col2
        with col:
            mean = stats["mean"]
            std = stats["std"]
            label = AGENT_LABELS.get(agent, agent)
            color = AGENT_COLORS.get(agent, "#888")
            st.markdown(
                f"**{label}**  \n"
                f"Mean: `{mean:.2f}` · Std: `{std:.3f}`"
            )
            st.progress(mean)

st.divider()

# ── Evidence strength by query ───────────────────────────────────────────
st.subheader("Evidence Strength by Query")
st.caption(
    "Average evidence_strength across agents per query. "
    "Low scores indicate the corpus poorly covers this topic."
)

es_data = evidence_strength_by_query(runs)
if es_data:
    for query_snippet, data in list(es_data.items())[:10]:
        mean_es = data["mean"]
        color = (
            "green" if mean_es >= 60
            else "orange" if mean_es >= 35
            else "red"
        )
        st.markdown(
            f":{color}[**{mean_es:.0f}/100**] "
            f"`{query_snippet}...`"
        )

st.divider()

# ── Prior knowledge rates ────────────────────────────────────────────────
st.subheader("Prior Knowledge Rate by Agent")
st.caption(
    "Fraction of claims marked as prior knowledge "
    "(not grounded in retrieved documents). "
    "Lower is better for RAG grounding."
)

if prior_rates:
    for agent, rate in prior_rates.items():
        label = AGENT_LABELS.get(agent, agent)
        color = AGENT_COLORS.get(agent, "#888")
        st.markdown(f"**{label}:** `{rate:.1%}`")
        st.progress(rate)

st.divider()

# ── Critic issue taxonomy ────────────────────────────────────────────────
st.subheader("Critic Issue Taxonomy")
st.caption(
    "Types of issues identified by the Critic agent "
    "across all runs."
)

taxonomy = critic_issue_taxonomy(runs)
if taxonomy:
    total_issues = sum(taxonomy.values())
    for issue_type, count in sorted(
        taxonomy.items(), key=lambda x: -x[1]
    ):
        pct = count / max(total_issues, 1)
        st.markdown(
            f"**{issue_type}**: {count} "
            f"({pct:.0%})"
        )
        st.progress(pct)
else:
    st.info("No critic issues found in these runs.")

st.divider()

# ── Latency breakdown ────────────────────────────────────────────────────
st.subheader("Latency Statistics")
if lat:
    l1, l2, l3 = st.columns(3)
    l1.metric("Mean", f"{lat.get('mean_ms',0)/1000:.1f}s")
    l2.metric("P50", f"{lat.get('p50_ms',0)/1000:.1f}s")
    l3.metric("P95", f"{lat.get('p95_ms',0)/1000:.1f}s")

st.divider()

# ── Temperature sweep analysis ───────────────────────────────────────────
if experiment_type == "temp_sweep":
    st.subheader("Temperature Sweep — Confidence Variance")
    st.caption(
        "How does temperature affect confidence stability? "
        "Higher temperature should increase variance."
    )

    temp_groups: dict[float, list] = {}
    for run in runs:
        temp = run.get("temperature")
        if temp is not None:
            temp_groups.setdefault(temp, []).append(run)

    for temp in sorted(temp_groups.keys()):
        temp_runs = temp_groups[temp]
        conf_values = []
        for run in temp_runs:
            for agent in AGENT_NAMES:
                c = run.get(agent, {}).get("self_confidence")
                if c is not None:
                    conf_values.append(c)
        if conf_values:
            mean = statistics.mean(conf_values)
            std = (statistics.stdev(conf_values)
                   if len(conf_values) > 1 else 0.0)
            st.markdown(
                f"**Temperature {temp}** — "
                f"Mean: `{mean:.2f}` · Std: `{std:.3f}` "
                f"({len(temp_runs)} runs)"
            )

# ── Critic ablation analysis ─────────────────────────────────────────────
if experiment_type == "critic_ablation":
    st.subheader("Critic Ablation — Quality by Mode")
    st.caption(
        "Does the Critic improve output quality? "
        "Compare judge scores across ablation modes."
    )

    mode_groups: dict[str, list] = {}
    for run in runs:
        mode = run.get("ablation_mode", "unknown")
        mode_groups.setdefault(mode, []).append(run)

    for mode in ["single", "multi", "full"]:
        mode_runs = mode_groups.get(mode, [])
        if not mode_runs:
            continue
        critic_quality = [
            r.get("critic", {}).get("overall_quality")
            for r in mode_runs
            if r.get("critic", {}).get("overall_quality") is not None
        ]
        n_issues = [
            len(r.get("critic", {}).get("issues", []))
            for r in mode_runs
        ]
        st.markdown(
            f"**{mode.upper()}** — "
            f"{len(mode_runs)} runs · "
            f"Avg critic quality: "
            f"`{statistics.mean(critic_quality):.2f}`"
            if critic_quality
            else f"**{mode.upper()}** — {len(mode_runs)} runs"
        )
