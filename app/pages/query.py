import html

import streamlit as st
from graph.runner import stream_query
from app.display.agent_display import to_display
from app.components.agent_card import render_agent_card, render_agent_skeleton, render_critic_card

_AGENT_NODES: dict[str, tuple[str, str]] = {
    "macroeconomist": ("macroeconomist", "macroeconomist_output"),
    "labor_economist": ("labor_economist", "labor_economist_output"),
    "trade_unionist":  ("trade_unionist",  "trade_unionist_output"),
    "institutional":   ("institutional",   "institutional_output"),
    "fiscal_expert":   ("fiscal_expert",   "fiscal_output"),
}
_AGENT_ORDER = ["macroeconomist", "labor_economist", "trade_unionist", "institutional", "fiscal_expert"]

# ── Hero header ────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1.5rem;">
  <div style="display:inline-flex;align-items:center;gap:0.6rem;margin-bottom:0.75rem;">
    <span style="font-size:2rem;">🌐</span>
    <h1 style="
      margin:0;font-size:2.25rem;font-weight:800;letter-spacing:-0.04em;
      background:linear-gradient(135deg,#4A9EFF 0%,#A78BFA 55%,#F472B6 100%);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;">
      Economic Policy Analyst
    </h1>
  </div>
  <p style="color:rgba(232,234,240,0.35);font-size:0.78rem;margin:0;
            letter-spacing:0.14em;font-weight:500;">
    FIVE SPECIALISTS &nbsp;·&nbsp; PARALLEL ANALYSIS &nbsp;·&nbsp; RAG-GROUNDED
  </p>
</div>
""", unsafe_allow_html=True)

# ── Query form ─────────────────────────────────────────────────────────────
_, form_col, _ = st.columns([1, 5, 1])
with form_col:
    with st.form("query_form"):
        query = st.text_area(
            "Policy question",
            placeholder="e.g. Should Hungary implement a Universal Basic Income?",
            height=90,
            label_visibility="collapsed",
        )
        col_mode, col_btn = st.columns([3, 1])
        with col_mode:
            ablation_mode = st.selectbox(
                "Mode",
                options=["full", "multi", "single"],
                help="full = all agents + critic | multi = all agents, no critic | single = macroeconomist only",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Analyze →", type="primary", use_container_width=True)

# ── Pipeline ───────────────────────────────────────────────────────────────
if submitted and query.strip():
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    status = st.empty()
    status.info("Running retrieval...")

    st.markdown("""
<p style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;
          color:rgba(232,234,240,0.3);margin:1.5rem 0 0.75rem;">
  SPECIALIST ANALYSES
</p>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    placeholders: dict[str, st.empty] = {}
    active_agents = ["macroeconomist"] if ablation_mode == "single" else _AGENT_ORDER

    for i, agent_name in enumerate(_AGENT_ORDER):
        col = col1 if i % 2 == 0 else col2
        with col:
            placeholders[agent_name] = st.empty()
            if agent_name in active_agents:
                with placeholders[agent_name].container():
                    render_agent_skeleton(agent_name)

    critic_slot = st.empty()
    synthesis_slot = st.empty()

    for update in stream_query(query=query.strip(), ablation_mode=ablation_mode):
        if not update:
            continue
        for node_name, node_data in update.items():
            if not node_data:
                continue

            if node_name == "rag":
                status.info("Retrieval complete — agents analyzing in parallel...")

            elif node_name in _AGENT_NODES:
                agent_name, output_key = _AGENT_NODES[node_name]
                agent_output = node_data.get(output_key)
                if agent_output:
                    display = to_display(agent_name, agent_output)
                    with placeholders[agent_name].container():
                        render_agent_card(display)

            elif node_name == "fan_in":
                for agent_name, output_key in _AGENT_NODES.values():
                    agent_out = node_data.get(output_key)
                    if agent_out and agent_out.get("status") == "skipped":
                        display = to_display(agent_name, agent_out)
                        with placeholders[agent_name].container():
                            render_agent_card(display)

            elif node_name == "critic":
                critic_output = node_data.get("critic_output")
                if critic_output:
                    with critic_slot.container():
                        st.markdown("""
<p style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;
          color:rgba(232,234,240,0.3);margin:1.75rem 0 0.75rem;">
  CRITIC REVIEW
</p>""", unsafe_allow_html=True)
                        render_critic_card(critic_output)
                status.info("Critic review complete — synthesizing...")

            elif node_name == "synthesizer":
                final_output = node_data.get("final_output", "")
                if final_output:
                    safe_output = html.escape(final_output)
                    with synthesis_slot.container():
                        st.markdown("""
<p style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;
          color:rgba(232,234,240,0.3);margin:1.75rem 0 0.75rem;">
  SYNTHESIS
</p>""", unsafe_allow_html=True)
                        with st.container(border=True):
                            st.markdown(
                                f'<div style="font-size:0.92rem;line-height:1.8;">{safe_output}</div>',
                                unsafe_allow_html=True,
                            )

    status.success("Analysis complete.")

elif submitted:
    st.warning("Please enter a policy question.")
