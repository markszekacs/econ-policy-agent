import streamlit as st
from app.display.models import AgentDisplayOutput

AGENT_ICONS = {
    "Macroeconomist":                  "📊",
    "Labor Economist":                 "⚙️",
    "Trade Unionist / People's Voice": "✊",
    "Institutional Economist":         "🏛️",
    "Fiscal Policy Expert":            "💰",
}

AGENT_HEX = {
    "Macroeconomist":                  "#4A9EFF",
    "Labor Economist":                 "#34D399",
    "Trade Unionist / People's Voice": "#FB923C",
    "Institutional Economist":         "#C084FC",
    "Fiscal Policy Expert":            "#F87171",
}

_INTERNAL_TO_DISPLAY = {
    "macroeconomist": "Macroeconomist",
    "labor_economist": "Labor Economist",
    "trade_unionist":  "Trade Unionist / People's Voice",
    "institutional":   "Institutional Economist",
    "fiscal_expert":   "Fiscal Policy Expert",
}

STATUS_ICON  = {"success": "●", "retry_success": "●", "failed": "✕", "skipped": "○"}
STATUS_COLOR = {"success": "#34D399", "retry_success": "#FCD34D", "failed": "#F87171", "skipped": "#6B7280"}


def _bar(label: str, value: float, hex_color: str, badge: str = "") -> None:
    pct = int(value * 100)
    badge_html = (
        f'<span style="font-size:0.68rem;font-weight:600;color:{hex_color};'
        f'background:{hex_color}1A;padding:1px 6px;border-radius:100px;margin-left:5px;">'
        f'{badge}</span>'
    ) if badge else ""
    st.markdown(f"""
<div style="margin-bottom:0.55rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;
              font-size:0.73rem;color:rgba(232,234,240,0.45);margin-bottom:4px;">
    <span>{label}{badge_html}</span>
    <span style="font-weight:700;color:{hex_color};">{pct}%</span>
  </div>
  <div style="height:4px;background:rgba(255,255,255,0.07);border-radius:100px;overflow:hidden;">
    <div style="height:100%;width:{pct}%;
                background:linear-gradient(90deg,{hex_color}88,{hex_color});
                border-radius:100px;"></div>
  </div>
</div>""", unsafe_allow_html=True)


def _evidence_bar(value: int) -> None:
    pct = max(1, min(100, value))
    bar_color = "#F87171" if pct < 40 else "#FB923C" if pct < 70 else "#34D399"
    label = "Weak evidence" if pct < 40 else "Moderate evidence" if pct < 70 else "Strong evidence"

    st.markdown(f"""
<div style="margin-bottom:0.55rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;
              font-size:0.73rem;color:rgba(232,234,240,0.45);margin-bottom:4px;">
    <span>Evidence strength <span style="color:{bar_color};font-weight:600;">{label}</span></span>
    <span style="font-weight:700;color:{bar_color};">{pct}/100</span>
  </div>
  <div style="height:6px;background:rgba(255,255,255,0.07);border-radius:100px;overflow:hidden;">
    <div style="height:100%;width:{pct}%;background:{bar_color};border-radius:100px;"></div>
  </div>
</div>""", unsafe_allow_html=True)


def _card_header(display_name: str, icon: str, hex_color: str, status: str = "success") -> None:
    s_icon  = STATUS_ICON.get(status, "○")
    s_color = STATUS_COLOR.get(status, "#6B7280")
    st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding-bottom:0.6rem;margin-bottom:0.65rem;
            border-bottom:1px solid {hex_color}28;">
  <div style="display:flex;align-items:center;gap:0.5rem;">
    <span style="font-size:1.25rem;line-height:1;">{icon}</span>
    <span style="font-size:0.8rem;font-weight:700;letter-spacing:0.1em;
                 color:{hex_color};text-transform:uppercase;">{display_name}</span>
  </div>
  <span style="font-size:0.72rem;color:{s_color};font-weight:700;">{s_icon}</span>
</div>""", unsafe_allow_html=True)


ISSUE_TYPE_LABELS = {
    "unsupported_claim":        "Unsupported claim",
    "weak_assumption":          "Weak assumption",
    "perspective_bias":         "Perspective bias",
    "internal_inconsistency":   "Internal inconsistency",
    "cross_agent_conflict":     "Cross-agent conflict",
    "confidence_miscalibration":"Confidence miscalibration",
}

AGENT_SHORT = {
    "macroeconomist": "Macro",
    "labor_economist": "Labor",
    "trade_unionist": "Trade",
    "institutional": "Institutional",
    "fiscal_expert": "Fiscal",
}


def render_critic_card(critic_output: dict) -> None:
    import html as _html

    status = critic_output.get("status", "failed")
    quality = critic_output.get("overall_quality")
    reasoning = critic_output.get("reasoning") or ""
    issues = critic_output.get("issues") or []
    adjustments = critic_output.get("confidence_adjustments") or {}

    with st.container(border=True):
        # Header
        st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding-bottom:0.6rem;margin-bottom:0.65rem;
            border-bottom:1px solid #A78BFA28;">
  <div style="display:flex;align-items:center;gap:0.5rem;">
    <span style="font-size:1.25rem;line-height:1;">🔍</span>
    <span style="font-size:0.8rem;font-weight:700;letter-spacing:0.1em;
                 color:#A78BFA;text-transform:uppercase;">Critic Review</span>
  </div>
</div>""", unsafe_allow_html=True)

        if status != "success":
            st.markdown(
                f'<p style="font-size:0.82rem;color:rgba(232,234,240,0.3);">Critic: {status}</p>',
                unsafe_allow_html=True,
            )
            return

        # Reasoning
        if reasoning:
            st.markdown(
                f'<p style="font-size:0.88rem;line-height:1.75;color:rgba(232,234,240,0.85);margin:0 0 0.9rem;">'
                f'{_html.escape(reasoning)}</p>',
                unsafe_allow_html=True,
            )

        # Overall quality bar
        if quality is not None:
            q_hex = "#34D399" if quality >= 0.7 else "#FCD34D" if quality >= 0.4 else "#F87171"
            _bar("Overall Analysis Quality", quality, q_hex)

        # Confidence adjustments
        if adjustments:
            rows = "".join(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'font-size:0.72rem;padding:2px 0;">'
                f'<span style="color:rgba(232,234,240,0.5);">{AGENT_SHORT.get(agent, agent)}</span>'
                f'<span style="color:#A78BFA;font-weight:700;">{round(v * 100)}%</span>'
                f'</div>'
                for agent, v in adjustments.items()
            )
            st.markdown(f"""
<div style="margin-bottom:0.75rem;">
  <div style="font-size:0.7rem;letter-spacing:0.08em;color:rgba(232,234,240,0.3);
              margin-bottom:0.35rem;font-weight:600;">ADJUSTED CONFIDENCE</div>
  {rows}
</div>""", unsafe_allow_html=True)

        # Top issues by severity
        top_issues = sorted(issues, key=lambda x: x.get("severity", 0), reverse=True)[:4]
        if top_issues:
            issue_html = ""
            for iss in top_issues:
                sev = iss.get("severity", 0)
                sev_color = "#F87171" if sev >= 0.7 else "#FCD34D" if sev >= 0.4 else "#9CA3AF"
                label = ISSUE_TYPE_LABELS.get(iss.get("issue_type", ""), iss.get("issue_type", ""))
                agent_short = AGENT_SHORT.get(iss.get("source_agent", ""), iss.get("source_agent", ""))
                claim = _html.escape((iss.get("claim") or "")[:120])
                issue_html += (
                    f'<div style="padding:0.45rem 0;border-top:1px solid rgba(255,255,255,0.05);">'
                    f'<div style="display:flex;gap:0.4rem;align-items:center;margin-bottom:3px;">'
                    f'<span style="font-size:0.66rem;font-weight:700;color:{sev_color};'
                    f'background:{sev_color}1A;padding:1px 6px;border-radius:100px;">{label}</span>'
                    f'<span style="font-size:0.66rem;color:rgba(232,234,240,0.3);">{agent_short}</span>'
                    f'</div>'
                    f'<p style="margin:0;font-size:0.78rem;color:rgba(232,234,240,0.55);line-height:1.5;">'
                    f'{claim}</p>'
                    f'</div>'
                )
            st.markdown(f"""
<div style="margin-top:0.2rem;">
  <div style="font-size:0.7rem;letter-spacing:0.08em;color:rgba(232,234,240,0.3);
              margin-bottom:0.25rem;font-weight:600;">TOP ISSUES</div>
  {issue_html}
</div>""", unsafe_allow_html=True)


def render_agent_skeleton(agent_name: str) -> None:
    display_name = _INTERNAL_TO_DISPLAY.get(agent_name, agent_name)
    icon    = AGENT_ICONS.get(display_name, "🤖")
    hex_col = AGENT_HEX.get(display_name, "#4A9EFF")

    with st.container(border=True):
        _card_header(display_name, icon, hex_col, status="skipped")
        st.markdown(f"""
<div style="animation:pulse 1.6s ease-in-out infinite;">
  <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:6px;margin-bottom:7px;width:90%;"></div>
  <div style="height:8px;background:rgba(255,255,255,0.04);border-radius:6px;margin-bottom:7px;width:74%;"></div>
  <div style="height:8px;background:rgba(255,255,255,0.03);border-radius:6px;width:58%;"></div>
</div>
<div style="margin-top:1rem;">
  <div style="height:4px;background:rgba(255,255,255,0.06);border-radius:100px;margin-bottom:7px;"></div>
  <div style="height:4px;background:rgba(255,255,255,0.04);border-radius:100px;margin-bottom:7px;"></div>
  <div style="height:4px;background:rgba(255,255,255,0.03);border-radius:100px;"></div>
</div>""", unsafe_allow_html=True)


def render_agent_card(output: AgentDisplayOutput) -> None:
    icon    = AGENT_ICONS.get(output.agent_name, "🤖")
    hex_col = AGENT_HEX.get(output.agent_name, "#4A9EFF")

    with st.container(border=True):
        _card_header(output.agent_name, icon, hex_col, status=output.status)

        if output.status not in ("success", "retry_success"):
            st.markdown(
                f'<p style="font-size:0.82rem;color:rgba(232,234,240,0.3);">Status: {output.status}</p>',
                unsafe_allow_html=True,
            )
            return

        # Response text — rendered as plain HTML paragraph, never as markdown
        import html as _html
        st.markdown(
            f'<p style="font-size:0.88rem;line-height:1.75;color:rgba(232,234,240,0.85);margin:0;">'
            f'{_html.escape(output.response)}</p>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:0.9rem;'></div>", unsafe_allow_html=True)

        # Evidence strength bar
        _evidence_bar(output.evidence_strength)

        # Confidence bar
        conf_hex = (
            "#34D399" if output.confidence >= 0.7 else
            "#FCD34D" if output.confidence >= 0.4 else
            "#F87171"
        )
        _bar("Confidence", output.confidence, conf_hex, output.confidence_label)

        # RAG grounding bar
        rag_hex = (
            "#34D399" if output.rag_grounding_pct >= 0.7 else
            "#FCD34D" if output.rag_grounding_pct >= 0.4 else
            "#F87171"
        )
        _bar("Source Grounding", output.rag_grounding_pct, rag_hex)
