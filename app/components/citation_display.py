import streamlit as st
from app.display.models import CitationDisplay


def render_citations(citations: list[CitationDisplay]) -> None:
    if not citations:
        st.caption("No RAG citations.")
        return
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for c in citations:
            st.markdown(f"- {c.source}")
