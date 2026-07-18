"""Pipeline B: Streamlit chat UI tying retrieve.py + llm.py together.

Run: streamlit run src/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import answer_question
from retrieve import TOP_K, retrieve

st.set_page_config(page_title="FIFA Rules RAG Assistant", page_icon="⚽")

st.title("⚽ FIFA Rules RAG Assistant")
st.caption(
    "Ask about the IFAB Laws of the Game or FIFA World Cup 2026 regulations. "
    "Answers are grounded in the official documents and cite the source and page. "
    "Not affiliated with FIFA or IFAB — not an official ruling."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(chunks: list[dict]):
    with st.expander("Sources"):
        for chunk in chunks:
            label = chunk["law"] or chunk["source"]
            st.markdown(f"- **{label}** — {chunk['source']}, p.{chunk['page']}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask a question about the Laws of the Game...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the rulebook..."):
            chunks = retrieve(question, top_k=TOP_K)

        with st.spinner("Asking Claude..."):
            try:
                answer = answer_question(question, chunks)
            except Exception as exc:
                answer = f"Something went wrong calling Claude: {exc}"

        st.markdown(answer)
        render_sources(chunks)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })
