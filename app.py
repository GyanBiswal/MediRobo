"""
Medical Information RAG Chatbot — Streamlit UI.

Educational medical information assistant. Not a diagnostic tool.
Answers are grounded in retrieved medical reference documents only.
"""

from pathlib import Path
import streamlit as st

from src.config import VECTORSTORE_DIR
from src.rag import build_rag_chain, answer_question
from src.vectorstore import build_vectorstore


st.set_page_config(page_title="Medical Info Assistant", page_icon="🩺")


@st.cache_resource
def get_chain():
    """
    Build the RAG chain once and cache it across reruns/sessions.
    If no FAISS index exists yet (e.g. first run on a fresh deployment
    where only the source PDFs were committed), build it automatically.
    """
    index_path = Path(VECTORSTORE_DIR) / "index.faiss"
    if not index_path.exists():
        with st.spinner("First-time setup: building the knowledge base from source documents..."):
            build_vectorstore()

    return build_rag_chain()


def format_sources(source_documents):
    """Deduplicate and format source citations for display."""
    seen = set()
    lines = []
    for doc in source_documents:
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        page = doc.metadata.get("page", "unknown")
        key = (source, page)
        if key not in seen:
            lines.append(f"- **{source}**, page {page}")
            seen.add(key)
    return "\n".join(lines) if lines else "No sources retrieved."


st.title("🩺 Medical Information Assistant")
st.caption(
    "Educational tool only — grounded in a small set of reference documents. "
    "Not a diagnostic system and not a substitute for professional medical advice."
)

chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                st.markdown(message["sources"])

user_question = st.chat_input("Ask a health-related question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Looking through medical documents..."):
            result = answer_question(user_question, chain)
            answer = result["answer"]
            sources_text = format_sources(result["source_documents"])

        st.markdown(answer)
        with st.expander("Sources"):
            st.markdown(sources_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_text,
    })