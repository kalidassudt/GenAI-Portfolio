"""Streamlit interface for document ingestion, retrieval, and grounded chat."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag_app.assistant import KnowledgeAssistant
from rag_app.config import Settings
from rag_app.documents import load_documents
from rag_app.evaluation import RetrievalCase, evaluate_retrieval
from rag_app.index import build_vector_store

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "sample_docs" / "employee_handbook.txt"
SAMPLE_FILENAME = SAMPLE_PATH.name


def create_assistant(files: list[tuple[str, bytes]]) -> tuple[KnowledgeAssistant, int, str]:
    settings = Settings.from_env()
    documents = load_documents(files)
    vector_store, chunk_count, embedding_provider = build_vector_store(
        documents, settings
    )
    assistant = KnowledgeAssistant(
        vector_store, settings, embedding_provider=embedding_provider
    )
    return assistant, chunk_count, embedding_provider


st.set_page_config(page_title="Knowledge Desk", layout="wide")
st.caption("GENAI + ML PORTFOLIO  /  03")
st.title("Knowledge Desk")
st.markdown(
    "Ask questions about your indexed files and inspect the cited passages "
    "behind each answer."
)

if "assistant" not in st.session_state:
    try:
        with st.spinner("Indexing the sample handbook"):
            assistant, chunk_count, embedding_provider = create_assistant(
                [(SAMPLE_FILENAME, SAMPLE_PATH.read_bytes())]
            )
        st.session_state.assistant = assistant
        st.session_state.chunk_count = chunk_count
        st.session_state.embedding_provider = embedding_provider
        st.session_state.indexed_files = [SAMPLE_FILENAME]
        st.session_state.messages = []
    except Exception as error:
        st.error(f"Could not index the sample document: {error}")
        st.stop()

with st.sidebar:
    st.subheader("Knowledge base")
    st.caption(
        f"{st.session_state.chunk_count} chunks | "
        f"{st.session_state.embedding_provider} embeddings"
    )
    st.write("Indexed: " + ", ".join(st.session_state.indexed_files))
    st.divider()
    uploaded_files = st.file_uploader(
        "Add PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Build index from uploads", disabled=not uploaded_files):
        file_bytes = [(item.name, item.getvalue()) for item in uploaded_files]
        try:
            with st.spinner("Reading documents and building the FAISS index"):
                assistant, chunk_count, embedding_provider = create_assistant(file_bytes)
            st.session_state.assistant = assistant
            st.session_state.chunk_count = chunk_count
            st.session_state.embedding_provider = embedding_provider
            st.session_state.indexed_files = [name for name, _ in file_bytes]
            st.session_state.messages = []
            st.rerun()
        except (ValueError, RuntimeError) as error:
            st.error(str(error))

    if st.button("Evaluate sample retrieval"):
        try:
            evaluation_assistant, _, _ = create_assistant(
                [(SAMPLE_FILENAME, SAMPLE_PATH.read_bytes())]
            )
            cases = [
                RetrievalCase(
                    "How many weeks of parental leave are paid?", {SAMPLE_FILENAME}
                ),
                RetrievalCase("When are travel receipts due?", {SAMPLE_FILENAME}),
                RetrievalCase(
                    "How quickly must a security incident be reported?",
                    {SAMPLE_FILENAME},
                ),
            ]
            report = evaluate_retrieval(evaluation_assistant, cases, k=3)
            st.metric(f"Sample recall@{report.k}", f"{report.recall_at_k:.0%}")
            st.caption(f"{report.evaluated_queries} labeled sample questions")
        except Exception as error:
            st.error(f"Retrieval evaluation failed: {error}")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.session_state.messages:
        transcript_lines = []
        for message in st.session_state.messages:
            transcript_lines.append(f"{message['role'].title()}: {message['content']}")
            for source in message.get("sources", []):
                transcript_lines.append(f"Source: [{source['citation']}]")
        st.download_button(
            "Download conversation",
            "\n\n".join(transcript_lines),
            file_name="knowledge-desk-conversation.txt",
            mime="text/plain",
            use_container_width=True,
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"**[{source['citation']}]** {source['excerpt']}")

st.caption("Try: How many weeks of parental leave are paid?")
question = st.chat_input("Ask about the indexed documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the indexed documents"):
                answer = st.session_state.assistant.ask(question)
            st.markdown(answer.text)
            st.caption(f"Answer mode: {answer.provider}")
            source_records = [
                {
                    "citation": source.citation,
                    "excerpt": source.excerpt,
                    "distance": source.distance,
                }
                for source in answer.sources
            ]
            if source_records:
                with st.expander(f"Sources ({len(source_records)})"):
                    for source in source_records:
                        st.markdown(
                            f"**[{source['citation']}]** {source['excerpt']}"
                        )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer.text,
                    "sources": source_records,
                }
            )
        except (ValueError, RuntimeError) as error:
            st.error(f"Could not answer from the indexed documents: {error}")

st.divider()
st.caption("Portfolio demo | Verify important answers against the cited source documents.")