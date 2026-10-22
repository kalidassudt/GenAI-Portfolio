"""Streamlit chat interface for the assistant service."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from assistant_app.config import Settings
from assistant_app.service import ChatService

st.set_page_config(page_title="GenAI Assistant", layout="centered")
st.title("GenAI Assistant")
st.caption("LangChain prompt | Ollama or deterministic mock mode")

settings = Settings.from_env()
ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
}
service = ChatService(replace(settings, ollama_enabled=ollama_enabled))

with st.sidebar:
    temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.1)
    max_tokens = st.slider("Maximum response tokens", 64, 2048, 600, 64)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            try:
                result = service.answer(
                    question,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                st.markdown(result.answer)
                st.caption(f"Provider: {result.provider}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.answer}
                )
            except ValueError as error:
                st.error(f"Assistant configuration error: {error}")