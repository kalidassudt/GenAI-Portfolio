"""Streamlit chat client for the FastAPI assistant."""

from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="GenAI Assistant", layout="centered")
st.title("GenAI Assistant")
st.caption("LangChain prompt | Ollama or deterministic mock mode")

with st.sidebar:
    api_url = st.text_input(
        "Assistant API", os.getenv("GENAI_API_URL", "http://127.0.0.1:8000")
    )
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
                response = requests.post(
                    f"{api_url.rstrip('/')}/chat",
                    json={
                        "question": question,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=45,
                )
                response.raise_for_status()
                payload = response.json()
                answer = payload["answer"]
                st.markdown(answer)
                st.caption(f"Provider: {payload['provider']}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except (requests.RequestException, KeyError, ValueError) as error:
                st.error(f"Assistant API unavailable: {error}")