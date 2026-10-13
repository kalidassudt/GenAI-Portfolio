"""Prompt templates and response-format instructions."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a precise and helpful portfolio demonstration assistant.
Answer the user's question directly, distinguish facts from uncertainty, and
never claim access to private systems or information not present in the chat.
Use concise language. If the request lacks needed context, ask one clear question."""

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "{system_prompt}"), ("human", "{question}")]
)


def build_messages(question: str):
    """Build LangChain messages using the application system prompt."""
    return CHAT_PROMPT.invoke(
        {"system_prompt": SYSTEM_PROMPT, "question": question}
    ).to_messages()