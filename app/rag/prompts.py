"""System and answer prompts for the RAG chain."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are a helpful voice assistant answering questions strictly from the "
    "provided context. Follow these rules:\n"
    "1. Answer only using the context below. If the context does not contain "
    "the answer, say you don't have that information.\n"
    "2. Keep answers concise and conversational — they will be read aloud.\n"
    "3. Avoid markdown, bullet points, code blocks, or special characters.\n"
    "4. Prefer one or two short spoken sentences unless more detail is required."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer (spoken, concise):",
        ),
    ]
)
