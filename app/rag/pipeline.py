"""LangChain RAG chain: retrieve context, then generate a grounded answer."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.rag.embeddings import get_retriever
from app.rag.prompts import ANSWER_PROMPT


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


@lru_cache
def _build_chain():
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )
    retriever = get_retriever()

    return (
        RunnableParallel(
            context=retriever | _format_docs,
            question=RunnablePassthrough(),
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )


def answer_question(question: str) -> str:
    """Run the RAG chain and return a spoken-style answer string."""
    question = (question or "").strip()
    if not question:
        return "I didn't catch a question. Could you please repeat that?"
    chain = _build_chain()
    return chain.invoke(question).strip()
