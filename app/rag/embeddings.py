"""Pinecone ingestion and retrieval using OpenAI embeddings."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from app.config import get_settings

# Dimension of OpenAI text-embedding-3-small
_EMBED_DIM = 1536


def _embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def ensure_index() -> None:
    """Create the Pinecone serverless index if it does not exist."""
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = {idx["name"] for idx in pc.list_indexes()}
    if settings.pinecone_index not in existing:
        pc.create_index(
            name=settings.pinecone_index,
            dimension=_EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )


def get_vector_store() -> PineconeVectorStore:
    """Return a LangChain vector store bound to the Pinecone index."""
    settings = get_settings()
    ensure_index()
    return PineconeVectorStore(
        index_name=settings.pinecone_index,
        embedding=_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
    )


def split_text(text: str, source: str = "inline") -> List[Document]:
    """Chunk raw text into LangChain documents."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.create_documents([text], metadatas=[{"source": source}])


def ingest_text(text: str, source: str = "inline") -> int:
    """Chunk and upsert text into Pinecone. Returns number of chunks added."""
    docs = split_text(text, source=source)
    if not docs:
        return 0
    store = get_vector_store()
    store.add_documents(docs)
    return len(docs)


def get_retriever():
    """Return a retriever configured with the top_k setting."""
    settings = get_settings()
    return get_vector_store().as_retriever(search_kwargs={"k": settings.top_k})
