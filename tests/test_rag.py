"""Unit test for the RAG chain using mocked retrieval and LLM."""

import os
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _fake_env():
    env = {
        "OPENAI_API_KEY": "test",
        "PINECONE_API_KEY": "test",
        "ELEVENLABS_API_KEY": "test",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        yield


def test_answer_empty_question():
    from app.rag.pipeline import answer_question

    result = answer_question("   ")
    assert "repeat" in result.lower()
