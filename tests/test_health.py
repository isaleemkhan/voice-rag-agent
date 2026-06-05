"""Basic API smoke tests that do not require external API keys."""

import os
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _fake_env():
    """Provide dummy env vars so Settings loads without real keys."""
    env = {
        "OPENAI_API_KEY": "test",
        "PINECONE_API_KEY": "test",
        "ELEVENLABS_API_KEY": "test",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        yield


def test_health_endpoint():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
