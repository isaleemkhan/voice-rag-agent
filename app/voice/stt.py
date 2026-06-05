"""Speech-to-text using OpenAI Whisper."""

from __future__ import annotations

import io

from openai import OpenAI

from app.config import get_settings


def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Transcribe raw audio bytes into text using Whisper."""
    settings = get_settings()
    buffer = io.BytesIO(audio_bytes)
    buffer.name = filename
    result = _client().audio.transcriptions.create(
        model=settings.whisper_model,
        file=buffer,
    )
    return (result.text or "").strip()
