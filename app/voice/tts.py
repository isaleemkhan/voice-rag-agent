"""Text-to-speech using ElevenLabs streaming."""

from __future__ import annotations

from typing import Iterator

from elevenlabs.client import ElevenLabs

from app.config import get_settings


def _client() -> ElevenLabs:
    return ElevenLabs(api_key=get_settings().elevenlabs_api_key)


def stream_speech(text: str) -> Iterator[bytes]:
    """Stream MP3 audio chunks for the given text."""
    settings = get_settings()
    audio_stream = _client().text_to_speech.convert(
        voice_id=settings.elevenlabs_voice_id,
        model_id=settings.elevenlabs_model,
        text=text,
        output_format="mp3_44100_128",
    )
    for chunk in audio_stream:
        if chunk:
            yield chunk


def synthesize(text: str) -> bytes:
    """Return the full synthesized audio as a single bytes blob."""
    return b"".join(stream_speech(text))
