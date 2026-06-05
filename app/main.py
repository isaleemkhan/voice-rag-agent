"""FastAPI application: REST + WebSocket endpoints for the voice RAG agent."""

from __future__ import annotations

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag.embeddings import ingest_text
from app.rag.pipeline import answer_question
from app.voice.stt import transcribe
from app.voice.tts import stream_speech

app = FastAPI(
    title="Voice RAG Agent",
    description="Speak a question, get an AI answer spoken back.",
    version="0.1.0",
)


class TextQuery(BaseModel):
    question: str


class IngestRequest(BaseModel):
    text: str
    source: str = "api"


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {"status": "ok"}


@app.post("/ask")
async def ask(file: UploadFile = File(...)) -> StreamingResponse:
    """Audio in, audio answer out."""
    audio_bytes = await file.read()
    question = transcribe(audio_bytes, filename=file.filename or "audio.wav")
    answer = answer_question(question)
    return StreamingResponse(
        stream_speech(answer),
        media_type="audio/mpeg",
        headers={
            "X-Question": question,
            "X-Answer": answer,
        },
    )


@app.post("/ask/text")
async def ask_text(payload: TextQuery) -> StreamingResponse:
    """Text in, audio answer out."""
    answer = answer_question(payload.question)
    return StreamingResponse(
        stream_speech(answer),
        media_type="audio/mpeg",
        headers={"X-Answer": answer},
    )


@app.post("/ingest")
async def ingest(payload: IngestRequest) -> dict:
    """Add a document to the knowledge base."""
    chunks = ingest_text(payload.text, source=payload.source)
    return {"ingested_chunks": chunks, "source": payload.source}


@app.websocket("/ws/ask")
async def ws_ask(websocket: WebSocket) -> None:
    """Real-time streaming: receive a text question, stream audio chunks back."""
    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            answer = answer_question(question)
            await websocket.send_json({"type": "answer", "text": answer})
            for chunk in stream_speech(answer):
                await websocket.send_bytes(chunk)
            await websocket.send_json({"type": "end"})
    except WebSocketDisconnect:
        return
