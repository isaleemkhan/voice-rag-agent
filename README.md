# voice-rag-agent

**Speak a question. Get an AI answer spoken back.**

A production-ready voice RAG pipeline: transcribes speech, retrieves context from your documents, and responds with a natural voice in under 3 seconds.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

---

## What It Does

Most RAG demos are text-in, text-out. This project makes RAG fully conversational:

1. **Listen** - Records audio from the user (or accepts an audio file)
2. **Transcribe** - OpenAI Whisper converts speech to text
3. **Retrieve** - LangChain queries Pinecone vector DB for relevant document chunks
4. **Generate** - GPT-4o generates a grounded answer from retrieved context
5. **Speak** - ElevenLabs TTS converts the answer to natural speech and streams it back

Real use cases: internal knowledge bots, voice-enabled customer support, document Q&A over PDFs.

---

## Architecture

```
User Voice Input --> Whisper STT --> Transcribed Query
                                           |
                                    LangChain RAG
                                    Pinecone + GPT-4o
                                           |
                                    ElevenLabs TTS
                                           |
                               FastAPI (REST + WebSocket)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Speech-to-Text | OpenAI Whisper (whisper-1) |
| Vector Store | Pinecone (serverless) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4o |
| Text-to-Speech | ElevenLabs (streaming) |
| Orchestration | LangChain |
| API Server | FastAPI + Uvicorn |

---

## Project Structure

```
voice-rag-agent/
├── app/
│   ├── main.py              # FastAPI app and routes
│   ├── rag/
│   │   ├── pipeline.py      # LangChain RAG chain
│   │   ├── embeddings.py    # Pinecone ingestion and retrieval
│   │   └── prompts.py       # System prompts
│   ├── voice/
│   │   ├── stt.py           # Whisper transcription
│   │   └── tts.py           # ElevenLabs streaming TTS
│   └── config.py            # Settings and env vars
├── docs/                    # Drop your documents here
├── scripts/
│   └── ingest.py            # Ingest docs into Pinecone
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/isaleemkhan/voice-rag-agent.git
cd voice-rag-agent
pip install -r requirements.txt
cp .env.example .env
python scripts/ingest.py
uvicorn app.main:app --reload
```

Configure `.env` with your keys:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
ELEVENLABS_API_KEY=...
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | /ask | Audio in, audio answer out |
| POST | /ask/text | Text in, audio answer out |
| POST | /ingest | Add a document to the knowledge base |
| GET | /health | Health check |
| WS | /ws/ask | Real-time streaming WebSocket |

---

## License

MIT - built by [Saleem Khan](https://github.com/isaleemkhan)# voice-rag-agent
Voice-enabled RAG assistant — speak a question, get an AI answer spoken back. Built with Whisper, LangChain, Pinecone, FastAPI, and ElevenLabs.
