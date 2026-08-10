import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

from ingest_dataset import discover_files, extract_text
from rag_pipeline import RAGPipeline


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


STATIC_DIR = Path(__file__).parent / "static"
AUTO_INGEST_ENABLED = _env_bool("AUTO_INGEST_ENABLED", "true")
AUTO_INGEST_DIR = os.getenv("AUTO_INGEST_DIR", "dataset")
HEARTBEAT_ENABLED = _env_bool("QDRANT_HEARTBEAT_ENABLED", "true")
HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("QDRANT_HEARTBEAT_INTERVAL_SECONDS", "15"))

pipeline: RAGPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        with _pipeline_lock:
            if pipeline is None:
                pipeline = RAGPipeline()
    return pipeline


def _wait_for_qdrant(rag: RAGPipeline, attempts: int = 60, interval_seconds: float = 5.0) -> bool:
    for attempt in range(attempts):
        try:
            rag.has_data()
            return True
        except Exception as error:
            if attempt == 0:
                print(f"[auto-ingest] qdrant not reachable yet, retrying: {error}")
            elif attempt % 6 == 0:
                print(f"[auto-ingest] still waiting for qdrant (attempt {attempt}/{attempts}): {error}")
            time.sleep(interval_seconds)
    return False


def _auto_ingest_worker() -> None:
    try:
        rag = get_pipeline()
    except Exception as error:
        print(f"[auto-ingest] pipeline init failed, skipping: {error}")
        return

    if not _wait_for_qdrant(rag):
        print("[auto-ingest] qdrant never became reachable, giving up.")
        return

    if rag.has_data():
        print(f"[auto-ingest] collection '{rag.config.qdrant_collection}' already has data, skipping.")
        return

    root = Path(AUTO_INGEST_DIR)
    if not root.exists():
        print(f"[auto-ingest] directory not found: {root}, skipping.")
        return

    files = discover_files(root)
    if not files:
        print(f"[auto-ingest] no supported files found under {root}.")
        return

    print(f"[auto-ingest] found {len(files)} file(s) under {root}, ingesting on first startup...")
    ingested = 0
    for path in files:
        try:
            text = extract_text(path)
            if not text.strip():
                continue
            result = rag.ingest_text(text=text, source=str(path))
            ingested += 1
            print(f"[auto-ingest] ingested {path} -> {result.get('chunks_added', 0)} chunks")
        except Exception as error:
            print(f"[auto-ingest] failed on {path}: {error}")

    print(f"[auto-ingest] done: {ingested}/{len(files)} file(s) ingested.")


def _qdrant_heartbeat_worker(interval_seconds: float) -> None:
    """Periodically pings Qdrant and logs only on reachability transitions.

    Keeps the connection path warm and gives us a timestamped record of
    outages on shared/unreliable infrastructure, without spamming logs.
    """
    last_ok = True
    while True:
        try:
            rag = get_pipeline()
            rag.qdrant.get_collections()
            if not last_ok:
                print("[heartbeat] qdrant reachable again")
            last_ok = True
        except Exception as error:
            if last_ok:
                print(f"[heartbeat] qdrant unreachable: {error}")
            last_ok = False
        time.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if AUTO_INGEST_ENABLED:
        threading.Thread(target=_auto_ingest_worker, daemon=True).start()
    if HEARTBEAT_ENABLED:
        threading.Thread(
            target=_qdrant_heartbeat_worker, args=(HEARTBEAT_INTERVAL_SECONDS,), daemon=True
        ).start()
    yield


app = FastAPI(title="RAG Server", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class IngestFileRequest(BaseModel):
    file_path: str = Field(..., description="Path to UTF-8 text file")


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field(default="inline")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    use_llm: bool = Field(default=True, description="Generate the answer with Ollama; falls back to extractive if false or on error")


class DebugEmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of texts to embed")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest/file")
def ingest_file(request: IngestFileRequest) -> dict:
    try:
        return get_pipeline().ingest_file(request.file_path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="Input file must be UTF-8 encoded text.",
        ) from error


@app.post("/ingest/text")
def ingest_text(request: IngestTextRequest) -> dict:
    return get_pipeline().ingest_text(text=request.text, source=request.source)


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    try:
        reader = PdfReader(file.file)
    except Exception as error:  # pypdf raises various error types for malformed PDFs
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {error}") from error

    text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

    return get_pipeline().ingest_text(text=text, source=file.filename)


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return get_pipeline().answer_question(
        question=request.question, top_k=request.top_k, use_llm=request.use_llm
    )


@app.post("/debug/embed")
def debug_embed(request: DebugEmbedRequest) -> dict:
    try:
        embeddings = get_pipeline().embed_texts(request.texts)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {
        "count": len(embeddings),
        "dimension": 0 if not embeddings else len(embeddings[0]),
        "embeddings": embeddings,
    }
