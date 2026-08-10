import os
import socket
import time
import uuid
from dataclasses import dataclass
import json
from typing import Iterable
from urllib import request
from urllib.error import HTTPError, URLError

from qdrant_client import QdrantClient
from qdrant_client.http import models

from splitter import SplitterConfig, TextSplitter


@dataclass
class RAGConfig:
    top_k: int = 4
    embedding_mode: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_api_url: str | None = None
    embedding_api_key: str | None = None
    embedding_api_model: str | None = "bge-m3"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "Beees"
    embedding_api_timeout_seconds: int = 120
    embedding_batch_size: int = 32
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout_seconds: int = 120


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None):
        base_config = RAGConfig()

        def env_or_default(name: str, default: str | None) -> str | None:
            value = os.getenv(name)
            if value is None:
                return default
            stripped = value.strip()
            if stripped == "":
                return default
            return stripped

        default_config = RAGConfig(
            embedding_mode=env_or_default("EMBEDDING_MODE", base_config.embedding_mode)
            or base_config.embedding_mode,
            embedding_model=env_or_default("EMBEDDING_MODEL", base_config.embedding_model)
            or base_config.embedding_model,
            embedding_api_url=env_or_default("EMBEDDING_API_URL", base_config.embedding_api_url),
            embedding_api_key=env_or_default("EMBEDDING_API_KEY", base_config.embedding_api_key),
            embedding_api_model=env_or_default("EMBEDDING_API_MODEL", base_config.embedding_api_model),
            qdrant_url=env_or_default("QDRANT_URL", base_config.qdrant_url) or base_config.qdrant_url,
            qdrant_api_key=env_or_default("QDRANT_API_KEY", base_config.qdrant_api_key),
            qdrant_collection=env_or_default("QDRANT_COLLECTION", base_config.qdrant_collection)
            or base_config.qdrant_collection,
            embedding_api_timeout_seconds=int(
                env_or_default(
                    "EMBEDDING_API_TIMEOUT",
                    str(base_config.embedding_api_timeout_seconds),
                )
                or base_config.embedding_api_timeout_seconds
            ),
            embedding_batch_size=max(
                1,
                int(
                    env_or_default(
                        "EMBEDDING_BATCH_SIZE",
                        str(base_config.embedding_batch_size),
                    )
                    or base_config.embedding_batch_size
                ),
            ),
            ollama_url=env_or_default("OLLAMA_URL", base_config.ollama_url)
            or base_config.ollama_url,
            ollama_model=env_or_default("OLLAMA_MODEL", base_config.ollama_model)
            or base_config.ollama_model,
            ollama_timeout_seconds=int(
                env_or_default(
                    "OLLAMA_TIMEOUT",
                    str(base_config.ollama_timeout_seconds),
                )
                or base_config.ollama_timeout_seconds
            ),
        )
        self.config = config if config is not None else default_config
        self.config.embedding_mode = self.config.embedding_mode.lower()

        if self.config.embedding_mode not in {"api", "local"}:
            raise RuntimeError("`EMBEDDING_MODE` must be either 'api' or 'local'.")

        self.local_embedder = None
        if self.config.embedding_mode == "api":
            if not self.config.embedding_api_url:
                raise RuntimeError("`EMBEDDING_API_URL` must be set when EMBEDDING_MODE=api.")
            if not self.config.embedding_api_key:
                raise RuntimeError("`EMBEDDING_API_KEY` must be set when EMBEDDING_MODE=api.")
        else:
            sentence_transformers_module = __import__("sentence_transformers")
            SentenceTransformer = getattr(sentence_transformers_module, "SentenceTransformer")
            self.local_embedder = SentenceTransformer(self.config.embedding_model)

        self.splitter = TextSplitter(SplitterConfig())
        self.qdrant = QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            timeout=30,
        )

    def _qdrant_retry(
        self,
        func,
        *args,
        attempts: int = 6,
        delay_seconds: float = 1.0,
        max_delay_seconds: float = 8.0,
        **kwargs,
    ):
        last_error: Exception | None = None
        delay = delay_seconds
        for attempt in range(attempts):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                last_error = error
                if attempt < attempts - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay_seconds)
        raise last_error

    def _collection_exists(self) -> bool:
        collections = self._qdrant_retry(self.qdrant.get_collections).collections
        return any(c.name == self.config.qdrant_collection for c in collections)

    def has_data(self) -> bool:
        if not self._collection_exists():
            return False
        return self._qdrant_retry(self.qdrant.count, collection_name=self.config.qdrant_collection).count > 0

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_exists():
            return
        self._qdrant_retry(
            self.qdrant.create_collection,
            collection_name=self.config.qdrant_collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _embed_with_api(self, texts: list[str]) -> list[list[float]]:
        if not self.config.embedding_api_url:
            raise RuntimeError("Embedding API URL is not configured.")

        model_name = self.config.embedding_api_model
        payload = {
            "model": model_name,
            "input": texts,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.embedding_api_key:
            headers["Authorization"] = f"Bearer {self.config.embedding_api_key}"

        req = request.Request(
            url=self.config.embedding_api_url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.embedding_api_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Embedding API error ({error.code}): {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Embedding API connection failed: {error}") from error
        except (TimeoutError, socket.timeout) as error:
            raise RuntimeError(
                "Embedding API request timed out. "
                f"Try lowering EMBEDDING_BATCH_SIZE (current: {self.config.embedding_batch_size}) "
                f"or increasing EMBEDDING_API_TIMEOUT (current: {self.config.embedding_api_timeout_seconds})."
            ) from error

        parsed = json.loads(raw)

        items: list
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            for key in ("embeddings", "data", "results", "items"):
                candidate = parsed.get(key)
                if isinstance(candidate, list):
                    items = candidate
                    break
            else:
                raise RuntimeError(
                    "Embedding API response must be a JSON array or an object with one of: embeddings, data, results, items."
                )
        else:
            raise RuntimeError("Embedding API response must be valid JSON array/object.")

        embeddings: list[list[float]] = []
        for item in items:
            if isinstance(item, list):
                embeddings.append([float(value) for value in item])
                continue

            if isinstance(item, dict):
                vector = item.get("embedding")
                if vector is None:
                    vector = item.get("vector")
                if isinstance(vector, list):
                    embeddings.append([float(value) for value in vector])
                    continue

            raise RuntimeError(
                "Each item in embedding API response must be a vector array or object with `embedding`."
            )

        if len(embeddings) != len(texts):
            raise RuntimeError("Embedding API returned invalid embedding count.")

        return embeddings

    def _embed_with_local_model(self, texts: list[str]) -> list[list[float]]:
        if self.local_embedder is None:
            raise RuntimeError("Local embedding model is not initialized.")
        vectors = self.local_embedder.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def _embed(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []

        if self.config.embedding_mode == "local":
            return self._embed_with_local_model(text_list)

        batch_size = self.config.embedding_batch_size
        all_embeddings: list[list[float]] = []
        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]
            all_embeddings.extend(self._embed_with_api(batch))

        return all_embeddings

    def _is_nan_encode_error(self, error: RuntimeError) -> bool:
        message = str(error)
        return "unsupported value: NaN" in message or "nan" in message.lower()

    def _embed_ingest_chunks(self, chunks: list[str]) -> tuple[list[str], list[list[float]], int]:
        if self.config.embedding_mode == "local":
            return chunks, self._embed(chunks), 0

        batch_size = self.config.embedding_batch_size
        kept_chunks: list[str] = []
        kept_embeddings: list[list[float]] = []
        skipped_chunks = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            try:
                batch_embeddings = self._embed_with_api(batch)
                kept_chunks.extend(batch)
                kept_embeddings.extend(batch_embeddings)
                continue
            except RuntimeError as error:
                if not self._is_nan_encode_error(error):
                    raise

            for chunk in batch:
                try:
                    embedding = self._embed_with_api([chunk])[0]
                    kept_chunks.append(chunk)
                    kept_embeddings.append(embedding)
                except RuntimeError as error:
                    if self._is_nan_encode_error(error):
                        skipped_chunks += 1
                        continue
                    raise

        return kept_chunks, kept_embeddings, skipped_chunks

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def ingest_text(self, text: str, source: str = "inline") -> dict:
        chunks = self.splitter.chunk_text(text)
        if not chunks:
            return {"chunks_added": 0, "message": "No chunks created from input text."}

        kept_chunks, embeddings, skipped_chunks = self._embed_ingest_chunks(chunks)
        if not kept_chunks:
            raise RuntimeError(
                "All chunks failed embedding. Your embedding API returned invalid NaN output for all chunks."
            )

        self._ensure_collection(vector_size=len(embeddings[0]))

        points = []
        for index, chunk in enumerate(kept_chunks):
            point_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embeddings[index],
                    payload={
                        "text": chunk,
                        "source": source,
                        "chunk_index": index,
                    },
                )
            )

        self._qdrant_retry(
            self.qdrant.upsert,
            collection_name=self.config.qdrant_collection,
            points=points,
            wait=True,
        )

        return {
            "chunks_added": len(kept_chunks),
            "chunks_skipped": skipped_chunks,
            "chunks_total": len(chunks),
            "source": source,
        }

    def ingest_file(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return self.ingest_text(text, source=file_path)

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.config.top_k
        if not self._collection_exists():
            return []

        query_embedding = self._embed([question])[0]

        hits = self._qdrant_retry(
            self.qdrant.query_points,
            collection_name=self.config.qdrant_collection,
            query=query_embedding,
            limit=k,
            with_payload=True,
        ).points

        contexts = [
            {
                "text": (hit.payload or {}).get("text", ""),
                "metadata": {
                    "source": (hit.payload or {}).get("source"),
                    "chunk_index": (hit.payload or {}).get("chunk_index"),
                },
                "score": float(hit.score),
            }
            for hit in hits
        ]

        return contexts

    def _fallback_answer(self, question: str, contexts: list[dict]) -> str:
        if not contexts:
            return "I could not find relevant context in the knowledge base."

        context_preview = "\n\n".join(
            [
                f"[{index + 1}] {context['text'][:300]}"
                for index, context in enumerate(contexts[:3])
            ]
        )
        return (
            "No external LLM configured, so this is an extractive response.\n"
            f"Question: {question}\n"
            f"Best matching context:\n{context_preview}"
        )

    def _generate_with_ollama(self, question: str, contexts: list[dict]) -> str:
        context_block = "\n\n".join(
            f"[{index + 1}] {context['text']}" for index, context in enumerate(contexts)
        )
        prompt = (
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say you don't know.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.config.ollama_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.ollama_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama error ({error.code}): {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Ollama connection failed: {error}") from error
        except (TimeoutError, socket.timeout) as error:
            raise RuntimeError("Ollama request timed out.") from error

        parsed = json.loads(raw)
        answer = parsed.get("response")
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")
        return answer.strip()

    def answer_question(
        self, question: str, top_k: int | None = None, use_llm: bool = True
    ) -> dict:
        contexts = self.retrieve(question=question, top_k=top_k)

        if use_llm and contexts:
            try:
                answer = self._generate_with_ollama(question=question, contexts=contexts)
                return {"answer": answer, "contexts": contexts, "mode": "llm"}
            except RuntimeError as error:
                return {
                    "answer": self._fallback_answer(question=question, contexts=contexts),
                    "contexts": contexts,
                    "mode": "extractive",
                    "llm_error": str(error),
                }

        answer = self._fallback_answer(question=question, contexts=contexts)
        return {"answer": answer, "contexts": contexts, "mode": "extractive"}
