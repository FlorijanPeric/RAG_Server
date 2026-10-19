import importlib.util
import os
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass
import json
from typing import Iterable
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.http import models

from bm25 import BM25Index
from splitter import SplitterConfig, TextSplitter


@dataclass
class RAGConfig:
    top_k: int = 4
    embedding_mode: str = "api"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_api_url: str | None = None
    embedding_api_key: str | None = None
    embedding_api_model: str | None = "bge-m3"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "Beees"
    embedding_api_timeout_seconds: int = 120
    embedding_batch_size: int = 32
    # LLM_MODE: "ollama" (Ollama server at OLLAMA_URL), "api" (OpenAI-compatible
    # chat completions endpoint at LLM_API_URL) or "local" (HuggingFace model
    # loaded in-process with transformers).
    llm_mode: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:4b"
    llm_api_url: str | None = None
    llm_api_key: str | None = None
    llm_api_model: str | None = None
    llm_local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_max_new_tokens: int = 512
    llm_timeout_seconds: int = 120
    # Ollama only: let "thinking" models (qwen3, ...) reason before answering.
    # Much slower; off by default.
    llm_think: bool = False
    # RETRIEVAL_MODE: "hybrid" (vector search + BM25 keywords, fused), "vector" or "bm25".
    retrieval_mode: str = "hybrid"
    # RERANK_MODE: "off", "local" (cross-encoder loaded in-process with
    # sentence-transformers) or "api" (a /rerank endpoint at RERANK_API_URL).
    rerank_mode: str = "off"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_api_url: str | None = None
    rerank_api_key: str | None = None
    # How many candidates hybrid search / the reranker look at before keeping top_k.
    rerank_candidates: int = 20


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
            top_k=max(1, int(env_or_default("TOP_K", str(base_config.top_k)) or base_config.top_k)),
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
            llm_mode=env_or_default("LLM_MODE", base_config.llm_mode) or base_config.llm_mode,
            ollama_url=env_or_default("OLLAMA_URL", base_config.ollama_url)
            or base_config.ollama_url,
            ollama_model=env_or_default("OLLAMA_MODEL", base_config.ollama_model)
            or base_config.ollama_model,
            llm_api_url=env_or_default("LLM_API_URL", base_config.llm_api_url),
            llm_api_key=env_or_default("LLM_API_KEY", base_config.llm_api_key),
            llm_api_model=env_or_default("LLM_API_MODEL", base_config.llm_api_model),
            llm_local_model=env_or_default("LLM_LOCAL_MODEL", base_config.llm_local_model)
            or base_config.llm_local_model,
            llm_max_new_tokens=int(
                env_or_default("LLM_MAX_NEW_TOKENS", str(base_config.llm_max_new_tokens))
                or base_config.llm_max_new_tokens
            ),
            llm_think=(env_or_default("LLM_THINK", "false") or "false").lower() in {"1", "true", "yes"},
            retrieval_mode=env_or_default("RETRIEVAL_MODE", base_config.retrieval_mode)
            or base_config.retrieval_mode,
            rerank_mode=env_or_default("RERANK_MODE", base_config.rerank_mode) or base_config.rerank_mode,
            rerank_model=env_or_default("RERANK_MODEL", base_config.rerank_model) or base_config.rerank_model,
            rerank_api_url=env_or_default("RERANK_API_URL", base_config.rerank_api_url),
            rerank_api_key=env_or_default("RERANK_API_KEY", base_config.rerank_api_key),
            rerank_candidates=max(
                1,
                int(
                    env_or_default("RERANK_CANDIDATES", str(base_config.rerank_candidates))
                    or base_config.rerank_candidates
                ),
            ),
            llm_timeout_seconds=int(
                env_or_default(
                    "LLM_TIMEOUT",
                    env_or_default("OLLAMA_TIMEOUT", str(base_config.llm_timeout_seconds)),
                )
                or base_config.llm_timeout_seconds
            ),
        )
        self.config = config if config is not None else default_config
        self.config.embedding_mode = self.config.embedding_mode.lower()
        self.config.llm_mode = self.config.llm_mode.lower()
        self.config.retrieval_mode = self.config.retrieval_mode.lower()
        self.config.rerank_mode = self.config.rerank_mode.lower()

        if self.config.embedding_mode not in {"api", "local"}:
            raise RuntimeError("`EMBEDDING_MODE` must be either 'api' or 'local'.")
        if self.config.llm_mode not in {"ollama", "api", "local"}:
            raise RuntimeError("`LLM_MODE` must be one of 'ollama', 'api' or 'local'.")
        if self.config.llm_mode == "api":
            if not self.config.llm_api_url:
                raise RuntimeError("`LLM_API_URL` must be set when LLM_MODE=api.")
            if not self.config.llm_api_model:
                raise RuntimeError("`LLM_API_MODEL` must be set when LLM_MODE=api.")
        if self.config.retrieval_mode not in {"hybrid", "vector", "bm25"}:
            raise RuntimeError("`RETRIEVAL_MODE` must be one of 'hybrid', 'vector' or 'bm25'.")
        if self.config.rerank_mode not in {"off", "local", "api"}:
            raise RuntimeError("`RERANK_MODE` must be one of 'off', 'local' or 'api'.")
        if self.config.rerank_mode == "api" and not self.config.rerank_api_url:
            # Empty URL = the reranker in the models container, next to Ollama
            # (same host as OLLAMA_URL, port 8001).
            ollama = urlparse(self.config.ollama_url)
            self.config.rerank_api_url = f"{ollama.scheme}://{ollama.hostname}:8001/rerank"
        if self.config.rerank_mode == "local" and importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError(
                "RERANK_MODE=local needs sentence-transformers: install requirements-local.txt "
                "(Docker: set LOCAL_MODELS=true in .env and rebuild)."
            )

        # The local LLM is loaded lazily on first use (it can take a while and a lot
        # of memory), so ingestion-only runs never pay for it.
        self._local_llm = None
        self._local_llm_lock = threading.Lock()
        self._vector_size: int | None = None
        # BM25 index over all chunks in the collection, rebuilt when the chunk count changes.
        self._bm25: BM25Index | None = None
        self._bm25_points: list = []
        self._bm25_count = -1
        self._bm25_lock = threading.Lock()
        self._reranker = None
        self._reranker_lock = threading.Lock()

        self.local_embedder = None
        if self.config.embedding_mode == "api":
            if not self.config.embedding_api_url:
                # Empty URL = the embedding model on the same Ollama as the LLM
                # (the bundled `models` container by default).
                self.config.embedding_api_url = f"{self.config.ollama_url.rstrip('/')}/api/embed"
        else:
            try:
                sentence_transformers_module = __import__("sentence_transformers")
            except ImportError as error:
                raise RuntimeError(
                    "EMBEDDING_MODE=local needs sentence-transformers: install requirements-local.txt "
                    "(Docker: set LOCAL_MODELS=true in .env and rebuild)."
                ) from error
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

    def _collection_vector_size(self) -> int | None:
        info = self._qdrant_retry(
            self.qdrant.get_collection, collection_name=self.config.qdrant_collection
        )
        vectors = info.config.params.vectors
        return getattr(vectors, "size", None)

    def embedding_dimension(self) -> int:
        if self._vector_size is None:
            self._vector_size = len(self._embed(["dimension probe"])[0])
        return self._vector_size

    def check_collection_compatible(self) -> None:
        """Raises if the existing collection was built with a different embedder."""
        if not self._collection_exists():
            return
        stored = self._collection_vector_size()
        current = self.embedding_dimension()
        if stored is not None and stored != current:
            raise RuntimeError(
                f"Collection '{self.config.qdrant_collection}' stores {stored}-dim vectors, but the "
                f"current embedder produces {current}-dim vectors. The embedding model changed - "
                "re-initialize with `scripts/init.sh --reset` (or use a different QDRANT_COLLECTION)."
            )

    def reset_collection(self) -> bool:
        if not self._collection_exists():
            return False
        self._qdrant_retry(self.qdrant.delete_collection, collection_name=self.config.qdrant_collection)
        self._bm25 = None
        return True

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_exists():
            self._vector_size = vector_size
            self.check_collection_compatible()
            return
        self._qdrant_retry(
            self.qdrant.create_collection,
            collection_name=self.config.qdrant_collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _post_json(
        self, url: str, payload: dict, api_key: str | None, timeout: int, label: str
    ) -> dict | list:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = request.Request(url=url, data=body, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"{label} error ({error.code}): {detail}") from error
        except URLError as error:
            raise RuntimeError(f"{label} connection failed ({url}): {error}") from error
        except (TimeoutError, socket.timeout) as error:
            raise RuntimeError(f"{label} request timed out after {timeout}s.") from error

        return json.loads(raw)

    def _embed_with_api(self, texts: list[str]) -> list[list[float]]:
        if not self.config.embedding_api_url:
            raise RuntimeError("Embedding API URL is not configured.")

        payload = {
            "model": self.config.embedding_api_model,
            "input": texts,
        }
        try:
            parsed = self._post_json(
                self.config.embedding_api_url,
                payload,
                self.config.embedding_api_key,
                self.config.embedding_api_timeout_seconds,
                "Embedding API",
            )
        except RuntimeError as error:
            if "timed out" in str(error):
                raise RuntimeError(
                    f"{error} Try lowering EMBEDDING_BATCH_SIZE (current: "
                    f"{self.config.embedding_batch_size}) or increasing EMBEDDING_API_TIMEOUT."
                ) from error
            raise

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
        self._bm25 = None

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

    # Reciprocal rank fusion constant: score = sum(1 / (RRF_K + rank)) over the rankings.
    RRF_K = 60

    @staticmethod
    def _context_from_payload(payload: dict | None, score: float) -> dict:
        payload = payload or {}
        return {
            "text": payload.get("text", ""),
            "metadata": {
                "source": payload.get("source"),
                "chunk_index": payload.get("chunk_index"),
            },
            "score": score,
        }

    def _get_bm25(self) -> tuple[BM25Index | None, list]:
        """BM25 index over every chunk in the collection (built on first use, rebuilt
        whenever the number of chunks changes, e.g. after an ingest from another process)."""
        count = self._qdrant_retry(self.qdrant.count, collection_name=self.config.qdrant_collection).count
        with self._bm25_lock:
            if self._bm25 is not None and count == self._bm25_count:
                return self._bm25, self._bm25_points
            points = []
            offset = None
            while True:
                batch, offset = self._qdrant_retry(
                    self.qdrant.scroll,
                    collection_name=self.config.qdrant_collection,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points.extend(batch)
                if offset is None:
                    break
            self._bm25 = BM25Index([(point.payload or {}).get("text", "") for point in points]) if points else None
            self._bm25_points = points
            self._bm25_count = count
            return self._bm25, self._bm25_points

    def _vector_search(self, question: str, limit: int) -> list[tuple[object, dict]]:
        query_embedding = self._embed([question])[0]
        if self._vector_size is None:
            self._vector_size = len(query_embedding)
            self.check_collection_compatible()

        hits = self._qdrant_retry(
            self.qdrant.query_points,
            collection_name=self.config.qdrant_collection,
            query=query_embedding,
            limit=limit,
            with_payload=True,
        ).points
        return [(hit.id, self._context_from_payload(hit.payload, float(hit.score))) for hit in hits]

    def _keyword_search(self, question: str, limit: int) -> list[tuple[object, dict]]:
        index, points = self._get_bm25()
        if index is None:
            return []
        return [
            (points[position].id, self._context_from_payload(points[position].payload, score))
            for position, score in index.search(question, limit)
        ]

    def _get_reranker(self):
        with self._reranker_lock:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.config.rerank_model, max_length=512)
            return self._reranker

    def _rerank_scores(self, question: str, texts: list[str]) -> list[float]:
        if self.config.rerank_mode == "local":
            scores = self._get_reranker().predict([(question, text) for text in texts])
            return [float(score) for score in scores]

        # Cohere/Jina-style /rerank (llama.cpp server, Infinity, vLLM, ...) takes
        # "documents"; Hugging Face TEI takes "texts". Send both.
        payload = {"query": question, "documents": texts, "texts": texts}
        if self.config.rerank_model:
            payload["model"] = self.config.rerank_model
        data = self._post_json(
            self.config.rerank_api_url,
            payload,
            self.config.rerank_api_key,
            self.config.embedding_api_timeout_seconds,
            "Rerank API",
        )
        results = data.get("results", data.get("data")) if isinstance(data, dict) else data
        scores = [float("-inf")] * len(texts)
        for item in results or []:
            score = item.get("relevance_score", item.get("score"))
            if score is not None and 0 <= int(item.get("index", -1)) < len(texts):
                scores[int(item["index"])] = float(score)
        return scores

    def warm_up_reranker(self) -> None:
        if self.config.rerank_mode == "local":
            self._get_reranker()

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.config.top_k
        if not self._collection_exists():
            return []

        mode = self.config.retrieval_mode
        rerank = self.config.rerank_mode != "off"
        # Hybrid fusion and the reranker both need a wider candidate pool than top_k.
        pool = max(k, self.config.rerank_candidates) if (rerank or mode == "hybrid") else k

        rankings = []
        if mode in {"vector", "hybrid"}:
            rankings.append(self._vector_search(question, pool))
        if mode in {"bm25", "hybrid"}:
            rankings.append(self._keyword_search(question, pool))

        if len(rankings) == 1:
            candidates = [context for _, context in rankings[0]]
        else:
            fused: dict[object, dict] = {}
            for ranking in rankings:
                for rank, (point_id, context) in enumerate(ranking):
                    entry = fused.setdefault(point_id, {**context, "score": 0.0})
                    entry["score"] += 1.0 / (self.RRF_K + rank + 1)
            candidates = sorted(fused.values(), key=lambda context: -context["score"])[:pool]

        if rerank and len(candidates) > 1:
            try:
                scores = self._rerank_scores(question, [context["text"] for context in candidates])
                for context, score in zip(candidates, scores):
                    context["score"] = score
                candidates.sort(key=lambda context: -context["score"])
            except Exception as error:
                # A broken reranker must not break answering: keep the fused order.
                print(f"[retrieve] reranker failed, using unreranked order: {error}")

        return candidates[:k]

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

    def _build_prompt(self, question: str, contexts: list[dict]) -> str:
        context_block = "\n\n".join(
            f"[{index + 1}] {context['text']}" for index, context in enumerate(contexts)
        )
        return (
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say you don't know. "
            "Answer in the same language as the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _generate_with_ollama(
        self, prompt: str, json_output: bool = False, max_new_tokens: int | None = None
    ) -> str:
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "think": self.config.llm_think,
            # Room for long input texts (question generation); Ollama's default is smaller.
            "options": {"num_ctx": 8192},
        }
        if json_output:
            payload["format"] = "json"
        if max_new_tokens:
            payload["options"]["num_predict"] = max_new_tokens
        parsed = self._post_json(
            f"{self.config.ollama_url.rstrip('/')}/api/generate",
            payload,
            None,
            self.config.llm_timeout_seconds,
            "Ollama",
        )
        return parsed.get("response") or ""

    def _generate_with_api(self, prompt: str, max_new_tokens: int | None = None) -> str:
        url = self.config.llm_api_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        parsed = self._post_json(
            url,
            {
                "model": self.config.llm_api_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens or self.config.llm_max_new_tokens,
                "stream": False,
            },
            self.config.llm_api_key,
            self.config.llm_timeout_seconds,
            "LLM API",
        )
        try:
            return parsed["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"LLM API returned an unexpected response: {parsed}") from error

    def _get_local_llm(self):
        if self._local_llm is None:
            with self._local_llm_lock:
                if self._local_llm is None:
                    import torch
                    from transformers import pipeline as hf_pipeline

                    if torch.cuda.is_available():
                        device = "cuda"
                    elif torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                    print(f"[llm] loading local model {self.config.llm_local_model} on {device}...")
                    self._local_llm = hf_pipeline(
                        "text-generation",
                        model=self.config.llm_local_model,
                        torch_dtype="auto",
                        device=device,
                    )
                    print("[llm] local model ready")
        return self._local_llm

    def _generate_with_local_model(self, prompt: str, max_new_tokens: int | None = None) -> str:
        try:
            generator = self._get_local_llm()
            output = generator(
                [{"role": "user", "content": prompt}],
                max_new_tokens=max_new_tokens or self.config.llm_max_new_tokens,
                do_sample=False,
            )
            return output[0]["generated_text"][-1]["content"]
        except Exception as error:
            raise RuntimeError(f"Local LLM failed: {error}") from error

    def warm_up_llm(self) -> None:
        """Loads the local LLM up front so the first question isn't slow."""
        if self.config.llm_mode == "local":
            self._get_local_llm()

    def describe_llm(self) -> str:
        if self.config.llm_mode == "ollama":
            return f"ollama:{self.config.ollama_model} @ {self.config.ollama_url}"
        if self.config.llm_mode == "api":
            return f"api:{self.config.llm_api_model} @ {self.config.llm_api_url}"
        return f"local:{self.config.llm_local_model}"

    def describe_retrieval(self) -> str:
        description = self.config.retrieval_mode
        if self.config.rerank_mode == "local":
            description += f" + rerank local:{self.config.rerank_model}"
        elif self.config.rerank_mode == "api":
            description += f" + rerank api:{self.config.rerank_model} @ {self.config.rerank_api_url}"
        chunk = self.splitter.config
        return f"{description} (candidates={self.config.rerank_candidates}, chunk={chunk.max_chars} chars)"

    def describe_embedder(self) -> str:
        if self.config.embedding_mode == "api":
            return f"api:{self.config.embedding_api_model} @ {self.config.embedding_api_url}"
        return f"local:{self.config.embedding_model}"

    def generate(
        self, prompt: str, json_output: bool = False, max_new_tokens: int | None = None
    ) -> str:
        """Runs the configured LLM. `json_output` asks Ollama for strict JSON; other
        backends rely on the prompt asking for JSON."""
        if self.config.llm_mode == "api":
            answer = self._generate_with_api(prompt, max_new_tokens)
        elif self.config.llm_mode == "local":
            answer = self._generate_with_local_model(prompt, max_new_tokens)
        else:
            answer = self._generate_with_ollama(prompt, json_output, max_new_tokens)
        if not answer or not answer.strip():
            raise RuntimeError("LLM returned an empty response.")
        return answer.strip()

    def answer_question(
        self, question: str, top_k: int | None = None, use_llm: bool = True
    ) -> dict:
        contexts = self.retrieve(question=question, top_k=top_k)

        if use_llm and contexts:
            try:
                answer = self.generate(self._build_prompt(question=question, contexts=contexts))
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

    # Long texts are cut to keep the prompt inside the LLM's context window.
    QUESTION_TEXT_MAX_CHARS = 6000

    def _build_questions_prompt(self, text: str, num_questions: int, contexts: list[dict]) -> str:
        prompt = (
            f"Write {num_questions} open-ended questions that test understanding of the TEXT below, "
            "each with a short, correct answer. Questions must be about the TEXT. Answers must be "
            "based on the TEXT; the additional context may only be used to make answers more "
            "complete. Write the questions and answers in the same language as the TEXT.\n\n"
            f"TEXT:\n{text}\n\n"
        )
        if contexts:
            context_block = "\n\n".join(
                f"[{index + 1}] {context['text']}" for index, context in enumerate(contexts)
            )
            prompt += f"Additional context from the knowledge base:\n{context_block}\n\n"
        prompt += (
            'Respond with JSON only, in this form: '
            '{"questions": [{"question": "...", "answer": "..."}]}'
        )
        return prompt

    @staticmethod
    def _parse_questions(raw: str) -> list[dict]:
        """Extracts [{question, answer}] from the LLM output (JSON, or "Q:/A:" lines)."""
        match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    data = next((v for v in data.values() if isinstance(v, list)), [])
                questions = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    lowered = {str(k).lower(): v for k, v in item.items()}
                    question = lowered.get("question") or lowered.get("vprašanje") or lowered.get("q")
                    answer = lowered.get("answer") or lowered.get("odgovor") or lowered.get("a")
                    if question:
                        questions.append({"question": str(question).strip(), "answer": str(answer or "").strip()})
                if questions:
                    return questions
            except json.JSONDecodeError:
                pass

        questions = []
        for line in raw.splitlines():
            line = line.strip().lstrip("-*0123456789. )")
            q = re.match(r"^(?:Q|V|Question|Vprašanje)\s*[:.]\s*(.+)", line, re.IGNORECASE)
            a = re.match(r"^(?:A|O|Answer|Odgovor)\s*[:.]\s*(.+)", line, re.IGNORECASE)
            if q:
                questions.append({"question": q.group(1).strip(), "answer": ""})
            elif a and questions and not questions[-1]["answer"]:
                questions[-1]["answer"] = a.group(1).strip()
        return questions

    def generate_questions(
        self, text: str, num_questions: int = 5, top_k: int | None = None, use_context: bool = True
    ) -> dict:
        """Text -> open questions with answers, grounded in the text and the knowledge base."""
        text = text.strip()
        truncated = len(text) > self.QUESTION_TEXT_MAX_CHARS
        if truncated:
            text = text[: self.QUESTION_TEXT_MAX_CHARS]

        contexts = self.retrieve(text[:2000], top_k=top_k) if use_context else []
        raw = self.generate(
            self._build_questions_prompt(text, num_questions, contexts),
            json_output=True,
            max_new_tokens=max(512, 160 * num_questions),
        )
        questions = self._parse_questions(raw)
        if not questions:
            raise RuntimeError(f"Could not read questions from the LLM output: {raw[:300]}")
        return {
            "questions": questions[:num_questions],
            "contexts": contexts,
            "text_truncated": truncated,
        }
