"""Cross-encoder reranker served next to Ollama in the models container (GPU if present).

    POST /rerank  {"query": "...", "documents": ["...", ...]}   ("texts" works too)
      -> {"results": [{"index": 0, "relevance_score": 0.98}, ...]}   (best first)
    GET  /health
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from sentence_transformers import CrossEncoder

MODEL = os.getenv("RERANK_MODEL") or "BAAI/bge-reranker-v2-m3"
PORT = int(os.getenv("RERANKER_PORT") or 8001)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[reranker] loading {MODEL} on {device}...", flush=True)
model = CrossEncoder(MODEL, max_length=512, device=device)
# One GPU, one model: run requests one after another.
lock = threading.Lock()
print(f"[reranker] ready on :{PORT} ({device})", flush=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": MODEL, "device": device})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") not in {"/rerank", "/v1/rerank"}:
            self._send(404, {"error": "not found"})
            return
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            query = request["query"]
            documents = request.get("documents") or request.get("texts") or []
        except (ValueError, KeyError) as error:
            self._send(400, {"error": f"expected {{query, documents}}: {error}"})
            return
        if not documents:
            self._send(200, {"model": MODEL, "results": []})
            return
        with lock:
            scores = model.predict([(query, str(document)) for document in documents])
        results = sorted(
            ({"index": index, "relevance_score": float(score)} for index, score in enumerate(scores)),
            key=lambda item: -item["relevance_score"],
        )
        self._send(200, {"model": MODEL, "results": results})

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
