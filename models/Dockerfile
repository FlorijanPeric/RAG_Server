# Pinned: a cached "latest" can be months old and too old for newer models (qwen3.5).
ARG OLLAMA_VERSION=0.40.4
FROM ollama/ollama:${OLLAMA_VERSION}

# RERANKER=true also installs a cross-encoder reranker (PyTorch + sentence-transformers,
# several GB) that runs next to Ollama, on the GPU when there is one. scripts/start.sh
# sets it when RERANK_MODE=api uses the models container.
ARG RERANKER=false
# cu126 wheels still include older GPUs (Pascal, e.g. GTX 1080 Ti); CUDA 12.8+ ones don't.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN if [ "$RERANKER" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
      && rm -rf /var/lib/apt/lists/* \
      && python3 -m venv /opt/reranker \
      && /opt/reranker/bin/pip install --no-cache-dir torch --index-url "$TORCH_INDEX_URL" \
      && /opt/reranker/bin/pip install --no-cache-dir sentence-transformers; \
    fi

COPY reranker_server.py /reranker_server.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
