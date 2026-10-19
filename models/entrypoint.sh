#!/bin/sh
# Ollama serving both the embedding model and the LLM. Pulls the models listed in
# OLLAMA_PULL (space separated) on startup; already-present models are skipped quickly.
# With RERANKER_ENABLED=true (image built with RERANKER=true) also starts the
# reranker on port RERANKER_PORT (8001).
set -e

ollama serve &
SERVE_PID=$!

if [ "${RERANKER_ENABLED:-false}" = "true" ]; then
  if [ -x /opt/reranker/bin/python ]; then
    /opt/reranker/bin/python /reranker_server.py &
  else
    echo "WARNING: RERANKER_ENABLED=true, but the image was built without RERANKER=true"
  fi
fi

echo "Waiting for Ollama server to become ready..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

for model in $OLLAMA_PULL; do
  echo "Pulling model: $model"
  ollama pull "$model" || echo "WARNING: could not pull $model"
done
echo "Models ready: $(ollama list | tail -n +2 | awk '{print $1}' | tr '\n' ' ')"

wait "$SERVE_PID"
