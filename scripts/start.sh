#!/usr/bin/env bash
# Start the stack in Docker: `web` (UI + API) + `qdrant`, and `models` (Ollama with
# the embedder and the LLM) when they don't come from external URLs. Extra
# arguments are passed to `docker compose up`.
#
#   scripts/start.sh            # build + start in the background
#   scripts/start.sh --logs     # ...and follow the web logs afterwards
source "$(dirname "$0")/_common.sh"

FOLLOW_LOGS=false
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --logs) FOLLOW_LOGS=true ;;
    *) ARGS+=("$arg") ;;
  esac
done

if llm_uses_models_container; then
  echo "LLM:      ${OLLAMA_MODEL:-qwen3.5:4b} in the models container"
elif [ "$LLM_MODE" = "ollama" ]; then
  echo "LLM:      ${OLLAMA_MODEL:-qwen3.5:4b} @ $OLLAMA_URL"
else
  echo "LLM:      LLM_MODE=$LLM_MODE"
fi
if embedder_uses_models_container; then
  echo "Embedder: ${EMBEDDING_API_MODEL:-bge-m3} in the models container"
elif [ "$EMBEDDING_MODE" = "api" ]; then
  echo "Embedder: ${EMBEDDING_API_MODEL:-bge-m3} @ ${EMBEDDING_API_URL:-$OLLAMA_URL/api/embed}"
else
  echo "Embedder: EMBEDDING_MODE=$EMBEDDING_MODE (${EMBEDDING_MODEL:-})"
fi
if reranker_uses_models_container; then
  echo "Search:   ${RETRIEVAL_MODE:-hybrid}, reranker: ${RERANK_MODEL:-BAAI/bge-reranker-v2-m3} in the models container"
else
  echo "Search:   ${RETRIEVAL_MODE:-hybrid}, reranker: ${RERANK_MODE:-off}"
fi
if { [ "$EMBEDDING_MODE" = "local" ] || [ "$LLM_MODE" = "local" ] || [ "${RERANK_MODE:-off}" = "local" ]; } \
  && [ "${LOCAL_MODELS:-false}" != "true" ]; then
  echo "WARNING: EMBEDDING_MODE=local / LLM_MODE=local / RERANK_MODE=local need LOCAL_MODELS=true in .env" >&2
  echo "         (installs PyTorch into the web image)." >&2
fi
if uses_models_container; then
  models_use_gpu && echo "GPU:      NVIDIA GPU for the models container" || echo "GPU:      none (models run on CPU)"
else
  echo "models container not needed - not started."
fi

compose up --build -d --remove-orphans ${ARGS[@]+"${ARGS[@]}"}
compose ps

PORT="${RAG_API_HOST_PORT:-8000}"
echo
echo "Web UI:    http://localhost:$PORT"
echo "Logs:      docker compose logs -f web"
echo "Re-init:   scripts/init.sh --reset"
echo "Stop:      scripts/stop.sh"
if uses_models_container; then
  echo "First start downloads the models ($OLLAMA_PULL) - see: docker compose logs -f models"
fi

if [ "$FOLLOW_LOGS" = true ]; then
  compose logs -f web
fi
