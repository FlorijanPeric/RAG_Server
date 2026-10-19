#!/usr/bin/env bash
# (Re)initialize the RAG system - checks embedder/Qdrant/LLM, then ingests dataset/.
# Runs inside the web container if the Docker stack is up, otherwise locally
# (.venv, as set up by start-local.sh). All arguments go to init_rag.py:
#
#   scripts/init.sh                 # ingest if the collection is empty
#   scripts/init.sh --reset         # drop the collection and re-ingest (after changing embedder!)
#   scripts/init.sh --check         # only test embedder, Qdrant and LLM
#   scripts/init.sh --dir my_docs   # ingest another directory
source "$(dirname "$0")/_common.sh"

if web_running; then
  compose exec web python init_rag.py "$@"
else
  if [ ! -d .venv ]; then
    echo "Docker stack isn't running and there's no .venv."
    echo "Start it first: scripts/start.sh (Docker) or scripts/start-local.sh (local)."
    exit 1
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:${QDRANT_HOST_PORT:-6333}}"
  export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:${OLLAMA_HOST_PORT:-11434}}"
  python init_rag.py "$@"
fi
