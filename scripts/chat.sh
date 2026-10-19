#!/usr/bin/env bash
# Terminal chat against the pipeline (inside the container if the stack is up,
# otherwise locally). Shows full tracebacks instead of a generic HTTP 500.
source "$(dirname "$0")/_common.sh"

if web_running; then
  compose exec web python chat_cli.py
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:${QDRANT_HOST_PORT:-6333}}"
  export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:${OLLAMA_HOST_PORT:-11434}}"
  python chat_cli.py
fi
