#!/usr/bin/env bash
# Shared helpers for the scripts in this folder. Sourced, not run directly.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example - edit it to pick your embedder / LLM."
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

LLM_MODE="${LLM_MODE:-ollama}"

# Docker is checked lazily: start-local.sh / init.sh / chat.sh also work without it.
# Uses sudo only if the daemon runs but the current user may not use it.
DOCKER_SUDO=""
DOCKER_STATE=""   # "", "ok" or "missing"

docker_available() {
  if [ -z "$DOCKER_STATE" ]; then
    local err
    if ! command -v docker >/dev/null 2>&1; then
      DOCKER_STATE="missing"
      DOCKER_ERR="docker is not installed"
    elif err="$(docker info 2>&1 >/dev/null)"; then
      DOCKER_STATE="ok"
    elif echo "$err" | grep -qi "permission denied" && sudo -n true 2>/dev/null; then
      DOCKER_SUDO="sudo"
      DOCKER_STATE="ok"
    else
      DOCKER_STATE="missing"
      DOCKER_ERR="$(echo "$err" | grep -v '^$' | tail -1)"
    fi
  fi
  [ "$DOCKER_STATE" = "ok" ]
}

require_docker() {
  if ! docker_available; then
    echo "Docker is not usable: $DOCKER_ERR" >&2
    echo "Start Docker (Docker Desktop, 'colima start', 'sudo systemctl start docker'), or add" >&2
    echo "yourself to the docker group ('sudo usermod -aG docker \$USER', then log in again)." >&2
    echo "Without Docker use scripts/start-local.sh." >&2
    exit 1
  fi
}

docker_cmd() {
  require_docker
  $DOCKER_SUDO docker "$@"
}

EMBEDDING_MODE="${EMBEDDING_MODE:-api}"

# The `models` container (Ollama) serves the LLM and/or the embedder when they
# don't come from an external URL. An empty EMBEDDING_API_URL means "the embedding
# model on the same Ollama as the LLM" (OLLAMA_URL).
llm_uses_models_container() {
  [ "$LLM_MODE" = "ollama" ] && [ -z "${OLLAMA_URL:-}" ]
}
embedder_uses_models_container() {
  [ "$EMBEDDING_MODE" = "api" ] && [ -z "${EMBEDDING_API_URL:-}" ] && [ -z "${OLLAMA_URL:-}" ]
}
# RERANK_MODE=api with an empty RERANK_API_URL = the reranker in the models container.
reranker_uses_models_container() {
  [ "${RERANK_MODE:-off}" = "api" ] && [ -z "${RERANK_API_URL:-}" ] && [ -z "${OLLAMA_URL:-}" ]
}
uses_models_container() {
  llm_uses_models_container || embedder_uses_models_container || reranker_uses_models_container
}

# Exactly the models the `models` container has to pull.
OLLAMA_PULL=""
llm_uses_models_container && OLLAMA_PULL="${OLLAMA_MODEL:-qwen3.5:4b}"
embedder_uses_models_container && OLLAMA_PULL="$OLLAMA_PULL ${EMBEDDING_API_MODEL:-bge-m3}"
export OLLAMA_PULL
MODELS_RERANKER=false
reranker_uses_models_container && MODELS_RERANKER=true
export MODELS_RERANKER

# Prefer the `docker compose` plugin, fall back to standalone `docker-compose`.
compose_cmd() {
  require_docker
  if docker compose version >/dev/null 2>&1; then
    docker_cmd compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    $DOCKER_SUDO docker-compose "$@"
  else
    echo "Neither 'docker compose' nor 'docker-compose' is installed." >&2
    exit 1
  fi
}

# GPU for the models container: MODELS_GPU=auto (default) uses it when Docker has
# the nvidia runtime, true forces it, false disables it.
models_use_gpu() {
  case "${MODELS_GPU:-auto}" in
    true) return 0 ;;
    false) return 1 ;;
    *) docker_available && $DOCKER_SUDO docker info 2>/dev/null | grep -qi "runtimes:.*nvidia" ;;
  esac
}

compose() {
  local args=(-f docker-compose.yml)
  if uses_models_container; then
    args+=(--profile models)
    models_use_gpu && args+=(-f docker-compose.gpu.yml)
  fi
  compose_cmd "${args[@]}" "$@"
}

web_running() {
  docker_available && [ -n "$(compose ps --status running -q web 2>/dev/null)" ]
}

# Qdrant run as a plain binary (no Docker), used by start-local.sh / stop.sh.
QDRANT_LOCAL_DIR="$ROOT_DIR/.qdrant"
RAG_PID_FILE="$ROOT_DIR/logs/rag-api.pid"

stop_pid_file() {
  local pid_file="$1" name="$2"
  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    kill "$(cat "$pid_file")"
    echo "Stopped $name (pid $(cat "$pid_file"))."
  fi
  rm -f "$pid_file"
}

is_reachable() {
  curl -fs -o /dev/null --max-time 3 "$1"
}
