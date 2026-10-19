#!/usr/bin/env bash
# Run the RAG server directly on this machine (no Docker for the app itself).
# Creates .venv, installs requirements, starts Qdrant in Docker if it isn't
# already reachable, then runs uvicorn. Local embedder / LLM_MODE=local models
# are loaded into this process, using CUDA / Apple MPS / CPU.
#
#   scripts/start-local.sh               # serve on 0.0.0.0:8000 in the foreground
#   scripts/start-local.sh --background  # keep running after logout (logs/rag-api.log)
#   PYTHON=python3.12 scripts/start-local.sh
#
# Without Docker, Qdrant is downloaded as a standalone binary into .qdrant/ and
# run from there (no root needed).
source "$(dirname "$0")/_common.sh"

BACKGROUND=false
[ "${1:-}" = "--background" ] && BACKGROUND=true

# --- Python / venv ---
if [ -z "${PYTHON:-}" ]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [ ! -d .venv ]; then
  echo "Creating .venv with $PYTHON ($("$PYTHON" --version))..."
  if ! "$PYTHON" -c "import ensurepip" 2>/dev/null; then
    # Debian/Ubuntu without the python3-venv package: create the venv without pip
    # and bootstrap pip separately (no root needed).
    "$PYTHON" -m venv --without-pip .venv
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python - --quiet
  else
    "$PYTHON" -m venv .venv
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Reinstall only when the requirements (or the need for local models) changed.
NEED_LOCAL_MODELS=false
if [ "${EMBEDDING_MODE:-api}" = "local" ] || [ "$LLM_MODE" = "local" ] || [ "${RERANK_MODE:-off}" = "local" ]; then
  NEED_LOCAL_MODELS=true
fi
STAMP=.venv/.requirements.sha
CURRENT_SHA="$(cat requirements.txt requirements-local.txt | shasum | cut -d' ' -f1)-$NEED_LOCAL_MODELS"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$CURRENT_SHA" ]; then
  echo "Installing requirements..."
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
  if [ "$NEED_LOCAL_MODELS" = true ]; then
    # On Linux without an NVIDIA GPU, the CPU-only PyTorch build is ~10x smaller.
    if [ "$(uname -s)" = "Linux" ] && ! nvidia-smi >/dev/null 2>&1; then
      pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
    pip install -r requirements-local.txt
  fi
  echo "$CURRENT_SHA" > "$STAMP"
fi

# --- Qdrant ---
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:${QDRANT_HOST_PORT:-6333}}"
if ! is_reachable "$QDRANT_URL/readyz"; then
  case "$QDRANT_URL" in
    *127.0.0.1*|*localhost*)
      QDRANT_PORT="${QDRANT_URL##*:}"
      QDRANT_PORT="${QDRANT_PORT%%/*}"
      if docker_available; then
        echo "Qdrant not reachable at $QDRANT_URL, starting it in Docker (container: rag-qdrant)..."
        if [ -n "$(docker_cmd ps -aq -f name=^rag-qdrant$)" ]; then
          docker_cmd start rag-qdrant >/dev/null
        else
          docker_cmd run -d --name rag-qdrant \
            -p "127.0.0.1:$QDRANT_PORT:6333" \
            -v rag_qdrant_local:/qdrant/storage \
            qdrant/qdrant:latest >/dev/null
        fi
      else
        echo "Qdrant not reachable at $QDRANT_URL and Docker isn't usable - running the Qdrant binary from .qdrant/"
        mkdir -p "$QDRANT_LOCAL_DIR/storage"
        if [ ! -x "$QDRANT_LOCAL_DIR/qdrant" ]; then
          case "$(uname -s)-$(uname -m)" in
            Linux-x86_64) ASSET=qdrant-x86_64-unknown-linux-gnu.tar.gz ;;
            Linux-aarch64) ASSET=qdrant-aarch64-unknown-linux-musl.tar.gz ;;
            Darwin-arm64) ASSET=qdrant-aarch64-apple-darwin.tar.gz ;;
            Darwin-x86_64) ASSET=qdrant-x86_64-apple-darwin.tar.gz ;;
            *) echo "No Qdrant binary for $(uname -s)-$(uname -m)." >&2; exit 1 ;;
          esac
          QDRANT_VERSION="${QDRANT_VERSION:-v1.17.1}"
          echo "Downloading Qdrant $QDRANT_VERSION ($ASSET)..."
          curl -fsSL "https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/$ASSET" \
            | tar -xz -C "$QDRANT_LOCAL_DIR"
        fi
        # Bound to localhost only; gRPC on the next port.
        QDRANT__SERVICE__HOST=127.0.0.1 \
        QDRANT__SERVICE__HTTP_PORT="$QDRANT_PORT" \
        QDRANT__SERVICE__GRPC_PORT="$((QDRANT_PORT + 1))" \
        QDRANT__STORAGE__STORAGE_PATH="$QDRANT_LOCAL_DIR/storage" \
        QDRANT__STORAGE__SNAPSHOTS_PATH="$QDRANT_LOCAL_DIR/snapshots" \
        QDRANT__TELEMETRY_DISABLED=true \
          nohup "$QDRANT_LOCAL_DIR/qdrant" > "$QDRANT_LOCAL_DIR/qdrant.log" 2>&1 &
        echo $! > "$QDRANT_LOCAL_DIR/qdrant.pid"
      fi
      for _ in $(seq 1 30); do
        is_reachable "$QDRANT_URL/readyz" && break
        sleep 1
      done
      is_reachable "$QDRANT_URL/readyz" || { echo "Qdrant did not start (see .qdrant/qdrant.log)." >&2; exit 1; }
      ;;
    *)
      echo "WARNING: external Qdrant at $QDRANT_URL is not reachable."
      ;;
  esac
fi

# --- Ollama models (LLM and/or embedder) ---
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:${OLLAMA_HOST_PORT:-11434}}"
OLLAMA_NEEDED=()
[ "$LLM_MODE" = "ollama" ] && OLLAMA_NEEDED+=("${OLLAMA_MODEL:-qwen3.5:4b}")
[ "${EMBEDDING_MODE:-api}" = "api" ] && [ -z "${EMBEDDING_API_URL:-}" ] && OLLAMA_NEEDED+=("${EMBEDDING_API_MODEL:-bge-m3}")

if [ ${#OLLAMA_NEEDED[@]} -gt 0 ]; then
  if is_reachable "$OLLAMA_URL/api/tags"; then
    for MODEL in "${OLLAMA_NEEDED[@]}"; do
      case "$MODEL" in *:*) ;; *) MODEL="$MODEL:latest" ;; esac
      if ! curl -fsS "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"$MODEL\""; then
        echo "Pulling Ollama model $MODEL into $OLLAMA_URL..."
        curl -fsS "$OLLAMA_URL/api/pull" -d "{\"model\":\"$MODEL\",\"stream\":false}" \
          || echo "WARNING: pull failed - pull it manually on that Ollama server."
        echo
      fi
    done
  else
    echo "WARNING: Ollama not reachable at $OLLAMA_URL (needed for: ${OLLAMA_NEEDED[*]})."
    echo "         Install/start Ollama (https://ollama.com, then 'ollama serve'), or set"
    echo "         OLLAMA_URL / EMBEDDING_API_URL / LLM_MODE in .env."
  fi
fi
case "$LLM_MODE" in
  local) echo "LLM_MODE=local: ${LLM_LOCAL_MODEL:-Qwen/Qwen2.5-1.5B-Instruct} is loaded on the first question." ;;
  api) echo "LLM_MODE=api: ${LLM_API_MODEL:-?} @ ${LLM_API_URL:-?}" ;;
esac

PORT="${RAG_API_HOST_PORT:-8000}"
echo
echo "Chat UI: http://localhost:$PORT"
if [ "$BACKGROUND" = true ]; then
  mkdir -p logs
  stop_pid_file "$RAG_PID_FILE" "previous rag-api"
  PYTHONUNBUFFERED=1 nohup uvicorn app:app --host "${HOST:-0.0.0.0}" --port "$PORT" > logs/rag-api.log 2>&1 &
  echo $! > "$RAG_PID_FILE"
  echo "Running in the background (pid $!). Logs: tail -f logs/rag-api.log   Stop: scripts/stop.sh"
else
  exec uvicorn app:app --host "${HOST:-0.0.0.0}" --port "$PORT"
fi
