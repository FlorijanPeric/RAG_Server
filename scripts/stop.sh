#!/usr/bin/env bash
# Stop everything: the Docker stack, and what start-local.sh started (background
# web app, Qdrant container or binary). Data is kept. With Docker, pass --volumes
# to also wipe the Docker volumes (Qdrant data, models).
source "$(dirname "$0")/_common.sh"

if docker_available; then
  compose_cmd --profile models down "$@"
  if [ -n "$(docker_cmd ps -q -f name=^rag-qdrant$)" ]; then
    docker_cmd stop rag-qdrant >/dev/null
    echo "Stopped rag-qdrant."
  fi
fi
stop_pid_file "$RAG_PID_FILE" "rag-api"
stop_pid_file "$QDRANT_LOCAL_DIR/qdrant.pid" "Qdrant"
