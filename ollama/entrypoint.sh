#!/bin/sh
set -e

ollama serve &
SERVE_PID=$!

echo "Waiting for Ollama server to become ready..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

if [ -n "$OLLAMA_MODEL" ]; then
  echo "Pulling model: $OLLAMA_MODEL"
  ollama pull "$OLLAMA_MODEL"
else
  echo "OLLAMA_MODEL not set, skipping model pull."
fi

wait "$SERVE_PID"
