#!/usr/bin/env bash
# Packages this project into a clean, deployable folder (no venv/caches/unrelated
# coursework) so it can be copied straight to a server and run with docker-compose.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${1:-$SOURCE_DIR/../RAG_Server_deploy}"

EXCLUDES=(
  ".venv"
  "__pycache__"
  "*.pyc"
  ".DS_Store"
  ".git"
  ".env"
  "dataset_text"
  "Jezikovne tehnologije"
  "package.sh"
  ".qdrant"
  "logs"
)

RSYNC_EXCLUDES=()
for pattern in "${EXCLUDES[@]}"; do
  RSYNC_EXCLUDES+=(--exclude "$pattern")
done

echo "Packaging: $SOURCE_DIR"
echo "       -> : $DEST_DIR"

mkdir -p "$DEST_DIR"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$SOURCE_DIR"/ "$DEST_DIR"/

echo
echo "Done. Deployable folder ready at:"
echo "  $DEST_DIR"
echo
echo "To copy it to a server:"
echo "  scp -r \"$DEST_DIR\" user@server:/path/to/deploy/"
echo "  # or: rsync -avz \"$DEST_DIR\"/ user@server:/path/to/deploy/RAG_Server/"
echo
echo "On the server, then run:"
echo "  cd /path/to/deploy/RAG_Server && cp .env.example .env && docker compose up --build -d"
