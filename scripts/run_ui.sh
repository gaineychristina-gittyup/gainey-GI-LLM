#!/usr/bin/env bash
# Launch the GI Guidelines Assistant Streamlit UI.
#
# Prerequisites:
#   - Postgres + pgvector running (`docker compose up -d postgres`)
#   - FastAPI backend running (`uvicorn src.api.app:app --port 8000 --reload`)
#   - .env populated with VOYAGE_API_KEY / COHERE_API_KEY / ANTHROPIC_API_KEY
#
# Usage:
#   bash scripts/run_ui.sh           # default port 8501
#   PORT=8888 bash scripts/run_ui.sh # custom port

set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8501}"

# Pick up the venv's streamlit if present, otherwise fall back to system PATH.
if [ -x .venv/bin/streamlit ]; then
    STREAMLIT=".venv/bin/streamlit"
else
    STREAMLIT="streamlit"
fi

echo "Launching Streamlit UI on http://localhost:${PORT}"
echo "(Backend is expected at \$GI_API_BASE = ${GI_API_BASE:-http://localhost:8000})"

exec "$STREAMLIT" run src/ui/streamlit_app.py \
    --server.port "$PORT" \
    --server.headless true
