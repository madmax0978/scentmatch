#!/usr/bin/env bash
# Start the API (trained model) and the Streamlit front end in one command.
# PYTHONPATH=src makes the package importable even if the editable install's .pth file
# is ignored (macOS can flag it "hidden" when the folder is synced by iCloud).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
MODE="${SCENTMATCH_MODE:-model}"
[ "$MODE" = model ] && [ ! -f checkpoints/nlgcl.pt ] && { echo "no checkpoints/nlgcl.pt: train first or run SCENTMATCH_MODE=mock $0"; exit 1; }
SCENTMATCH_MODE="$MODE" .venv/bin/uvicorn scentmatch.api.main:app --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
for _ in $(seq 1 30); do curl -s -o /dev/null http://localhost:8000/health && break; sleep 1; done
echo "API ready on http://localhost:8000 (mode: $MODE)"
.venv/bin/streamlit run app/streamlit_app.py --server.port 8501
