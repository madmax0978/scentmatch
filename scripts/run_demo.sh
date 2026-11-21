#!/usr/bin/env bash
# Start the API (trained model) and the Streamlit front end in one command.
# Works with the repo's .venv or with any already-activated Python environment.
# PYTHONPATH=src makes the package importable even if the editable install's .pth file
# is ignored (macOS can flag it "hidden" when the folder is synced by iCloud).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY="python3"; [ -x .venv/bin/python ] && PY=".venv/bin/python"
CKPT="${SCENTMATCH_CHECKPOINT:-$($PY -c "import yaml;print(yaml.safe_load(open('configs/default.yaml'))['serving']['checkpoint'])")}"
MODE="${SCENTMATCH_MODE:-model}"
if [ "$MODE" = model ] && [ ! -f "checkpoints/$CKPT" ]; then
  echo "no checkpoints/$CKPT: run scripts/fetch_release.sh (or train), or use SCENTMATCH_MODE=mock $0"; exit 1
fi
SCENTMATCH_MODE="$MODE" SCENTMATCH_CHECKPOINT="$CKPT" $PY -m uvicorn scentmatch.api.main:app --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
for _ in $(seq 1 60); do curl -s -o /dev/null http://localhost:8000/health && break; sleep 1; done
echo "API ready on http://localhost:8000 (mode: $MODE, checkpoint: $CKPT)"
$PY -m streamlit run app/streamlit_app.py --server.port 8501
