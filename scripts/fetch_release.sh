#!/usr/bin/env bash
# Download the processed data and the trained model from the GitHub Release (a few MB),
# check the SHA-256 and unpack into data/ and checkpoints/. No GitHub account needed.
#
# Usage:  scripts/fetch_release.sh [tag]          (default: latest release)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="madmax0978/scentmatch"
TAG="${1:-latest}"
if [ "$TAG" = latest ]; then BASE="https://github.com/$REPO/releases/latest/download"
else BASE="https://github.com/$REPO/releases/download/$TAG"; fi
mkdir -p dist
curl -fL --retry 3 -o dist/scentmatch-data.tar.gz "$BASE/scentmatch-data.tar.gz"
curl -fL --retry 3 -o dist/SHA256SUMS "$BASE/SHA256SUMS"
(cd dist && shasum -a 256 -c SHA256SUMS)
tar -xzf dist/scentmatch-data.tar.gz
echo "data/processed, data/scentmatch.db and checkpoints/ are ready. Next: scripts/run_demo.sh"
