#!/usr/bin/env bash
# Build the GitHub Release assets so that anyone can run ScentMatch in minutes, without the
# 4.2 GB raw download. Produces dist/scentmatch-data.tar.gz (processed data, empty SQLite
# catalogue DB, the served checkpoint) and dist/SHA256SUMS.
#
# Usage:  scripts/make_release.sh [checkpoint]        (default: the one in configs/default.yaml)
# Publish (done by a team member, needs the GitHub CLI):
#   gh release create v0.3-week3 dist/scentmatch-data.tar.gz dist/SHA256SUMS \
#       --title "Week 3: data + trained model" --notes "Assets for scripts/fetch_release.sh"
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-$(python3 -c "import yaml;print(yaml.safe_load(open('configs/default.yaml'))['serving']['checkpoint'])")}"
[ -f "checkpoints/$CKPT" ] || { echo "missing checkpoints/$CKPT"; exit 1; }
STAGE="$(mktemp -d)"                       # outside the repo (iCloud can re-create deleted folders)
trap 'rm -rf "$STAGE"' EXIT
rm -rf dist && mkdir -p dist "$STAGE/data" "$STAGE/checkpoints"
cp -R data/processed "$STAGE/data/processed"
# catalogue DB with an empty interactions table (no demo clicks in the release)
cp data/scentmatch.db "$STAGE/data/scentmatch.db"
sqlite3 "$STAGE/data/scentmatch.db" "DELETE FROM interactions; VACUUM;"
cp "checkpoints/$CKPT" "$STAGE/checkpoints/"
for f in checkpoints/cold_nlgcl.pt; do [ -f "$f" ] && cp "$f" "$STAGE/checkpoints/"; done
tar -czf dist/scentmatch-data.tar.gz -C "$STAGE" data checkpoints
(cd dist && shasum -a 256 scentmatch-data.tar.gz > SHA256SUMS)
ls -lh dist
