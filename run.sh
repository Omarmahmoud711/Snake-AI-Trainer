#!/usr/bin/env bash
# Convenience runner — assumes the conda env "snake-rl" exists.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snake-rl

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

exec uvicorn backend.server:app --host "$HOST" --port "$PORT" "$@"
