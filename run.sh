#!/usr/bin/env bash
set -euo pipefail

# ── EgoCut Pipeline ──────────────────────────────────────────────────
# Usage:
#   ./run.sh                          # full pipeline
#   ./run.sh label                    # labeling only
#   ./run.sh train                    # clips + extract + train only
#   OVERLAP=0.5 ./run.sh train        # training with 50% clip overlap

OVERLAP=${OVERLAP:-0.0}
STAGE=${1:-all}

# ── helpers ──────────────────────────────────────────────────────────
log() { echo "[$(date +%H:%M:%S)] $*"; }
need() { command -v "$1" &>/dev/null || { echo "ERROR: $1 not found"; exit 1; }; }

need python
need ffmpeg
need ffprobe

# ── 0. environment check ─────────────────────────────────────────────
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "ERROR: GEMINI_API_KEY not set (set in .env or environment)"
    exit 1
fi

# ── 1. label ─────────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "label" ]]; then
    if [[ ! -f cache_name.txt ]]; then
        log "Building Gemini context cache..."
        python create_cache.py
    else
        log "Cache found: $(cat cache_name.txt)"
    fi

    log "Batch labeling unlabeled videos..."
    python label_batch.py
fi

# ── 2. make clips ─────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    log "Building clip manifests (overlap=$OVERLAP)..."
    python make_clips.py --overlap "$OVERLAP"
fi

# ── 3. extract features ───────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    log "Extracting VJEPA2 features..."
    python extract.py
fi

# ── 4. train probe ────────────────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    log "Training..."
    python train.py
fi

log "Done."
