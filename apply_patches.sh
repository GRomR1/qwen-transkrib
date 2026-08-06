#!/bin/bash
# apply_patches.sh — Apply MetaX-specific patches after uv sync
# Run after: uv sync
# These patches are needed because:
# 1. transformers cache_utils.py has a bug with Qwen3-ASR's 4D key/value tensors
# 2. pyannote wespeaker uses torch.vmap which creates tensors without storage on MetaX

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: .venv not found. Run 'uv sync' first."
    exit 1
fi

echo "Applying patches to $VENV_DIR ..."

# Patch 1: transformers cache_utils.py (DynamicLayer.lazy_initialization)
TRANSFORMERS_FILE="$VENV_DIR/lib/python3.12/site-packages/transformers/cache_utils.py"
if [ -f "$TRANSFORMERS_FILE" ]; then
    echo "  Patching transformers cache_utils.py ..."
    python3 "$SCRIPT_DIR/patches/apply_transformers_patch.py" "$TRANSFORMERS_FILE"
    echo "  Done."
else
    echo "  Warning: $TRANSFORMERS_FILE not found, skipping."
fi

# Patch 2: pyannote wespeaker __init__.py (vmap -> list comprehension)
PYANNOTE_FILE="$VENV_DIR/lib/python3.12/site-packages/pyannote/audio/models/embedding/wespeaker/__init__.py"
if [ -f "$PYANNOTE_FILE" ]; then
    echo "  Patching pyannote wespeaker __init__.py ..."
    python3 "$SCRIPT_DIR/patches/apply_pyannote_patch.py" "$PYANNOTE_FILE"
    echo "  Done."
else
    echo "  Warning: $PYANNOTE_FILE not found, skipping."
fi

# Patch 3: pyannote io.py — suppress noisy torchcodec warning (noisy on MetaX, harmless)
PYANNOTE_IO="$VENV_DIR/lib/python3.12/site-packages/pyannote/audio/core/io.py"
if [ -f "$PYANNOTE_IO" ]; then
    echo "  Patching pyannote io.py (suppress torchcodec warning) ..."
    python3 "$SCRIPT_DIR/patches/apply_pyannote_io_patch.py" "$PYANNOTE_IO" || true
    echo "  Done."
else
    echo "  Warning: $PYANNOTE_IO not found, skipping."
fi

echo "All patches applied."
