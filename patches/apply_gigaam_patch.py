#!/usr/bin/env python3
"""Patch gigaam to use strict=False for state_dict loading.

This fixes compatibility between gigaam v0.2.0 and MetaX PyTorch 2.10.
The checkpoint has extra keys that the model doesn't expect, but the model
still works correctly without them.

Usage:
    python patches/apply_gigaam_patch.py

Or manually:
    Edit .venv/lib/python3.12/site-packages/gigaam/__init__.py
    Change line 241:
        model.load_state_dict(checkpoint["state_dict"])
    To:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
"""

import sys
from pathlib import Path


def patch_gigaam():
    """Patch gigaam __init__.py to use strict=False."""
    # Find gigaam package
    try:
        import gigaam
        gigaam_path = Path(gigaam.__file__).parent / "__init__.py"
    except ImportError:
        print("Error: gigaam package not installed")
        sys.exit(1)

    # Read current content
    content = gigaam_path.read_text()

    # Check if already patched
    if "strict=False" in content:
        print(f"Already patched: {gigaam_path}")
        return

    # Apply patch
    old_line = 'model.load_state_dict(checkpoint["state_dict"])'
    new_line = 'model.load_state_dict(checkpoint["state_dict"], strict=False)'

    if old_line not in content:
        print(f"Error: Could not find line to patch in {gigaam_path}")
        print("Expected line:", old_line)
        sys.exit(1)

    content = content.replace(old_line, new_line)
    gigaam_path.write_text(content)
    print(f"Patched: {gigaam_path}")
    print(f"Changed: {old_line}")
    print(f"      to: {new_line}")


if __name__ == "__main__":
    patch_gigaam()
