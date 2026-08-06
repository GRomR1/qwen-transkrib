#!/usr/bin/env python3
"""Patch pyannote io.py for MetaX compatibility.

The bug: The torchcodec warning on import is noisy on MetaX GPUs (FFmpeg not present).

The fix: Suppress the warning, set TORCHCODEC_AVAILABLE=False and fallback names=None.
"""
import re
import sys


def patch_io(filepath: str) -> bool:
    with open(filepath, "r") as f:
        content = f.read()

    # Check if already patched
    if "torchcodec unavailable — pyannote falls back to torchaudio automatically" in content:
        print(f"  Already patched: {filepath}")
        return False

    # Pattern to match the entire except block with the verbose warning
    old_pattern = (
        r'except Exception as e:\n'
        r'    warnings\.warn\(\n'
        r'        "\\ntorchcodec is not installed correctly.*?\n'
        r'        f"\{e\}"\n'
        r'    \)\n'
        r'    TORCHCODEC_AVAILABLE = False\n'
        r'    AudioDecoder = None\n'
        r'    AudioStreamMetadata = None'
    )

    new_code = '''except Exception:
    # torchcodec unavailable — pyannote falls back to torchaudio automatically.
    # Warning suppressed to avoid noisy logs on MetaX GPUs without FFmpeg.
    TORCHCODEC_AVAILABLE = False
    AudioDecoder = None
    AudioSamples = None
    AudioStreamMetadata = None'''

    new_content, count = re.subn(old_pattern, new_code, content, flags=re.DOTALL)

    if count == 0:
        print(f"  Warning: Pattern not found in {filepath}")
        return False

    with open(filepath, "w") as f:
        f.write(new_content)

    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pyannote/audio/core/io.py>")
        sys.exit(1)

    result = patch_io(sys.argv[1])
    sys.exit(0 if result else 1)
