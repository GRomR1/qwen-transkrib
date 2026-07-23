#!/usr/bin/env python3
"""Patch pyannote wespeaker for MetaX compatibility.

The bug: torch.vmap creates tensors without storage, which breaks kaldi.fbank on MetaX.

The fix: Replace torch.vmap(self._fbank)(waveforms) with list comprehension.
"""
import re
import sys


def patch_wespeaker(filepath: str) -> bool:
    with open(filepath, "r") as f:
        content = f.read()

    # Check if already patched
    if "PATCH: vmap creates tensors without storage" in content:
        print(f"  Already patched: {filepath}")
        return False

    # Pattern to match the vmap line
    old_pattern = r'        features = torch\.vmap\(self\._fbank\)\(waveforms\.to\(fft_device\)\)\.to\(device\)'

    new_code = '''        # PATCH: vmap creates tensors without storage, which breaks our fbank
        # Use list comprehension instead of vmap
        waveforms_cpu = waveforms.to(fft_device)
        feat_list = [self._fbank(w) for w in waveforms_cpu]
        features = torch.stack(feat_list, dim=0).to(device)'''

    new_content, count = re.subn(old_pattern, new_code, content)

    if count == 0:
        print(f"  Warning: Pattern not found in {filepath}")
        return False

    with open(filepath, "w") as f:
        f.write(new_content)

    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <wespeaker/__init__.py>")
        sys.exit(1)

    result = patch_wespeaker(sys.argv[1])
    sys.exit(0 if result else 1)
