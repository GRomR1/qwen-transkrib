#!/usr/bin/env python3
"""Patch transformers cache_utils.py for Qwen3-ASR compatibility on MetaX.

The bug: DynamicLayer.lazy_initialization creates 1D empty tensors via torch.tensor([]),
but torch.cat(..., dim=-2) fails because 1D tensors don't have dim=-2.

The fix: Create empty tensors with correct shape matching key_states dimensionality.
"""
import re
import sys


def patch_cache_utils(filepath: str) -> bool:
    with open(filepath, "r") as f:
        content = f.read()

    # Check if already patched
    if "PATCH: Create empty tensors with correct shape" in content:
        print(f"  Already patched: {filepath}")
        return False

    # Pattern to match the lazy_initialization method in DynamicLayer
    old_pattern = r'''    def lazy_initialization\(self, key_states: torch\.Tensor\):\n        self\.dtype, self\.device = key_states\.dtype, key_states\.device\n        self\.keys = torch\.tensor\(\[\], dtype=self\.dtype, device=self\.device\)\n        self\.values = torch\.tensor\(\[\], dtype=self\.dtype, device=self\.device\)\n        self\.is_initialized = True'''

    new_code = '''    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        # PATCH: Create empty tensors with correct shape for Qwen3-ASR (4D key_states)
        if key_states.dim() == 4:
            empty_shape = (key_states.shape[0], key_states.shape[1], 0, key_states.shape[3])
        elif key_states.dim() == 3:
            empty_shape = (key_states.shape[0], 0, key_states.shape[2])
        else:
            empty_shape = (0,)
        self.keys = torch.zeros(empty_shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(empty_shape, dtype=self.dtype, device=self.device)
        self.is_initialized = True'''

    new_content, count = re.subn(old_pattern, new_code, content)

    if count == 0:
        print(f"  Warning: Pattern not found in {filepath}")
        return False

    with open(filepath, "w") as f:
        f.write(new_content)

    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cache_utils.py>")
        sys.exit(1)

    result = patch_cache_utils(sys.argv[1])
    sys.exit(0 if result else 1)
