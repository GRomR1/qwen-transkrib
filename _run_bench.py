#!/usr/bin/env python3
"""Run benchmark and save results."""
import sys
from qwen_transkrib.bench import run_bench
from qwen_transkrib.asr import create_backend

# Supported datasets
DATASETS = {
    "podlodka": "bond005/podlodka_speech",      # Long audio (8-60s)
    "golos": "bond005/sberdevices_golos_10h_crowd",  # Short commands (0.7-27s)
}

dataset_key = sys.argv[1] if len(sys.argv) > 1 else "podlodka"
dataset_name = DATASETS.get(dataset_key, dataset_key)

print(f"Dataset: {dataset_name}", flush=True)
print("Loading backend...", flush=True)
asr = create_backend("gigaam")
print("Starting bench (10 samples)...", flush=True)
r = run_bench(
    dataset_name,
    "test",
    lambda p: asr.transcribe(p),
    max_samples=10,
)
print(f"WER: {r['wer']}%")
print(f"Sub: {r['substitutions']}, Ins: {r['insertions']}, Del: {r['deletions']}, Ref: {r['ref_words']}, Samples: {r['samples']}")
with open("/tmp/bench_result.txt", "w") as f:
    f.write(f"WER: {r['wer']}%\n")
    f.write(f"Sub: {r['substitutions']}, Ins: {r['insertions']}, Del: {r['deletions']}, Ref: {r['ref_words']}, Samples: {r['samples']}\n")
print("DONE", flush=True)
