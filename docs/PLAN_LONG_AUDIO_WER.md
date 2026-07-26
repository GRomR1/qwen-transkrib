# Plan: Reduce WER for Russian Long-Form Audio

## 1. Analysis of Articles & Current Implementation

### Key Article Findings

| Article | Key Insight | Impact |
|---------|------------|--------|
| **GigaAM CPU 3.3% WER** | RNNT >> CTC (2.6% vs 13.2% on complex texts); chunking hurts; T5 correction useless on strong models | Switch to RNNT; minimize chunking |
| **Whisper vs GigaAM prod** | Corpus quality matters more than model (15-20pp); don't chunk long audio (~6pp loss); number normalization -0.7pp; disable internal VAD, use Silero | Fix chunking; add number normalization |
| **ASRmy Knife** | Batching 4x speedup; VAD at natural pauses; long audio = main challenge | VAD-aware chunking |
| **LLM as ASR decoder** | SALM hybrid: speech encoder + LLM decoder; >90% inference time in LLM; can be teacher for distillation | Future: distillation pipeline |
| **WER decomposition** | Named entities = 15% of text but 27% of WER errors; decompose by semantic category | Better error analysis |
| **VibeVoice-ASR** | 9B params, 60-min single pass, joint ASR+diarization; too heavy for CPU | Not viable for current use case |

### Current Codebase Issues (all resolved)

1. **GigaAM uses `e2e_ctc`** — ✅ Fixed: switched to `e2e_rnnt` (5x better WER on complex Russian)
2. **Fixed 24s chunking** for GigaAM — ✅ Fixed: VAD-aware chunking with overlap and context carry-over
3. **Qwen3 VAD chunking** also loses context between chunks — ✅ Fixed: added context carry-over
4. **No number normalization** — ✅ Fixed: rule-based normalization in `normalize.py`
5. **No glossary/dictionary** post-processing — ✅ Fixed: `--glossary` flag with `TextNormalizer`
6. **No hotword support** for GigaAM backend — ✅ Fixed: `context` parameter in `transcribe_words()`
7. **Benchmark uses short clips** only — ✅ Fixed: added `bond005/podlodka_speech` dataset (8-60s)
8. **No long-form benchmark dataset** — ✅ Fixed: podlodka_speech as primary benchmark
9. **Native gigaam not used** — ✅ Fixed: installed with `--no-deps` + `strict=False` patch

### GigaAM on CPU

**Verdict: Yes, there is benefit.**
- Article shows 3.3% WER on CPU with GigaAM v3-e2e-rnnt
- 2.4x better than Whisper large-v3-turbo on RTX 4090 (7.9%)
- 240M params, ~1.5 GB VRAM, fast inference
- **BUT**: current code uses `e2e_ctc` (13.2% WER on complex texts) instead of `e2e_rnnt` (2.6%)

### VibeVoice-ASR Assessment

- 9B params — too heavy for CPU, requires significant GPU
- 60-minute single-pass — solves chunking problem
- Joint ASR + diarization + timestamps
- **Not viable** for current CPU/GPU constraints
- **Future option** if GPU resources expand

### Long Audio Datasets

| Dataset | Avg Duration | Total Samples | Domain |
|---------|-------------|---------------|--------|
| `bond005/sberdevices_golos_10h_crowd` | 0.7-27.5s | 18.8K | Crowdsourced commands |
| `bond005/podlodka_speech` | 6-47s | 107 | Podcast (tech) |
| `bond005/sberdevices_golos_10h_farfield` | ~10-30s | ~8K | Farfield (noisy) |

**Selected: `bond005/podlodka_speech`** — longer clips (6-47s vs 0.7-27.5s), natural speech, technical domain.

## 2. Implementation Plan

### Phase 1: Switch GigaAM to e2e_rnnt (Highest Impact)

**File**: `src/qwen_transkrib/asr.py`

- Change `GigaAMBackend` default revision from `"e2e_ctc"` to `"e2e_rnnt"`
- Update `create_backend` to use `revision="e2e_rnnt"`
- Update `cli.py` references
- Update `ARCHITECTURE.md` and `BENCHMARKS.md`

**Expected WER improvement**: 5-10pp (from 13.2% to 2.6-8% on complex texts)

### Phase 2: VAD-Aware Chunking with Context Carry-Over

**File**: `src/qwen_transkrib/asr.py`, `src/qwen_transkrib/vad.py`

- Replace fixed 24s chunking with VAD-aware chunking
- Use Silero VAD to find natural pause boundaries
- Chunk at VAD boundaries with overlap (1-2s)
- **Context carry-over**: pass end-of-previous-chunk text as context to next chunk
- For GigaAM native: use `model.transcribe(path)` per chunk (RNNT handles context better than CTC)

**Expected WER improvement**: 2-4pp (avoid boundary errors)

### Phase 3: Number Normalization Post-Processing

**File**: `src/qwen_transkrib/correct.py` (new module or extend)

- Russian number word → digit conversion:
  - "пять" → "5", "тридцать три" → "33", "двадцать седьмое" → "27"
- Applied after ASR, before punctuation
- Configurable (on/off)

**Expected WER improvement**: 0.5-1pp

### Phase 4: Glossary/Dictionary Post-Processing

**File**: `src/qwen_transkrib/correct.py`

- Custom term replacement: "гугл" → "Google", "майкрософт" → "Microsoft"
- Configurable via `--context` flag (already exists for Qwen3, extend to GigaAM)
- Applied after ASR

**Expected WER improvement**: 0.5-2pp (domain-specific)

### Phase 5: Long-Form Benchmark Dataset

**File**: `src/qwen_transkrib/bench.py`

- Add support for `bond005/podlodka_speech` dataset
- Add `--long-audio` flag to benchmark that concatenates VAD segments
- Add WER decomposition by segment length (short vs long)

### Phase 6: Hotword Support for GigaAM

**File**: `src/qwen_transkrib/asr.py`

- Extend `GigaAMBackend.transcribe_words()` to accept context
- Use context for term correction in post-processing

### Phase 7: WER Decomposition

**File**: `src/qwen_transkrib/bench.py`

- Track errors on named entities vs common words
- Report per-category WER breakdown

## 3. Execution Order

1. Phase 1: Switch to e2e_rnnt (highest impact, lowest risk)
2. Phase 2: VAD-aware chunking with context carry-over
3. Phase 3: Number normalization
4. Phase 4: Glossary/dictionary post-processing
5. Phase 5: Long-form benchmark dataset
6. Phase 6: Hotword support for GigaAM
7. Phase 7: WER decomposition

## 4. Testing Strategy

- Unit tests for number normalization
- Unit tests for glossary replacement
- Integration tests for VAD-aware chunking
- Benchmark before/after WER comparison
- Run on `bond005/podlodka_speech` dataset
