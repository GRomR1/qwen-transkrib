"""ASR error correction via zero-shot LLM post-processing.

Uses Qwen3-0.6B to fix transcription errors: misspellings, homophones,
grammar mistakes. Works zero-shot — no fine-tuning required.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

_SYSTEM_PROMPT = (
    "Ты — модель исправления ошибок автоматического распознавания речи (ASR). "
    "Исправь только очевидные ошибки: опечатки, однозвучные слова, грамматику. "
    "Не меняй правильные слова. Не меняй стиль. Не добавляй и не удаляй информацию. "
    "Сохрани пунктуацию. Ответь только исправленным текстом."
)

_MAX_CHARS = 1500


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (complete or incomplete) from Qwen3 output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


class CorrectionModel:
    """Zero-shot ASR error correction using Qwen3."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model_name is None:
            model_name = _DEFAULT_MODEL

        logger.info("Loading correction model: %s (device=%s)", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        )
        self.model.to(self.device).eval()
        logger.info("Correction model loaded")

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

    @torch.inference_mode()
    def _generate(self, text: str) -> str:
        messages = self._build_messages(text)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Give enough tokens for thinking + answer
        max_tokens = max(len(text) * 3, 512)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _strip_thinking(result).strip()

    def correct(self, text: str) -> str:
        """Correct ASR errors in text."""
        if not text.strip():
            return text

        chunks: list[str] = []
        current = ""

        for sentence in text.replace(". ", ".|").replace("? ", "?|").replace("! ", "!|").split("|"):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) < _MAX_CHARS:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)

        if len(chunks) <= 1:
            return self._generate(text) if chunks else text

        corrected = []
        for chunk in chunks:
            corrected.append(self._generate(chunk))
        return " ".join(corrected)
