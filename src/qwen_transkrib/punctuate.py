"""Punctuation restoration for multiple languages.

Adds missing punctuation (capitalization, periods, commas, question marks)
to raw text produced by ASR models like Qwen3-ASR that output unpunctuated text.

Supported languages:
- Russian: kontur-ai/sbert_punc_case_ru (sbert-based, no "й"/"ё" tokenizer bugs)
- English: oliverguhr/fullstop-punctuation-multilingual-base
- Default: oliverguhr/fullstop-punctuation-multilingual-base (multilingual)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class _KonturPunctuation:
    """Wrapper for kontur-ai/sbert_punc_case_ru (Russian)."""

    def __init__(self, device: str = "cpu") -> None:
        from sbert_punc_case_ru import SbertPuncCase

        logger.info("Loading kontur-ai/sbert_punc_case_ru (device=%s)", device)
        self._model = SbertPuncCase()
        logger.info("kontur-ai model loaded")

    def restore(self, text: str) -> str:
        if not text.strip():
            return text
        # Model expects lowercase input (designed for ASR output)
        return self._model.punctuate(text.lower())


class _TransformersPunctuation:
    """Wrapper for HuggingFace token-classification models (English, multilingual)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        # Label mapping: punctuation goes BEFORE the token (B/I scheme)
        self.label_map = {
            "B-,": ",",
            "I-,": ",",
            "B-.": ".",
            "I-.": ".",
            "B-?": "?",
            "I-?": "?",
            "B-!": "!",
            "I-!": "!",
        }

        logger.info("Loading punctuation model: %s (device=%s)", model_name, device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.model.to(device).eval()
        logger.info("Punctuation model loaded")

    @torch.inference_mode()
    def restore(self, text: str) -> str:
        if not text.strip():
            return text

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)

        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.model.config.id2label[p.item()] for p in predictions[0]]
        word_ids = inputs.word_ids(batch_index=0)

        # Group tokens by word
        words: list[list[tuple[str, str]]] = []
        current_word_id: int | None = None
        for token, label, wid in zip(tokens, labels, word_ids):
            if wid is None:
                continue
            if wid != current_word_id:
                words.append([])
                current_word_id = wid
            words[-1].append((token, label))

        # Reconstruct text with punctuation at word boundaries
        result: list[str] = []
        capitalize_next = True

        for word_tokens in words:
            raw = "".join(t.lstrip("#") for t, _ in word_tokens)

            punct_label: str | None = None
            for _, lbl in word_tokens:
                if lbl in self.label_map:
                    punct_label = self.label_map[lbl]

            if capitalize_next and raw:
                raw = raw[0].upper() + raw[1:]
                capitalize_next = False

            if result:
                result.append(" " + raw)
            else:
                result.append(raw)

            if punct_label:
                result[-1] += punct_label
                if punct_label in (".", "?", "!"):
                    capitalize_next = True

        result_text = "".join(result)
        if result_text and result_text[0].islower():
            result_text = result_text[0].upper() + result_text[1:]

        return result_text


# Language -> model factory
_MODEL_FACTORIES = {
    "Russian": lambda device: _KonturPunctuation(device),
    "English": lambda device: _TransformersPunctuation(
        "oliverguhr/fullstop-punctuation-multilingual-base", device
    ),
}
_DEFAULT_FACTORY = lambda device: _TransformersPunctuation(
    "oliverguhr/fullstop-punctuation-multilingual-base", device
)


class PunctuationModel:
    """Unified punctuation restoration interface.

    Usage::

        # Russian (kontur-ai/sbert_punc_case_ru)
        punct = PunctuationModel(language="Russian")
        result = punct.restore("привет как дела")
        # → "Привет, как дела?"

        # English (multilingual)
        punct = PunctuationModel(language="English")
        result = punct.restore("hello how are you")
        # → "Hello, how are you?"
    """

    def __init__(
        self,
        language: str = "Russian",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language = language

        # If custom model_name provided, use transformers path
        if model_name is not None:
            self._impl = _TransformersPunctuation(model_name, device)
        else:
            factory = _MODEL_FACTORIES.get(language, _DEFAULT_FACTORY)
            self._impl = factory(device)

    def restore(self, text: str) -> str:
        """Restore punctuation in raw text."""
        return self._impl.restore(text)
