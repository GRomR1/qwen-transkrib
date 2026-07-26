"""Post-processing: number normalization and glossary/hotword correction.

These lightweight rules reduce WER by 0.5-2pp:
- Number normalization: "пять" → "5", "тридцать три" → "33"
- Glossary replacement: "гугл" → "Google", etc.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Russian number words → digits
_NUM_MAP: dict[str, int] = {
    "ноль": 0,
    "один": 1,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
    "тысяча": 1000,
    "миллион": 1000000,
    "миллиард": 1000000000,
}

# Tens + ones: "двадцать пять" → "25"
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40,
    "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70,
    "восемьдесят": 80, "девяносто": 90,
}
_ONES = {
    "один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}

_ORD_MAP = {
    "первый": "1", "второй": "2", "третий": "3", "четвертый": "4",
    "пятый": "5", "шестой": "6", "седьмой": "7", "восьмой": "8",
    "девятый": "9", "десятый": "10",
    "двадцатый": "20", "тридцатый": "30", "сороковый": "40",
    "пятидесятый": "50", "шестидесятый": "60",
}


class TextNormalizer:
    """Lightweight ASR post-processing: number normalization, glossary."""

    def __init__(
        self,
        glossary: dict[str, str] | None = None,
        enable_number_norm: bool = True,
        enable_ordinal_norm: bool = True,
    ) -> None:
        self.glossary = glossary or {}
        self.enable_number_norm = enable_number_norm
        self.enable_ordinal_norm = enable_ordinal_norm

    def normalize(self, text: str) -> str:
        if not text.strip():
            return text

        # Apply glossary first (may include number-like terms)
        for src, dst in self.glossary.items():
            text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)

        # Number normalization
        if self.enable_number_norm:
            text = self._normalize_numbers(text)

        # Ordinal normalization
        if self.enable_ordinal_norm:
            text = self._normalize_ordinals(text)

        return text

    def _normalize_numbers(self, text: str) -> str:
        """Replace Russian number words with digits."""
        # Multi-word patterns: "двадцать пять", "сто двадцать три"
        def replace_multi(match: re.Match) -> str:
            parts = match.group(0).lower().split()
            total = 0
            for p in parts:
                if p in _ONES:
                    total += _ONES[p]
                elif p in _TENS:
                    total += _TENS[p]
                elif p in _NUM_MAP:
                    total += _NUM_MAP[p]
            if total > 0:
                return str(total)
            return match.group(0)

        # Try multi-word patterns first (2-3 words)
        _num_words = list(_TENS.keys()) + list(_ONES.keys()) + list(_NUM_MAP.keys())
        _alt = "|".join(re.escape(w) for w in _num_words)
        pattern = re.compile(
            rf"\b(?:{_alt})\b(?:\s+\b(?:{_alt})\b){{1,2}}",
            re.IGNORECASE,
        )
        text = pattern.sub(replace_multi, text)

        # Single-word replacements
        for word, digit in _NUM_MAP.items():
            text = re.sub(rf"\b{re.escape(word)}\b", str(digit), text, flags=re.IGNORECASE)

        return text

    def _normalize_ordinals(self, text: str) -> str:
        """Replace Russian ordinal words with digits: "пятый" → "5"."""
        for word, digit in _ORD_MAP.items():
            text = re.sub(rf"\b{re.escape(word)}\b", digit, text, flags=re.IGNORECASE)
        return text