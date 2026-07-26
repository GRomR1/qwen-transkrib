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
    "одна": 1,
    "два": 2,
    "две": 2,
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
}

# Multipliers (all grammatical forms)
_MULTIPLIERS: dict[str, int] = {
    "тысяча": 1000,
    "тысячи": 1000,
    "тысяч": 1000,
    "тысячу": 1000,
    "миллион": 1_000_000,
    "миллиона": 1_000_000,
    "миллионов": 1_000_000,
    "миллиард": 1_000_000_000,
    "миллиарда": 1_000_000_000,
    "миллиардов": 1_000_000_000,
}

_ORD_MAP = {
    "первый": "1", "второй": "2", "третий": "3", "четвертый": "4",
    "пятый": "5", "шестой": "6", "седьмой": "7", "восьмой": "8",
    "девятый": "9", "десятый": "10",
    "двадцатый": "20", "тридцатый": "30", "сороковый": "40",
    "пятидесятый": "50", "шестидесятый": "60",
}

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (module level, built once)
# ---------------------------------------------------------------------------

_ALL_NUM_WORDS = sorted(
    set(_NUM_MAP.keys()) | set(_MULTIPLIERS.keys()),
    key=len,
    reverse=True,
)
_alt = "|".join(re.escape(w) for w in _ALL_NUM_WORDS)

# Multi-word number patterns: 2-4 consecutive number words
_RE_MULTI_WORD = re.compile(
    rf"\b(?:{_alt})\b(?:\s+\b(?:{_alt})\b)+",
    re.IGNORECASE,
)

# Single-word number replacements (including multipliers)
_RE_SINGLE_WORD = re.compile(
    rf"\b(?:{_alt})\b",
    re.IGNORECASE,
)

# Ordinal replacements
_ord_alt = "|".join(re.escape(w) for w in _ORD_MAP)
_RE_ORDINALS = re.compile(rf"\b(?:{_ord_alt})\b", re.IGNORECASE)


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
        # Pre-compile glossary regexes
        self._glossary_re = [
            (re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE), dst)
            for src, dst in self.glossary.items()
        ]

    def normalize(self, text: str) -> str:
        if not text.strip():
            return text

        # Apply glossary first (may include number-like terms)
        for pattern, dst in self._glossary_re:
            text = pattern.sub(dst, text)

        # Number normalization
        if self.enable_number_norm:
            text = _normalize_numbers(text)

        # Ordinal normalization
        if self.enable_ordinal_norm:
            text = _RE_ORDINALS.sub(lambda m: _ORD_MAP[m.group(0).lower()], text)

        return text


def _parse_number_words(parts: list[str]) -> int:
    """Parse a sequence of Russian number words into an integer.

    Handles multipliers (тысяча, миллион, миллиард) correctly:
    - Small numbers (0-999) are accumulated: "двадцать пять" = 25
    - Multipliers multiply the accumulated sum: "пять тысяч" = 5000
    - Complex: "двадцать пять тысяч сто двадцать три" = 25123
    """
    total = 0
    current = 0

    for p in parts:
        if p in _MULTIPLIERS:
            # Multiply accumulated sum by the multiplier
            mult = _MULTIPLIERS[p]
            if current == 0:
                current = 1  # handle bare "тысяча" = 1000
            total += current * mult
            current = 0
        elif p in _NUM_MAP:
            current += _NUM_MAP[p]
        # ignore unknown words

    return total + current


def _replace_multi(match: re.Match) -> str:
    """Callback for multi-word number pattern replacement."""
    parts = match.group(0).lower().split()
    result = _parse_number_words(parts)
    if result > 0:
        return str(result)
    return match.group(0)


def _replace_single(match: re.Match) -> str:
    """Callback for single-word number replacement."""
    word = match.group(0).lower()
    if word in _MULTIPLIERS:
        return str(_MULTIPLIERS[word])
    if word in _NUM_MAP:
        return str(_NUM_MAP[word])
    return match.group(0)


def _normalize_numbers(text: str) -> str:
    """Replace Russian number words with digits (single pass)."""
    # Multi-word patterns first (greedy, longer matches)
    text = _RE_MULTI_WORD.sub(_replace_multi, text)
    # Single-word leftovers (including bare multipliers)
    text = _RE_SINGLE_WORD.sub(_replace_single, text)
    return text
