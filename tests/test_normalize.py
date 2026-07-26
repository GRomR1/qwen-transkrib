"""Tests for number normalization and glossary post-processing."""

from __future__ import annotations

import pytest

from qwen_transkrib.cli import _parse_glossary
from qwen_transkrib.normalize import TextNormalizer


class TestParseGlossary:
    def test_empty(self) -> None:
        assert _parse_glossary("") == {}

    def test_single_pair(self) -> None:
        assert _parse_glossary("гугл=Google") == {"гугл": "Google"}

    def test_multiple_pairs(self) -> None:
        result = _parse_glossary("гугл=Google,майкрософт=Microsoft")
        assert result == {"гугл": "Google", "майкрософт": "Microsoft"}

    def test_spaces_are_stripped(self) -> None:
        result = _parse_glossary("  гугл = Google  ,  майкрософт = Microsoft  ")
        assert result == {"гугл": "Google", "майкрософт": "Microsoft"}


class TestNumberNormalization:
    @pytest.fixture
    def normalizer(self) -> TextNormalizer:
        return TextNormalizer()

    def test_single_digit(self, normalizer: TextNormalizer) -> None:
        assert "5" in normalizer.normalize("пять человек")

    def test_tens(self, normalizer: TextNormalizer) -> None:
        assert "20" in normalizer.normalize("двадцать лет")

    def test_tens_plus_ones(self, normalizer: TextNormalizer) -> None:
        result = normalizer.normalize("двадцать пять рублей")
        assert "25" in result

    def test_hundreds(self, normalizer: TextNormalizer) -> None:
        result = normalizer.normalize("сто двадцать три")
        assert "123" in result

    def test_ordinal(self, normalizer: TextNormalizer) -> None:
        result = normalizer.normalize("пятый этаж")
        assert "5" in result

    def test_no_false_positives(self, normalizer: TextNormalizer) -> None:
        # "огонь" should not become "онь"
        result = normalizer.normalize("огонь")
        assert "огонь" in result


class TestGlossary:
    def test_glossary_replacement(self) -> None:
        normalizer = TextNormalizer(glossary={"гугл": "Google"})
        result = normalizer.normalize("гугл выпустил новинку")
        assert "Google" in result

    def test_case_insensitive(self) -> None:
        normalizer = TextNormalizer(glossary={"гугл": "Google"})
        result = normalizer.normalize("Гугл выпустил новинку")
        assert "Google" in result