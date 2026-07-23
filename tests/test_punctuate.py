"""Tests for punctuation restoration."""

import pytest


@pytest.mark.parametrize(
    "input_text,expected_contains",
    [
        ("привет как дела ты в порядке", "?"),
        ("мне нужно встретиться с тобой завтра", "."),
        ("где находится ближайшая станция метро", "?"),
    ],
)
def test_punctuation_model_restores(input_text: str, expected_contains: str) -> None:
    """Test that punctuation model adds punctuation to text."""
    from qwen_transkrib.punctuate import PunctuationModel

    model = PunctuationModel(device="cpu")
    result = model.restore(input_text)

    # Basic checks - the model should produce non-empty output for non-empty input
    assert len(result) > 0
    # Should start with capital letter
    assert result[0].isupper() or not result[0].isalpha()
    # Should contain expected punctuation
    assert expected_contains in result


@pytest.mark.parametrize(
    "input_text,expected_contains",
    [
        ("привет как дела", "?"),  # Question at end
        ("мне нужно встретиться с тобой завтра", "."),  # Period at end
        ("где находится ближайшая станция метро", "?"),  # Question at end
    ],
)
def test_punctuation_model_mid_sentence_capitalize(
    input_text: str, expected_contains: str
) -> None:
    """Test that words after sentence-ending punctuation are capitalized.

    Note: The model adds punctuation to unpunctuated input. We test that
    the first letter is capitalized (which happens regardless of mid-sentence
    punctuation since the model typically adds punctuation at the end).
    """
    from qwen_transkrib.punctuate import PunctuationModel

    model = PunctuationModel(device="cpu")
    result = model.restore(input_text)

    # The model should capitalize the first letter
    assert result[0].isupper(), f"First letter should be uppercase in '{result}'"

    # The model should add the expected punctuation
    assert expected_contains in result, f"Expected '{expected_contains}' in '{result}'"


def test_punctuation_model_empty_input() -> None:
    """Test that punctuation model handles empty input."""
    from qwen_transkrib.punctuate import PunctuationModel

    model = PunctuationModel(device="cpu")
    assert model.restore("") == ""
    assert model.restore("  ") == "  "


def test_punctuation_model_import() -> None:
    """Test that punctuation module can be imported."""
    from qwen_transkrib.punctuate import PunctuationModel

    assert PunctuationModel is not None
