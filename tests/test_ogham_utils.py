"""Tests for Ogham character utilities."""

import pytest
from src.utils.ogham import (
    is_ogham_character,
    validate_ogham_string,
    latin_to_ogham,
    ogham_to_latin,
    estimate_difficulty,
    normalize_ogham,
    ALL_CONSONANTS,
    ALL_VOWELS,
)


class TestOghamValidation:
    """Tests for Ogham character validation."""

    def test_is_ogham_character_valid(self):
        """Test that valid Ogham characters are recognized."""
        for char in ALL_CONSONANTS + ALL_VOWELS:
            assert is_ogham_character(char), f"'{char}' should be valid Ogham"

    def test_is_ogham_character_invalid(self):
        """Test that non-Ogham characters are rejected."""
        assert not is_ogham_character("A")
        assert not is_ogham_character("1")
        assert not is_ogham_character(" ")
        assert not is_ogham_character("")

    def test_validate_ogham_string_valid(self):
        """Test validation of valid Ogham strings."""
        valid_string = "ᚋᚐᚊᚔ"  # MAQI
        is_valid, message = validate_ogham_string(valid_string)
        assert is_valid
        assert "Valid" in message

    def test_validate_ogham_string_invalid(self):
        """Test validation rejects strings with non-Ogham characters."""
        invalid_string = "ᚋᚐᚊᚔX"  # Has Latin X
        is_valid, message = validate_ogham_string(invalid_string)
        assert not is_valid
        assert "Invalid" in message

    def test_validate_ogham_string_empty(self):
        """Test validation rejects empty strings."""
        is_valid, message = validate_ogham_string("")
        assert not is_valid


class TestTransliteration:
    """Tests for Latin/Ogham transliteration."""

    def test_latin_to_ogham_simple(self):
        """Test basic Latin to Ogham conversion."""
        result = latin_to_ogham("MAQI")
        assert result == "ᚋᚐᚊᚔ"

    def test_latin_to_ogham_with_digraph(self):
        """Test handling of NG digraph."""
        result = latin_to_ogham("NGETAL")
        assert "ᚍ" in result  # NG character

    def test_ogham_to_latin_roundtrip(self):
        """Test roundtrip conversion."""
        original = "MAQI"
        ogham = latin_to_ogham(original)
        back = ogham_to_latin(ogham)
        assert back == original


class TestDifficultyEstimation:
    """Tests for difficulty scoring."""

    def test_difficulty_short_common(self):
        """Short common patterns should be easy."""
        text = "ᚋᚐᚊᚔ"  # MAQI - very common
        difficulty = estimate_difficulty(text)
        assert difficulty < 0.5

    def test_difficulty_long_rare(self):
        """Long strings with rare characters should be hard."""
        text = "ᚋᚐᚊᚔᚈᚈᚐᚄᚍᚎᚏᚕᚖᚗ"  # Long with rare chars
        difficulty = estimate_difficulty(text)
        assert difficulty > 0.5

    def test_difficulty_empty(self):
        """Empty string should have zero difficulty."""
        assert estimate_difficulty("") == 0.0


class TestNormalization:
    """Tests for Ogham string normalization."""

    def test_normalize_removes_punctuation(self):
        """Test that punctuation is removed."""
        text = "᚛ᚋᚐᚊᚔ᚜"  # With start/end markers
        normalized = normalize_ogham(text)
        assert "᚛" not in normalized
        assert "᚜" not in normalized
        assert "ᚋᚐᚊᚔ" == normalized
