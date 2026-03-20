"""Utility modules for Ogham OCR pipeline."""

from .ogham import (
    # Constants
    ALL_CONSONANTS,
    ALL_VOWELS,
    ALL_FORFEDA,
    ALL_LETTERS,
    ALL_CHARACTERS,
    LATIN_TO_OGHAM,
    OGHAM_TO_LATIN,
    LETTER_FREQUENCIES,
    COMMON_PATTERNS,
    OGHAM_SPACE,
    # Functions
    is_ogham_character,
    is_valid_ogham_letter,
    validate_ogham_string,
    get_character_info,
    latin_to_ogham,
    ogham_to_latin,
    count_characters,
    normalize_ogham,
    estimate_difficulty,
    render_ogham_table,
)

__all__ = [
    "ALL_CONSONANTS",
    "ALL_VOWELS",
    "ALL_FORFEDA",
    "ALL_LETTERS",
    "ALL_CHARACTERS",
    "LATIN_TO_OGHAM",
    "OGHAM_TO_LATIN",
    "LETTER_FREQUENCIES",
    "COMMON_PATTERNS",
    "OGHAM_SPACE",
    "is_ogham_character",
    "is_valid_ogham_letter",
    "validate_ogham_string",
    "get_character_info",
    "latin_to_ogham",
    "ogham_to_latin",
    "count_characters",
    "normalize_ogham",
    "estimate_difficulty",
    "render_ogham_table",
]
