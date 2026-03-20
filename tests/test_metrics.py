"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    compute_cer,
    compute_exact_match,
    compute_per_sample_cer,
    edit_distance,
)


class TestEditDistance:
    """Tests for edit distance computation."""

    def test_identical_strings(self):
        """Identical strings should have zero distance."""
        assert edit_distance("hello", "hello") == 0

    def test_empty_strings(self):
        """Empty vs non-empty should equal length."""
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "") == 3

    def test_single_edit(self):
        """Single character difference should be 1."""
        assert edit_distance("cat", "bat") == 1  # Substitution
        assert edit_distance("cat", "cats") == 1  # Insertion
        assert edit_distance("cats", "cat") == 1  # Deletion


class TestCER:
    """Tests for Character Error Rate computation."""

    def test_perfect_predictions(self):
        """Perfect predictions should have 0 CER."""
        preds = ["ᚋᚐᚊᚔ", "ᚉᚓᚂᚔ"]
        refs = ["ᚋᚐᚊᚔ", "ᚉᚓᚂᚔ"]
        assert compute_cer(preds, refs) == 0.0

    def test_complete_mismatch(self):
        """Complete mismatch should have high CER."""
        preds = ["ᚁᚁᚁᚁ"]
        refs = ["ᚋᚐᚊᚔ"]
        cer = compute_cer(preds, refs)
        assert cer > 0.5

    def test_empty_lists(self):
        """Empty lists should return 0."""
        assert compute_cer([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        """Mismatched list lengths should raise."""
        with pytest.raises(ValueError):
            compute_cer(["a", "b"], ["a"])


class TestExactMatch:
    """Tests for exact match accuracy."""

    def test_all_correct(self):
        """All correct should be 1.0."""
        preds = ["ᚋᚐᚊᚔ", "ᚉᚓᚂᚔ"]
        refs = ["ᚋᚐᚊᚔ", "ᚉᚓᚂᚔ"]
        assert compute_exact_match(preds, refs) == 1.0

    def test_all_wrong(self):
        """All wrong should be 0.0."""
        preds = ["ᚁᚁᚁᚁ", "ᚁᚁᚁᚁ"]
        refs = ["ᚋᚐᚊᚔ", "ᚉᚓᚂᚔ"]
        assert compute_exact_match(preds, refs) == 0.0

    def test_half_correct(self):
        """Half correct should be 0.5."""
        preds = ["ᚋᚐᚊᚔ", "ᚁᚁᚁᚁ"]
        refs = ["ᚋᚐᚊᚔ", "ᚉᚓᚂᚔ"]
        assert compute_exact_match(preds, refs) == 0.5


class TestPerSampleCER:
    """Tests for per-sample CER computation."""

    def test_returns_list(self):
        """Should return list of CER values."""
        preds = ["ᚋᚐᚊᚔ", "ᚁᚁᚁᚁ"]
        refs = ["ᚋᚐᚊᚔ", "ᚉᚓᚂᚔ"]
        cers = compute_per_sample_cer(preds, refs)
        assert isinstance(cers, list)
        assert len(cers) == 2

    def test_first_correct_second_wrong(self):
        """First should be 0, second should be > 0."""
        preds = ["ᚋᚐᚊᚔ", "ᚁᚁᚁᚁ"]
        refs = ["ᚋᚐᚊᚔ", "ᚉᚓᚂᚔ"]
        cers = compute_per_sample_cer(preds, refs)
        assert cers[0] == 0.0
        assert cers[1] > 0.0
