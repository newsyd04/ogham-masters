"""
Evaluation metrics for Ogham OCR.

Provides character-level and word-level metrics for OCR evaluation.

★ Insight ─────────────────────────────────────
Key metrics for OCR:
- CER (Character Error Rate): Primary metric for OCR
- Exact Match: Percentage of perfect predictions
- WER (Word Error Rate): If word boundaries defined
Lower is better for all metrics.
─────────────────────────────────────────────────
"""

from typing import List, Optional, Tuple

try:
    import editdistance
    EDITDISTANCE_AVAILABLE = True
except ImportError:
    EDITDISTANCE_AVAILABLE = False


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein distance (edit distance).

    Fallback implementation when editdistance package not available.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def edit_distance(pred: str, ref: str) -> int:
    """
    Compute edit distance between two strings.

    Args:
        pred: Predicted string
        ref: Reference string

    Returns:
        Levenshtein distance
    """
    if EDITDISTANCE_AVAILABLE:
        return editdistance.eval(pred, ref)
    return _levenshtein_distance(pred, ref)


def compute_cer(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute Character Error Rate (CER).

    CER = (Substitutions + Insertions + Deletions) / Total Reference Characters

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        CER as a float (0.0 = perfect, higher = worse)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if not predictions:
        return 0.0

    total_errors = 0
    total_chars = 0

    for pred, ref in zip(predictions, references):
        total_errors += edit_distance(pred, ref)
        total_chars += len(ref)

    return total_errors / total_chars if total_chars > 0 else 0.0


def compute_wer(
    predictions: List[str],
    references: List[str],
    delimiter: str = " ",
) -> float:
    """
    Compute Word Error Rate (WER).

    Note: Traditional Ogham inscriptions don't have word separators,
    so this metric may not be meaningful for all data.

    Args:
        predictions: List of predicted strings
        references: List of reference strings
        delimiter: Word delimiter

    Returns:
        WER as a float
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if not predictions:
        return 0.0

    total_errors = 0
    total_words = 0

    for pred, ref in zip(predictions, references):
        pred_words = pred.split(delimiter)
        ref_words = ref.split(delimiter)

        total_errors += edit_distance(pred_words, ref_words)
        total_words += len(ref_words)

    return total_errors / total_words if total_words > 0 else 0.0


def compute_exact_match(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute exact match accuracy.

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Proportion of exact matches (0.0 to 1.0)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if not predictions:
        return 0.0

    matches = sum(1 for p, r in zip(predictions, references) if p == r)
    return matches / len(predictions)


def compute_per_sample_cer(
    predictions: List[str],
    references: List[str],
) -> List[float]:
    """
    Compute CER for each sample individually.

    Useful for error analysis.

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        List of CER values per sample
    """
    cers = []
    for pred, ref in zip(predictions, references):
        if len(ref) == 0:
            cer = 1.0 if len(pred) > 0 else 0.0
        else:
            cer = edit_distance(pred, ref) / len(ref)
        cers.append(cer)
    return cers


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
) -> dict:
    """
    Compute all standard metrics.

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Dictionary with all metrics
    """
    return {
        "cer": compute_cer(predictions, references),
        "exact_match": compute_exact_match(predictions, references),
        "n_samples": len(predictions),
    }


def analyze_errors(
    predictions: List[str],
    references: List[str],
    top_n: int = 10,
) -> dict:
    """
    Analyze common error patterns.

    Args:
        predictions: List of predicted strings
        references: List of reference strings
        top_n: Number of top errors to return

    Returns:
        Dictionary with error analysis
    """
    from collections import Counter

    # Count character-level errors
    substitutions = Counter()
    insertions = Counter()
    deletions = Counter()

    for pred, ref in zip(predictions, references):
        # Simple alignment-free error counting
        # (For detailed analysis, would need proper alignment)
        pred_set = set(pred)
        ref_set = set(ref)

        # Characters in ref but not pred (deletions)
        for c in ref_set - pred_set:
            deletions[c] += ref.count(c) - pred.count(c)

        # Characters in pred but not ref (insertions)
        for c in pred_set - ref_set:
            insertions[c] += pred.count(c) - ref.count(c)

    return {
        "top_deletions": deletions.most_common(top_n),
        "top_insertions": insertions.most_common(top_n),
        "total_deletion_chars": sum(deletions.values()),
        "total_insertion_chars": sum(insertions.values()),
    }
