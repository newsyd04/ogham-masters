"""
Ogham character sequence sampler for synthetic data generation.

Generates realistic Ogham text sequences based on character frequency
distributions and common patterns found in real inscriptions.

★ Insight ─────────────────────────────────────
Key considerations for realistic sequences:
1. Character frequencies match CIIC corpus statistics
2. Common formulaic patterns (MAQI, MUCOI) appear frequently
3. Length distribution matches real inscriptions (5-25 chars)
4. Forfeda (rare supplementary letters) are very uncommon
─────────────────────────────────────────────────
"""

from typing import Dict, List, Optional
import numpy as np

from ..utils.ogham import (
    ALL_CONSONANTS,
    ALL_VOWELS,
    ALL_FORFEDA,
    LETTER_FREQUENCIES,
    COMMON_PATTERNS,
    GENEALOGICAL_TEMPLATES,
)


class OghamSequenceSampler:
    """
    Sample realistic Ogham character sequences.

    Generates text that mimics the statistical properties of real
    Ogham inscriptions, including character frequencies and common
    formulaic patterns.
    """

    def __init__(
        self,
        min_length: int = 3,
        max_length: int = 25,
        include_forfeda: bool = False,
        use_realistic_distribution: bool = True,
        common_pattern_probability: float = 0.30,
        genealogy_probability: float = 0.25,
        seed: Optional[int] = None,
    ):
        """
        Initialize sequence sampler.

        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            include_forfeda: Include rare forfeda characters
            use_realistic_distribution: Use corpus-based frequencies
            common_pattern_probability: Probability of including common patterns
            genealogy_probability: Probability of using full genealogical templates
            seed: Random seed for reproducibility
        """
        self.min_length = min_length
        self.max_length = max_length
        self.include_forfeda = include_forfeda
        self.use_realistic_distribution = use_realistic_distribution
        self.common_pattern_probability = common_pattern_probability
        self.genealogy_probability = genealogy_probability

        self.rng = np.random.default_rng(seed)

        # Vowel set for phonotactic validation
        self._vowels = set(ALL_VOWELS)

        # Build sampling distribution
        self._build_distribution()

    def _build_distribution(self):
        """Build character sampling distribution."""
        if self.use_realistic_distribution:
            # Use corpus-based frequencies
            chars = list(LETTER_FREQUENCIES.keys())
            weights = list(LETTER_FREQUENCIES.values())
        else:
            # Uniform distribution
            chars = ALL_CONSONANTS + ALL_VOWELS
            weights = [1.0] * len(chars)

        if self.include_forfeda:
            chars.extend(ALL_FORFEDA)
            # Very low frequency for forfeda
            weights.extend([0.001] * len(ALL_FORFEDA))

        # Normalize weights
        total = sum(weights)
        self.char_probs = [w / total for w in weights]
        self.chars = chars

    def sample(self) -> str:
        """
        Generate a single Ogham sequence.

        Three generation strategies:
        1. Full genealogical template from real Irish inscriptions
        2. Partial pattern (MAQI, MUCOI, etc.) + random extension
        3. Random with phonotactic validation

        Returns:
            Unicode Ogham string
        """
        r = self.rng.random()

        if r < self.genealogy_probability:
            return self._sample_genealogy()
        elif r < self.genealogy_probability + self.common_pattern_probability:
            return self._sample_with_pattern()

        # Phonotactically-validated random sampling
        return self._sample_random_validated()

    def _sample_with_pattern(self) -> str:
        """Generate sequence that includes a common pattern."""
        pattern = self.rng.choice(COMMON_PATTERNS)

        # Optionally extend with random characters
        if self.rng.random() < 0.7:  # 70% chance to extend
            extra_len = self.rng.integers(0, 8)
            extra = self._sample_random_chars(extra_len)

            # 50% chance to prepend vs append
            if self.rng.random() < 0.5:
                return extra + pattern
            else:
                return pattern + extra

        return pattern

    def _sample_random_chars(self, length: int) -> str:
        """Sample random characters according to frequency distribution."""
        if length <= 0:
            return ""

        indices = self.rng.choice(len(self.chars), size=length, p=self.char_probs)
        return "".join(self.chars[i] for i in indices)

    def _sample_genealogy(self) -> str:
        """Sample a complete genealogical formula from real Irish inscriptions."""
        return str(self.rng.choice(GENEALOGICAL_TEMPLATES))

    def _sample_random_validated(self) -> str:
        """Sample random chars with basic phonotactic validation.

        Rejects sequences with >3 consecutive consonants or >3 consecutive
        vowels, which rarely occur in Primitive Irish names.
        """
        length = int(self.rng.integers(self.min_length, self.max_length + 1))

        for _ in range(10):
            seq = self._sample_random_chars(length)
            if self._is_phonotactically_valid(seq):
                return seq
        return seq  # fallback to last attempt

    def _is_phonotactically_valid(self, text: str) -> bool:
        """Check basic phonotactic constraints of Primitive Irish.

        Real Ogham names follow CVCV-like patterns. While geminate
        consonants (CC, TT, LL) are common, runs of 4+ consonants
        or 4+ vowels are extremely rare.
        """
        consonant_run = 0
        vowel_run = 0

        for char in text:
            if char in self._vowels:
                vowel_run += 1
                consonant_run = 0
            else:
                consonant_run += 1
                vowel_run = 0

            if consonant_run > 3 or vowel_run > 3:
                return False

        return True

    def sample_batch(self, n: int) -> List[str]:
        """
        Generate multiple sequences.

        Args:
            n: Number of sequences to generate

        Returns:
            List of Ogham strings
        """
        return [self.sample() for _ in range(n)]

    def get_length_distribution(self) -> Dict[int, float]:
        """
        Get the length distribution of generated sequences.

        Useful for verifying the sampler matches real data.
        """
        # Sample many sequences and compute length histogram
        samples = self.sample_batch(10000)
        lengths = [len(s) for s in samples]

        hist = {}
        for length in range(self.min_length, self.max_length + 1):
            hist[length] = lengths.count(length) / len(lengths)

        return hist

    def set_seed(self, seed: int):
        """Reset random generator with new seed."""
        self.rng = np.random.default_rng(seed)


class DifficultyAwareSequenceSampler(OghamSequenceSampler):
    """
    Sequence sampler that generates text based on difficulty level.

    Used for curriculum learning - start with easy (short, common)
    sequences and progress to harder (long, rare characters).
    """

    DIFFICULTY_CONFIGS = {
        "easy": {
            "min_length": 3,
            "max_length": 8,
            "include_forfeda": False,
            "common_pattern_probability": 0.35,
            "genealogy_probability": 0.30,  # More real patterns for easy start
        },
        "medium": {
            "min_length": 5,
            "max_length": 15,
            "include_forfeda": False,
            "common_pattern_probability": 0.30,
            "genealogy_probability": 0.25,
        },
        "hard": {
            "min_length": 8,
            "max_length": 25,
            "include_forfeda": True,
            "common_pattern_probability": 0.10,
            "genealogy_probability": 0.15,
        },
    }

    def __init__(self, difficulty: str = "medium", seed: Optional[int] = None):
        """
        Initialize with difficulty level.

        Args:
            difficulty: "easy", "medium", or "hard"
            seed: Random seed
        """
        if difficulty not in self.DIFFICULTY_CONFIGS:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        config = self.DIFFICULTY_CONFIGS[difficulty]
        super().__init__(seed=seed, **config)

        self.difficulty = difficulty

    def set_difficulty(self, difficulty: str):
        """Change difficulty level."""
        if difficulty not in self.DIFFICULTY_CONFIGS:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        config = self.DIFFICULTY_CONFIGS[difficulty]
        self.min_length = config["min_length"]
        self.max_length = config["max_length"]
        self.include_forfeda = config["include_forfeda"]
        self.common_pattern_probability = config["common_pattern_probability"]

        self._build_distribution()
        self.difficulty = difficulty


def sample_training_sequences(
    n: int,
    difficulty: str = "medium",
    seed: int = 42,
) -> List[str]:
    """
    Convenience function to sample training sequences.

    Args:
        n: Number of sequences
        difficulty: Difficulty level
        seed: Random seed

    Returns:
        List of Ogham strings
    """
    sampler = DifficultyAwareSequenceSampler(difficulty=difficulty, seed=seed)
    return sampler.sample_batch(n)
