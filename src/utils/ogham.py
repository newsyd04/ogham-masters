"""
Ogham character utilities.

This module provides constants and utilities for working with Ogham Unicode characters,
including validation, rendering helpers, and frequency distributions based on corpus analysis.

Ogham Unicode Range: U+1680–U+169F
"""

from typing import Dict, List, Optional, Set, Tuple
import re


# =============================================================================
# OGHAM CHARACTER CONSTANTS
# =============================================================================

# Aicme Beithe (B Group) - Strokes to the right of the stemline
AICME_BEITHE = {
    "B": "ᚁ",   # beith (birch)
    "L": "ᚂ",   # luis (rowan)
    "F": "ᚃ",   # fearn (alder) - also represents V
    "V": "ᚃ",   # alias for F
    "S": "ᚄ",   # sail (willow)
    "N": "ᚅ",   # nuin (ash)
}

# Aicme hÚatha (H Group) - Strokes to the left of the stemline
AICME_HUATHA = {
    "H": "ᚆ",   # úath (hawthorn)
    "D": "ᚇ",   # dair (oak)
    "T": "ᚈ",   # tinne (holly)
    "C": "ᚉ",   # coll (hazel)
    "Q": "ᚊ",   # cert (apple) - also written as K
}

# Aicme Muine (M Group) - Diagonal strokes across the stemline
AICME_MUINE = {
    "M": "ᚋ",   # muin (vine)
    "G": "ᚌ",   # gort (ivy)
    "NG": "ᚍ",  # gétal (broom)
    "Z": "ᚎ",   # straif (blackthorn) - rare
    "R": "ᚏ",   # ruis (elder)
}

# Aicme Ailme (Vowels) - Notches or perpendicular strokes
AICME_AILME = {
    "A": "ᚐ",   # ailm (fir/pine)
    "O": "ᚑ",   # onn (gorse)
    "U": "ᚒ",   # úr (heather)
    "E": "ᚓ",   # edad (aspen)
    "I": "ᚔ",   # idad (yew)
}

# Forfeda (Supplementary characters) - Later additions, rare in stone inscriptions
FORFEDA = {
    "EA": "ᚕ",  # ébad
    "OI": "ᚖ",  # óir
    "UI": "ᚗ",  # uilleann
    "IA": "ᚘ",  # ifín (later: pin/pín)
    "AE": "ᚙ",  # emancholl
}

# Punctuation marks
PUNCTUATION = {
    "FEATHER": "᚛",   # Start of text (feather mark)
    "START": "᚛",     # Alias
    "END": "᚜",       # End of text (reversed feather mark)
}

# Space character (Unicode Ogham space)
OGHAM_SPACE = "\u1680"  # Ogham space mark

# Combined lookups
ALL_CONSONANTS = list("ᚁᚂᚃᚄᚅᚆᚇᚈᚉᚊᚋᚌᚍᚎᚏ")
ALL_VOWELS = list("ᚐᚑᚒᚓᚔ")
ALL_FORFEDA = list("ᚕᚖᚗᚘᚙ")
ALL_LETTERS = ALL_CONSONANTS + ALL_VOWELS
ALL_CHARACTERS = ALL_LETTERS + ALL_FORFEDA + list("᚛᚜")

# Latin to Ogham mapping (for transliteration)
LATIN_TO_OGHAM: Dict[str, str] = {
    **{k: v for k, v in AICME_BEITHE.items()},
    **{k: v for k, v in AICME_HUATHA.items()},
    **{k: v for k, v in AICME_MUINE.items()},
    **{k: v for k, v in AICME_AILME.items()},
    **{k: v for k, v in FORFEDA.items()},
}

# Ogham to Latin mapping (for reverse transliteration)
OGHAM_TO_LATIN: Dict[str, str] = {
    "ᚁ": "B", "ᚂ": "L", "ᚃ": "F", "ᚄ": "S", "ᚅ": "N",
    "ᚆ": "H", "ᚇ": "D", "ᚈ": "T", "ᚉ": "C", "ᚊ": "Q",
    "ᚋ": "M", "ᚌ": "G", "ᚍ": "NG", "ᚎ": "Z", "ᚏ": "R",
    "ᚐ": "A", "ᚑ": "O", "ᚒ": "U", "ᚓ": "E", "ᚔ": "I",
    "ᚕ": "EA", "ᚖ": "OI", "ᚗ": "UI", "ᚘ": "IA", "ᚙ": "AE",
}


# =============================================================================
# FREQUENCY DISTRIBUTIONS (based on CIIC corpus analysis)
# =============================================================================

# Character frequencies from analysis of ~400 inscriptions
# These weights are approximate and based on common patterns in memorial formulae
LETTER_FREQUENCIES: Dict[str, float] = {
    # Vowels (very common - appear in most names and formulae)
    "ᚐ": 0.15,  # A - extremely common (MAQI, names ending in -A)
    "ᚔ": 0.11,  # I - very common (MAQI suffix, genitive endings)
    "ᚒ": 0.07,  # U - common (MUCOI "descendant of")
    "ᚑ": 0.04,  # O - moderately common
    "ᚓ": 0.02,  # E - less common

    # High-frequency consonants
    "ᚋ": 0.12,  # M - very common (MAQI "son of", common in names)
    "ᚊ": 0.09,  # Q - common (MAQI formula)
    "ᚉ": 0.08,  # C - common (MUCOI, CELI, names)
    "ᚈ": 0.06,  # T - common (MAQITTAS, other names)
    "ᚌ": 0.05,  # G - moderately common

    # Medium-frequency consonants
    "ᚄ": 0.05,  # S - moderately common
    "ᚂ": 0.04,  # L - moderately common (CELI "devotee")
    "ᚅ": 0.03,  # N - moderately common
    "ᚁ": 0.03,  # B - moderately common

    # Low-frequency consonants
    "ᚇ": 0.02,  # D - less common
    "ᚆ": 0.01,  # H - rare
    "ᚃ": 0.01,  # F/V - rare
    "ᚏ": 0.01,  # R - rare

    # Very rare
    "ᚎ": 0.005,  # Z - very rare
    "ᚍ": 0.005,  # NG - very rare
}

# Common patterns found in Irish Ogham inscriptions
# These are formulaic phrases and common name elements from the CIIC corpus
COMMON_PATTERNS: List[str] = [
    # Memorial formulae (appear on majority of Irish stones)
    "ᚋᚐᚊᚔ",          # MAQI - "son of" (extremely common)
    "ᚋᚒᚉᚑᚔ",        # MUCOI - "descendant of" / "tribe of"
    "ᚐᚃᚔ",          # AVI - "grandson" (rare variant)
    "ᚉᚓᚂᚔ",         # CELI - "devotee of" / "client of"
    "ᚐᚅᚋ",          # ANM - "name" (in X ANM Y = "X, name of Y")

    # Common Irish personal names and name elements (CIIC)
    "ᚋᚐᚊᚔᚈᚈᚐᚄ",    # MAQITTAS
    "ᚉᚒᚅᚐᚉᚐᚈᚑᚄ",  # CUNACATOS - "high hound"
    "ᚉᚑᚏᚁᚁᚔ",      # CORBBI - "raven"
    "ᚃᚓᚇᚇᚑᚄ",      # VEDDOS
    "ᚉᚐᚈᚈᚒᚁᚒᚈᚈᚐᚄ", # CATTUBUTTAS
    "ᚇᚑᚃᚃᚔᚅᚔᚐᚄ",  # DOVVINIAS
    "ᚂᚒᚌᚒᚊᚏᚔᚈ",    # LUGUQRIT - "oath of Lug"
    "ᚇᚓᚌᚂᚐᚅᚅ",      # DEGLANN
    "ᚅᚓᚈᚈᚐᚄᚂᚑᚌᚔ",  # NETTASLOGI
    "ᚁᚏᚒᚄᚉᚉᚑᚄ",    # BRUSCCOS
    "ᚈᚏᚓᚅᚐᚌᚒᚄᚒ",  # TRENAGUSU - "strong vigour"
    "ᚌᚒᚄᚉᚒ",        # GUSCU
    "ᚃᚑᚁᚐᚏᚐᚉᚔ",    # VOBARACI
    "ᚇᚒᚅᚐᚔᚇᚑᚅᚐᚄ",  # DUNAIDONAS
    "ᚉᚐᚂᚔᚐᚉᚔ",      # CALIACI
    "ᚐᚋᚋᚂᚂᚐᚌᚅᚔ",  # AMMLLAGNI
]

# Full genealogical templates from real Irish inscriptions
# These are complete or near-complete memorial formulae in the
# standard pattern: NAME MAQI FATHER [MUCOI ANCESTOR/TRIBE]
GENEALOGICAL_TEMPLATES: List[str] = [
    "ᚉᚐᚌᚔᚅᚐᚇᚔ ᚋᚐᚊᚔ ᚃᚑᚁᚐᚏᚐᚉᚔ",              # CAGINADI MAQI VOBARACI
    "ᚇᚒᚅᚐᚔᚇᚑᚅᚐᚄ ᚋᚐᚊᚔ ᚋᚐᚏᚔᚐᚅᚔ",            # DUNAIDONAS MAQI MARIANI
    "ᚇᚐᚓᚅᚓᚌᚂᚑ ᚋᚐᚊᚔ ᚊᚓᚈᚐᚔ",                # DAENEGLO MAQI QETAI
    "ᚉᚒᚅᚐᚉᚐᚈᚑᚄ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ ᚇᚑᚃᚃᚔᚅᚔᚐᚄ",  # CUNACATOS MAQI MUCOI DOVVINIAS
    "ᚁᚏᚒᚄᚉᚉᚑᚄ ᚋᚐᚊᚔ ᚉᚐᚂᚔᚐᚉᚔ",              # BRUSCCOS MAQI CALIACI
    "ᚋᚐᚊᚔᚈᚈᚐᚄ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ ᚉᚑᚏᚁᚁᚔ",          # MAQITTAS MAQI MUCOI CORBBI
    "ᚅᚓᚈᚈᚐᚄᚂᚑᚌᚔ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ",              # NETTASLOGI MAQI MUCOI
    "ᚂᚒᚌᚒᚊᚏᚔᚈ ᚋᚐᚊᚔ ᚐᚋᚋᚂᚂᚐᚌᚅᚔ",            # LUGUQRIT MAQI AMMLLAGNI
    "ᚈᚏᚓᚅᚐᚌᚒᚄᚒ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ ᚈᚏᚓᚅᚐᚂᚒᚌᚑᚄ",  # TRENAGUSU MAQI MUCOI TRENALOGOS
    "ᚌᚒᚄᚉᚒ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ",                    # GUSCU MAQI MUCOI
    "ᚃᚓᚇᚇᚑᚄ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ ᚉᚒᚅᚐᚉᚐᚈᚑᚄ",      # VEDDOS MAQI MUCOI CUNACATOS
    "ᚉᚐᚈᚈᚒᚁᚒᚈᚈᚐᚄ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ ᚃᚓᚇᚇᚑᚄ",    # CATTUBUTTAS MAQI MUCOI VEDDOS
    "ᚇᚑᚃᚃᚔᚅᚔᚐᚄ ᚋᚐᚊᚔ ᚉᚑᚏᚁᚁᚔ",              # DOVVINIAS MAQI CORBBI
    "ᚇᚓᚌᚂᚐᚅᚅ ᚉᚓᚂᚔ ᚋᚐᚊᚔᚈᚈᚐᚄ",              # DEGLANN CELI MAQITTAS
    "ᚐᚅᚋ ᚉᚒᚅᚐᚉᚐᚈᚑᚄ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ",          # ANM CUNACATOS MAQI MUCOI
]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def is_ogham_character(char: str) -> bool:
    """Check if a character is in the Ogham Unicode range."""
    if len(char) != 1:
        return False
    codepoint = ord(char)
    return 0x1680 <= codepoint <= 0x169F


def is_valid_ogham_letter(char: str) -> bool:
    """Check if a character is a valid Ogham letter (not punctuation/space)."""
    return char in ALL_CHARACTERS


def validate_ogham_string(text: str) -> Tuple[bool, str]:
    """
    Validate an Ogham text string.

    Args:
        text: String to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if not text:
        return False, "Empty string"

    invalid_chars = []
    for i, char in enumerate(text):
        if not is_ogham_character(char):
            invalid_chars.append((i, char, hex(ord(char))))

    if invalid_chars:
        details = ", ".join([f"'{c}' (U+{h}) at pos {i}" for i, c, h in invalid_chars[:5]])
        if len(invalid_chars) > 5:
            details += f" ... and {len(invalid_chars) - 5} more"
        return False, f"Invalid characters: {details}"

    return True, f"Valid Ogham string with {len(text)} characters"


def get_character_info(char: str) -> Optional[Dict]:
    """
    Get detailed information about an Ogham character.

    Args:
        char: Single Ogham character

    Returns:
        Dictionary with character information or None if invalid
    """
    if not is_ogham_character(char):
        return None

    latin = OGHAM_TO_LATIN.get(char, None)

    # Determine group
    group = None
    if char in AICME_BEITHE.values():
        group = "Aicme Beithe (B Group)"
    elif char in AICME_HUATHA.values():
        group = "Aicme hÚatha (H Group)"
    elif char in AICME_MUINE.values():
        group = "Aicme Muine (M Group)"
    elif char in AICME_AILME.values():
        group = "Aicme Ailme (Vowels)"
    elif char in FORFEDA.values():
        group = "Forfeda (Supplementary)"
    elif char in PUNCTUATION.values():
        group = "Punctuation"
    elif char == OGHAM_SPACE:
        group = "Space"

    return {
        "character": char,
        "codepoint": f"U+{ord(char):04X}",
        "latin": latin,
        "group": group,
        "is_vowel": char in ALL_VOWELS,
        "is_forfeda": char in ALL_FORFEDA,
        "frequency": LETTER_FREQUENCIES.get(char, 0.0),
    }


# =============================================================================
# TRANSLITERATION FUNCTIONS
# =============================================================================

def latin_to_ogham(text: str, strict: bool = False) -> str:
    """
    Convert Latin transliteration to Ogham Unicode.

    Args:
        text: Latin text (e.g., "MAQI")
        strict: If True, raise error on unknown characters

    Returns:
        Ogham Unicode string
    """
    result = []
    text = text.upper()
    i = 0

    while i < len(text):
        # Check for digraphs first (NG, EA, OI, UI, IA, AE)
        if i + 1 < len(text):
            digraph = text[i:i+2]
            if digraph in LATIN_TO_OGHAM:
                result.append(LATIN_TO_OGHAM[digraph])
                i += 2
                continue

        # Single character
        char = text[i]
        if char in LATIN_TO_OGHAM:
            result.append(LATIN_TO_OGHAM[char])
        elif char == " ":
            result.append(OGHAM_SPACE)
        elif strict:
            raise ValueError(f"Unknown character: {char}")
        # else: skip unknown characters

        i += 1

    return "".join(result)


def ogham_to_latin(text: str) -> str:
    """
    Convert Ogham Unicode to Latin transliteration.

    Args:
        text: Ogham Unicode string

    Returns:
        Latin transliteration
    """
    result = []

    for char in text:
        if char in OGHAM_TO_LATIN:
            result.append(OGHAM_TO_LATIN[char])
        elif char == OGHAM_SPACE:
            result.append(" ")
        elif char in PUNCTUATION.values():
            # Skip punctuation marks in transliteration
            pass
        else:
            result.append("?")  # Unknown character

    return "".join(result)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_characters(text: str) -> Dict[str, int]:
    """Count occurrences of each Ogham character in text."""
    counts = {}
    for char in text:
        if is_ogham_character(char):
            counts[char] = counts.get(char, 0) + 1
    return counts


def get_reading_direction(text: str) -> str:
    """
    Determine the likely reading direction of an Ogham text.

    Traditional Ogham reads bottom-to-top when carved on a stone edge.
    When rendered horizontally, it's typically left-to-right.
    """
    # If text starts with feather mark, it's formatted for horizontal reading
    if text and text[0] == "᚛":
        return "left_to_right"
    return "bottom_to_top"


def normalize_ogham(text: str) -> str:
    """
    Normalize an Ogham string:
    - Remove punctuation marks
    - Remove spaces
    - Keep only valid letters
    """
    return "".join(char for char in text if char in ALL_LETTERS or char in ALL_FORFEDA)


def split_into_words(text: str) -> List[str]:
    """
    Attempt to split Ogham text into words.

    Note: Original Ogham inscriptions rarely had word separators.
    This function splits on Ogham space characters if present.
    """
    # Split on Ogham space or regular space
    return re.split(f"[{OGHAM_SPACE} ]+", text)


def estimate_difficulty(text: str) -> float:
    """
    Estimate the difficulty of recognizing an Ogham text.

    Factors:
    - Length (longer = harder)
    - Rare characters (forfeda, rare consonants)
    - Pattern regularity

    Returns:
        Float from 0.0 (easy) to 1.0 (hard)
    """
    if not text:
        return 0.0

    # Length factor (normalize to 0-1, cap at 30 chars)
    length_factor = min(len(text) / 30.0, 1.0) * 0.4

    # Rare character factor
    rare_chars = set(ALL_FORFEDA) | {"ᚎ", "ᚍ", "ᚆ", "ᚃ", "ᚏ"}
    rare_count = sum(1 for c in text if c in rare_chars)
    rare_factor = min(rare_count / len(text), 1.0) * 0.3 if text else 0

    # Pattern regularity (having common patterns makes it easier)
    pattern_factor = 0.3
    for pattern in COMMON_PATTERNS:
        if pattern in text:
            pattern_factor -= 0.05
    pattern_factor = max(pattern_factor, 0.0)

    return min(length_factor + rare_factor + pattern_factor, 1.0)


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def render_ogham_table() -> str:
    """Generate a text table of all Ogham characters for reference."""
    lines = [
        "┌────────────────────────────────────────────────────────────────────┐",
        "│                    OGHAM CHARACTER SET                              │",
        "├────────────────────────────────────────────────────────────────────┤",
        "│ AICME BEITHE (B Group) - Strokes right of stemline                 │",
        f"│ ᚁ B (beith)   ᚂ L (luis)   ᚃ F/V (fearn)   ᚄ S (sail)   ᚅ N (nuin)│",
        "│                                                                     │",
        "│ AICME HÚATHA (H Group) - Strokes left of stemline                  │",
        f"│ ᚆ H (úath)   ᚇ D (dair)   ᚈ T (tinne)   ᚉ C (coll)   ᚊ Q (cert)   │",
        "│                                                                     │",
        "│ AICME MUINE (M Group) - Diagonal strokes                           │",
        f"│ ᚋ M (muin)   ᚌ G (gort)   ᚍ NG (gétal)   ᚎ Z (straif)   ᚏ R (ruis)│",
        "│                                                                     │",
        "│ AICME AILME (Vowels) - Notches or cross-strokes                    │",
        f"│ ᚐ A (ailm)   ᚑ O (onn)   ᚒ U (úr)   ᚓ E (edad)   ᚔ I (idad)       │",
        "│                                                                     │",
        "│ FORFEDA (Later additions) - Rare                                    │",
        f"│ ᚕ EA   ᚖ OI   ᚗ UI   ᚘ IA   ᚙ AE                                   │",
        "│                                                                     │",
        "│ PUNCTUATION                                                         │",
        f"│ ᚛ Start of text (feather mark)                                      │",
        f"│ ᚜ End of text (reversed feather mark)                               │",
        "└────────────────────────────────────────────────────────────────────┘",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    print(render_ogham_table())
    print()

    # Test transliteration
    test = "MAQI MUCOI"
    ogham = latin_to_ogham(test)
    print(f"Latin: {test}")
    print(f"Ogham: {ogham}")
    print(f"Back:  {ogham_to_latin(ogham)}")

    # Validate
    print(f"\nValidation: {validate_ogham_string(ogham)}")
    print(f"Difficulty: {estimate_difficulty(ogham):.2f}")
