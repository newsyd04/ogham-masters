"""
Tokenizer extension for Ogham Unicode characters.

Extends the TrOCR (RoBERTa-based) tokenizer vocabulary to include
all Ogham Unicode characters (U+1680–U+169F) as first-class tokens.

★ Insight ─────────────────────────────────────
Why extend rather than byte-fallback?
1. BPE byte-fallback encodes unknown Unicode as multiple tokens
   (e.g., ᚋ → 3 byte tokens), making the decoder's job much harder
2. Adding explicit tokens means 1 Ogham char = 1 token, matching
   the natural structure of the script
3. The model only needs to learn ~29 new embeddings (20 consonants,
   5 vowels, 5 forfeda, space, 2 punctuation marks minus overlaps)
─────────────────────────────────────────────────
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

log = logging.getLogger(__name__)

# All Ogham characters that need dedicated tokens
# Ordered by Unicode codepoint for consistency
#
# NOTE: U+1680 (Ogham Space Mark) is excluded from this list because it is
# Unicode category Zs (Space Separator). RoBERTa's pre-tokenizer normalizes
# all Zs-category characters during whitespace splitting *before* vocabulary
# lookup, so U+1680 can never be matched as a token — even if added to the
# vocab. Labels should normalize U+1680 → ASCII space (see normalize_ogham_labels).
OGHAM_TOKENS: List[str] = [
    "\u1681",  # ᚁ B (beith)
    "\u1682",  # ᚂ L (luis)
    "\u1683",  # ᚃ F/V (fearn)
    "\u1684",  # ᚄ S (sail)
    "\u1685",  # ᚅ N (nuin)
    "\u1686",  # ᚆ H (úath)
    "\u1687",  # ᚇ D (dair)
    "\u1688",  # ᚈ T (tinne)
    "\u1689",  # ᚉ C (coll)
    "\u168A",  # ᚊ Q (cert)
    "\u168B",  # ᚋ M (muin)
    "\u168C",  # ᚌ G (gort)
    "\u168D",  # ᚍ NG (gétal)
    "\u168E",  # ᚎ Z (straif)
    "\u168F",  # ᚏ R (ruis)
    "\u1690",  # ᚐ A (ailm)
    "\u1691",  # ᚑ O (onn)
    "\u1692",  # ᚒ U (úr)
    "\u1693",  # ᚓ E (edad)
    "\u1694",  # ᚔ I (idad)
    "\u1695",  # ᚕ EA (ébad)
    "\u1696",  # ᚖ OI (óir)
    "\u1697",  # ᚗ UI (uilleann)
    "\u1698",  # ᚘ IA (ifín)
    "\u1699",  # ᚙ AE (emancholl)
    "\u169A",  # ᚚ (peith) - rare but in Unicode block
    "\u169B",  # ᚛ Start of text (feather mark)
    "\u169C",  # ᚜ End of text (reversed feather mark)
]

# Epigraphic annotation tokens used in scholarly transcriptions
# These appear in real annotations (e.g., "[ᚂᚔ]" for uncertain readings)
ANNOTATION_TOKENS: List[str] = [
    "[",   # Start of uncertain/damaged section
    "]",   # End of uncertain/damaged section
    "…",   # Lacuna (missing/illegible section)
]


def _load_model_no_meta(model_cls: Any, model_name: str) -> Any:
    """
    Load a HuggingFace model ensuring NO tensors remain on the meta device.

    Tries multiple strategies in order of preference:
    1. from_pretrained with low_cpu_mem_usage=False (avoids accelerate meta init)
    2. If meta tensors still found, re-load the state dict from hub onto CPU
    3. Last resort: initialize remaining meta tensors with random values

    Returns the model with all tensors on CPU.
    """
    import torch

    # Strategy 1: load without lazy/meta device init
    model = model_cls.from_pretrained(model_name, low_cpu_mem_usage=False)

    # Check for meta tensors
    meta_names = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    meta_bufs = [n for n, b in model.named_buffers() if b.device.type == "meta"]

    if not meta_names and not meta_bufs:
        log.info("Model loaded cleanly — no meta tensors")
        return model

    log.warning(
        f"Found {len(meta_names)} meta params + {len(meta_bufs)} meta buffers "
        f"after from_pretrained. Attempting state_dict reload..."
    )

    # Strategy 2: re-download weights and load them directly on CPU
    try:
        from huggingface_hub import hf_hub_download
        import safetensors.torch
        import os

        # Try safetensors first (faster, memory-mapped), then pytorch bin
        for filename in ["model.safetensors", "pytorch_model.bin"]:
            try:
                path = hf_hub_download(model_name, filename)
                if filename.endswith(".safetensors"):
                    real_sd = safetensors.torch.load_file(path, device="cpu")
                else:
                    real_sd = torch.load(path, map_location="cpu", weights_only=True)

                # Load into model (strict=False in case of key mismatches)
                missing, unexpected = model.load_state_dict(real_sd, strict=False)
                if missing:
                    log.warning(f"Keys missing after state_dict reload: {missing}")

                # Re-check
                still_meta = [n for n, p in model.named_parameters()
                              if p.device.type == "meta"]
                if not still_meta:
                    log.info("State dict reload fixed all meta tensors")
                    return model
                else:
                    log.warning(f"Still meta after reload: {still_meta}")
                break
            except Exception as e:
                log.debug(f"Could not load {filename}: {e}")
                continue
    except ImportError:
        log.warning("huggingface_hub not available for state_dict reload")

    # Strategy 3: last resort — materialize with random init
    materialized = []
    for name, param in list(model.named_parameters()):
        if param.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            real = torch.nn.Parameter(
                torch.empty(param.shape, dtype=param.dtype, device="cpu"),
                requires_grad=param.requires_grad,
            )
            torch.nn.init.normal_(real, std=0.02)
            parent._parameters[parts[-1]] = real
            materialized.append(name)

    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            real = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
            parent._buffers[parts[-1]] = real
            materialized.append(name)

    if materialized:
        log.warning(
            f"Last-resort materialized {len(materialized)} meta tensors "
            f"(random init — pretrained weights lost for these): {materialized}"
        )

    return model


def normalize_ogham_labels(text: str) -> str:
    """
    Normalize Ogham text for tokenizer compatibility.

    Replaces U+1680 (Ogham Space Mark) with ASCII space, since RoBERTa's
    pre-tokenizer strips U+1680 as whitespace before token matching.
    ASCII space is already handled natively by the tokenizer.
    """
    return text.replace("\u1680", " ")


def extend_tokenizer_with_ogham(
    tokenizer: Any,
    include_annotations: bool = True,
) -> Tuple[Any, int]:
    """
    Add Ogham Unicode characters to the tokenizer vocabulary.

    Each Ogham character is added as a new token so it gets its own
    embedding vector. This avoids the BPE byte-fallback problem where
    unknown Unicode chars are split into multiple byte-level tokens.

    Args:
        tokenizer: HuggingFace tokenizer (RoBERTa-based from TrOCR)
        include_annotations: Also add epigraphic annotation tokens

    Returns:
        Tuple of (modified tokenizer, number of tokens added)
    """
    tokens_to_add = []

    # Check which Ogham tokens are not already in the vocabulary
    for token in OGHAM_TOKENS:
        # Test if the tokenizer already handles this as a single token
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != token:
            tokens_to_add.append(token)

    # Add annotation tokens if requested
    if include_annotations:
        for token in ANNOTATION_TOKENS:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            # Only add if not already a single clean token
            if len(encoded) != 1 or tokenizer.decode(encoded).strip() != token:
                tokens_to_add.append(token)

    if not tokens_to_add:
        log.info("All Ogham tokens already present in tokenizer vocabulary")
        return tokenizer, 0

    # Record vocab size before adding, so we know which IDs are new
    vocab_before = len(tokenizer)

    # Add new tokens to the tokenizer
    # Using add_tokens (not add_special_tokens) so they behave like regular tokens
    # during generation — special tokens can interfere with beam search
    num_added = tokenizer.add_tokens(tokens_to_add)

    # Build set of newly added token IDs for embedding initialization
    new_token_ids = set()
    for token in tokens_to_add:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1 and ids[0] >= vocab_before:
            new_token_ids.add(ids[0])
    tokenizer._ogham_new_token_ids = new_token_ids

    log.info(
        f"Extended tokenizer with {num_added} Ogham tokens. "
        f"New vocab size: {len(tokenizer)}"
    )

    return tokenizer, num_added


def resize_model_embeddings(
    model: Any,
    tokenizer: Any,
    init_strategy: str = "latin",
) -> None:
    """
    Resize model embedding matrices and initialize new Ogham token embeddings.

    Supports three initialization strategies:
      - "random": HuggingFace default (small random values)
      - "zero":   Initialize new embeddings to zero vectors
      - "latin":  Copy embeddings from Latin equivalents (e.g., ᚋ gets M's embedding)

    For digraphs (NG, EA, OI, etc.), "latin" averages the two component
    letter embeddings.

    Args:
        model: TrOCR model (VisionEncoderDecoderModel)
        tokenizer: Extended tokenizer
        init_strategy: "random", "zero", or "latin"
    """
    import torch

    old_size = model.decoder.config.vocab_size
    new_size = len(tokenizer)

    if new_size == old_size:
        log.info("Model embeddings already match tokenizer size")
        return

    # Resize the decoder's embeddings (encoder is ViT — no text tokens)
    # This initialises new rows with small random values by default
    model.decoder.resize_token_embeddings(new_size)

    # Update config to reflect new vocab size
    model.decoder.config.vocab_size = new_size
    model.config.decoder.vocab_size = new_size

    if init_strategy == "random":
        log.info(
            f"Resized embeddings: {old_size} → {new_size} "
            f"(+{new_size - old_size} tokens, random init)"
        )
        return

    # --- Apply non-random initialization to new token embeddings ---

    # Ogham → Latin mapping for seeding embeddings
    # U+1680 excluded — handled via label normalization (see normalize_ogham_labels)
    OGHAM_LATIN_MAP = {
        "\u1681": "B",    # ᚁ
        "\u1682": "L",    # ᚂ
        "\u1683": "F",    # ᚃ
        "\u1684": "S",    # ᚄ
        "\u1685": "N",    # ᚅ
        "\u1686": "H",    # ᚆ
        "\u1687": "D",    # ᚇ
        "\u1688": "T",    # ᚈ
        "\u1689": "C",    # ᚉ
        "\u168A": "Q",    # ᚊ
        "\u168B": "M",    # ᚋ
        "\u168C": "G",    # ᚌ
        "\u168D": "NG",   # ᚍ (digraph → average N + G)
        "\u168E": "Z",    # ᚎ
        "\u168F": "R",    # ᚏ
        "\u1690": "A",    # ᚐ
        "\u1691": "O",    # ᚑ
        "\u1692": "U",    # ᚒ
        "\u1693": "E",    # ᚓ
        "\u1694": "I",    # ᚔ
        "\u1695": "EA",   # ᚕ (digraph → average E + A)
        "\u1696": "OI",   # ᚖ (digraph → average O + I)
        "\u1697": "UI",   # ᚗ (digraph → average U + I)
        "\u1698": "IA",   # ᚘ (digraph → average I + A)
        "\u1699": "AE",   # ᚙ (digraph → average A + E)
        "\u169A": "P",    # ᚚ peith → P
        "\u169B": "(",    # ᚛ start mark → open paren
        "\u169C": ")",    # ᚜ end mark → close paren
    }

    inp_embeddings = model.decoder.get_input_embeddings()
    out_embeddings = model.decoder.get_output_embeddings()

    # Get the set of newly added token IDs (tracked during extend_tokenizer_with_ogham)
    new_token_ids = getattr(tokenizer, "_ogham_new_token_ids", set())

    initialized = 0

    with torch.no_grad():
        for ogham_char, latin_equiv in OGHAM_LATIN_MAP.items():
            # Get the token ID for this Ogham character
            ogham_ids = tokenizer.encode(ogham_char, add_special_tokens=False)
            if len(ogham_ids) != 1:
                continue  # Skip if not a single token
            ogham_id = ogham_ids[0]

            # Only modify tokens that were newly added
            # (tokens that existed before extension already have trained embeddings)
            if new_token_ids and ogham_id not in new_token_ids:
                continue

            if init_strategy == "zero":
                inp_embeddings.weight[ogham_id] = 0.0
                out_embeddings.weight[ogham_id] = 0.0
                initialized += 1

            elif init_strategy == "latin":
                # Get embeddings for each Latin character and average them
                latin_embs_inp = []
                latin_embs_out = []

                for ch in latin_equiv:
                    ch_ids = tokenizer.encode(ch, add_special_tokens=False)
                    if len(ch_ids) >= 1:
                        # Use the first token if multi-token
                        latin_embs_inp.append(inp_embeddings.weight[ch_ids[0]].clone())
                        latin_embs_out.append(out_embeddings.weight[ch_ids[0]].clone())

                if latin_embs_inp:
                    # Average for digraphs, direct copy for single chars
                    inp_embeddings.weight[ogham_id] = torch.stack(latin_embs_inp).mean(dim=0)
                    out_embeddings.weight[ogham_id] = torch.stack(latin_embs_out).mean(dim=0)
                    initialized += 1

    log.info(
        f"Resized embeddings: {old_size} → {new_size} "
        f"(+{new_size - old_size} tokens, {init_strategy} init, "
        f"{initialized} Ogham tokens initialized)"
    )


def verify_ogham_tokenization(tokenizer: Any) -> Dict[str, Any]:
    """
    Verify that each Ogham character tokenizes to exactly one token.

    Returns a diagnostic report useful for debugging tokenizer issues.

    Args:
        tokenizer: Extended tokenizer

    Returns:
        Report dictionary with per-character tokenization details
    """
    report = {
        "all_single_token": True,
        "characters": [],
        "failures": [],
    }

    for token in OGHAM_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        is_single = len(encoded) == 1
        roundtrip_ok = decoded.strip() == token

        char_info = {
            "character": token,
            "codepoint": f"U+{ord(token):04X}",
            "token_ids": encoded,
            "num_tokens": len(encoded),
            "decoded": decoded,
            "single_token": is_single,
            "roundtrip": roundtrip_ok,
        }
        report["characters"].append(char_info)

        if not is_single or not roundtrip_ok:
            report["all_single_token"] = False
            report["failures"].append(char_info)

    report["total_characters"] = len(OGHAM_TOKENS)
    report["successful"] = len(OGHAM_TOKENS) - len(report["failures"])
    report["failed"] = len(report["failures"])

    return report


def setup_ogham_model_and_tokenizer(
    model_name: str = "microsoft/trocr-base-stage1",
    include_annotations: bool = True,
    init_strategy: str = "latin",
) -> Tuple[Any, Any, Any]:
    """
    Complete setup: load model, extend tokenizer, resize embeddings.

    This is the main entry point for preparing a TrOCR model for
    Ogham OCR training with native Unicode output.

    Args:
        model_name: HuggingFace model identifier
        include_annotations: Include epigraphic annotation tokens
        init_strategy: Embedding init for new tokens: "random", "zero", or "latin"

    Returns:
        Tuple of (model, processor, tokenizer)
    """
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    log.info(f"Loading model: {model_name}")

    import torch
    model = _load_model_no_meta(VisionEncoderDecoderModel, model_name)
    processor = TrOCRProcessor.from_pretrained(model_name)

    # Extend tokenizer with Ogham characters
    tokenizer = processor.tokenizer
    tokenizer, num_added = extend_tokenizer_with_ogham(
        tokenizer, include_annotations=include_annotations
    )

    # Resize model embeddings to match
    if num_added > 0:
        resize_model_embeddings(model, tokenizer, init_strategy=init_strategy)

    # Verify tokenization is correct
    report = verify_ogham_tokenization(tokenizer)
    if not report["all_single_token"]:
        log.warning(
            f"Tokenization verification failed for {report['failed']} characters. "
            f"Details: {report['failures']}"
        )
    else:
        log.info(
            f"Tokenization verified: all {report['total_characters']} "
            f"Ogham characters map to single tokens"
        )

    # Set decoder config for generation
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id

    # Set reasonable generation defaults for Ogham via generation_config
    # (newer transformers requires this instead of model.config)
    model.generation_config.max_length = 64
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 0  # Ogham names can repeat chars
    model.generation_config.num_beams = 4
    model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.sep_token_id

    return model, processor, tokenizer


def setup_transliteration_model(
    model_name: str = "microsoft/trocr-base-stage1",
) -> Tuple[Any, Any, Any]:
    """
    Setup TrOCR for Latin transliteration output (no tokenizer extension needed).

    In this mode, the model outputs Latin characters (e.g., "MAQI MUCOI")
    instead of Ogham Unicode. The existing RoBERTa tokenizer handles Latin
    characters natively.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (model, processor, tokenizer)
    """
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    log.info(f"Loading model for transliteration: {model_name}")

    import torch
    model = _load_model_no_meta(VisionEncoderDecoderModel, model_name)
    processor = TrOCRProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # Set decoder config for generation via generation_config
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.generation_config.max_length = 64
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.num_beams = 4
    model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.sep_token_id

    return model, processor, tokenizer
