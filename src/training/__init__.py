"""Training infrastructure for Ogham OCR."""

from .trainer import OghamTrainer, TrainingConfig
from .checkpoint import CheckpointManager
from .colab_storage import ColabStorageManager
from .tokenizer_extension import (
    extend_tokenizer_with_ogham,
    resize_model_embeddings,
    setup_ogham_model_and_tokenizer,
    setup_transliteration_model,
    verify_ogham_tokenization,
)

__all__ = [
    "OghamTrainer",
    "TrainingConfig",
    "CheckpointManager",
    "ColabStorageManager",
    "extend_tokenizer_with_ogham",
    "resize_model_embeddings",
    "setup_ogham_model_and_tokenizer",
    "setup_transliteration_model",
    "verify_ogham_tokenization",
]
