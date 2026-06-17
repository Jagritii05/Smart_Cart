"""
text_embed_service.py — Dedicated text embedding service for semantic product search.

Uses sentence-transformers/all-MiniLM-L6-v2 (22 MB, 384-dim).
Trained specifically for semantic text similarity — correct for text-to-text retrieval.

CLIP handles image search (image↔text cosine space).
This handles text/voice query search (text↔text cosine space).
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DIM = 384


class TextEmbedService:
    _instance: Optional["TextEmbedService"] = None

    @classmethod
    def get_instance(cls) -> "TextEmbedService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading text embedding model '%s' …", TEXT_EMBED_MODEL)
        self._model = SentenceTransformer(TEXT_EMBED_MODEL)
        logger.info("TextEmbedService ready — dim=%d", TEXT_EMBED_DIM)

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    @property
    def vector_dim(self) -> int:
        return TEXT_EMBED_DIM
