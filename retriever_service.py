"""
retriever_service.py — Vector search layer over Qdrant Edge.

Provides three search methods (image, voice, text) that apply the
appropriate named-vector and payload filters before returning ranked
product dicts ready for the agentic RAG loop.
"""

import logging
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from config import (
    COLLECTION_NAME,
    VECTOR_BARCODE_VISUAL,
    VECTOR_VOICE_QUERY,
    VECTOR_AUDIO_WAVEFORM,
    DEFAULT_STORE_ID,
    DEFAULT_TOP_K,
)
from embed_service import EmbedService
from text_embed_service import TextEmbedService

logger = logging.getLogger(__name__)


class RetrieverService:
    """
    Vector similarity retriever backed by Qdrant Edge (file mode).

    Each search method selects the appropriate named vector and applies
    payload filters for store isolation, stock status, and optional
    aisle / tag constraints.
    """

    def __init__(self, client: QdrantClient, embed_service: EmbedService) -> None:
        self._client = client
        self._embed = embed_service
        self._text_embed = TextEmbedService.get_instance()

    # ── Public API ────────────────────────────────────────────────────────────

    def search_by_image(
        self,
        frame: np.ndarray,
        aisle_number: Optional[int] = None,
        store_id: int = DEFAULT_STORE_ID,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Embed a BGR camera frame and query the barcode_visual vector index.

        Args:
            frame:        H×W×3 uint8 numpy array in BGR (OpenCV) format.
            aisle_number: Optional aisle filter — restrict results to this aisle.
            store_id:     Multi-tenant store identifier.
            top_k:        Maximum number of results to return.

        Returns:
            List of product dicts sorted by descending similarity score.
        """
        vector = self._embed.embed(frame, modality="image")
        conditions = self._base_conditions(store_id)
        if aisle_number is not None:
            conditions.append(
                FieldCondition(key="aisle_number", match=MatchValue(value=aisle_number))
            )
        # Stock filter for visual queries (show only available items)
        conditions.append(
            FieldCondition(key="stock_status", match=MatchValue(value=True))
        )
        return self._query(VECTOR_BARCODE_VISUAL, vector, conditions, top_k)

    def search_by_voice(
        self,
        audio_waveform: np.ndarray,
        tags: Optional[list[str]] = None,
        store_id: int = DEFAULT_STORE_ID,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Embed a raw 16 kHz mono waveform via Whisper encoder and query the
        ``audio_waveform`` vector index (native acoustic embedding space).

        This is a transcription-free search path: the raw audio waveform is
        fed directly into the Whisper-base encoder, whose hidden states are
        mean-pooled to a 512-dim vector.  Because the catalog was also
        indexed with Whisper encoder embeddings of TTS-synthesised speech,
        both query and document vectors live in the same acoustic space and
        cosine similarity reflects genuine semantic audio closeness.

        Args:
            audio_waveform: 1-D float32 numpy array at 16 000 Hz.
            tags:           Optional list of dietary tags for filtering
                            (e.g. ["vegan", "gluten-free"]).
            store_id:       Multi-tenant store identifier.
            top_k:          Maximum number of results to return.

        Returns:
            List of in-stock product dicts sorted by descending similarity score.
        """
        # embed() routes modality='audio' → AudioEmbedService.embed_waveform()
        vector = self._embed.embed(audio_waveform, modality="audio")
        conditions = self._base_conditions(store_id)
        conditions.append(
            FieldCondition(key="stock_status", match=MatchValue(value=True))
        )
        if tags:
            conditions.append(
                FieldCondition(key="tags", match=MatchAny(any=tags))
            )
        # Query the native acoustic vector index (512-dim Whisper space)
        return self._query(VECTOR_AUDIO_WAVEFORM, vector, conditions, top_k)

    def search_by_text(
        self,
        query_str: str,
        tags: Optional[list[str]] = None,
        store_id: int = DEFAULT_STORE_ID,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Embed a text query string and search the voice_query vector index.

        Identical filter logic to :meth:`search_by_voice` — both use the
        voice_query named vector because spoken and typed queries occupy
        the same region of the embedding space.

        Args:
            query_str: Natural-language text query from the customer.
            tags:      Optional dietary tag filter list.
            store_id:  Multi-tenant store identifier.
            top_k:     Maximum number of results to return.

        Returns:
            List of in-stock product dicts sorted by descending similarity score.
        """
        vector = self._text_embed.embed(query_str)
        conditions = self._base_conditions(store_id)
        conditions.append(
            FieldCondition(key="stock_status", match=MatchValue(value=True))
        )
        if tags:
            conditions.append(
                FieldCondition(key="tags", match=MatchAny(any=tags))
            )
        return self._query(VECTOR_VOICE_QUERY, vector, conditions, top_k)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _base_conditions(self, store_id: int) -> list:
        """Return the store-isolation filter condition list."""
        return [FieldCondition(key="store_id", match=MatchValue(value=store_id))]

    def _query(
        self,
        vector_name: str,
        vector: list[float],
        conditions: list,
        top_k: int,
    ) -> list[dict]:
        """
        Execute a named-vector search with the given filter conditions.

        Args:
            vector_name: One of the three named vector keys.
            vector:      The query embedding as a list of floats.
            conditions:  Qdrant FieldCondition list for payload filtering.
            top_k:       Number of results to fetch.

        Returns:
            Parsed list of product dicts with a ``score`` field appended.
        """
        query_filter = Filter(must=conditions) if conditions else None

        try:
            result = self._client.query_points(
                collection_name=COLLECTION_NAME,
                using=vector_name,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = result.points
        except Exception as exc:  # noqa: BLE001
            logger.error("Qdrant search failed on vector '%s': %s", vector_name, exc)
            return []

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "product_id":   payload.get("product_id", ""),
                    "name":         payload.get("name", "Unknown"),
                    "brand":        payload.get("brand", "Unknown"),
                    "aisle_number": payload.get("aisle_number", 0),
                    "price":        payload.get("price", 0.0),
                    "tags":         payload.get("tags", []),
                    "description":  payload.get("description", ""),
                    "stock_status": payload.get("stock_status", False),
                    "score":        round(float(hit.score), 4),
                }
            )

        logger.debug(
            "Vector search '%s' returned %d result(s).", vector_name, len(results)
        )
        return results
