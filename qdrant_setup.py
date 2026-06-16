"""
qdrant_setup.py — Creates and configures the Qdrant Edge collection
for Smart Cart. All named vectors are persisted on disk to preserve
RAM for live neural model execution.
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
    HnswConfigDiff,
    OptimizersConfigDiff,
)

from config import (
    COLLECTION_NAME,
    QDRANT_STORAGE_PATH,
    VECTOR_BARCODE_VISUAL,
    VECTOR_VOICE_QUERY,
    VECTOR_NUTRITION_PDF,
    VECTOR_AUDIO_WAVEFORM,
    VECTOR_DIM,
    WHISPER_EMBED_DIM,
)

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """
    Return a file-backed QdrantClient using the configured storage path.
    No server process is required — pure edge mode.
    """
    client = QdrantClient(path=QDRANT_STORAGE_PATH)
    logger.info("QdrantClient initialised at: %s", QDRANT_STORAGE_PATH)
    return client


def create_collection(client: QdrantClient, vector_dim: Optional[int] = None) -> None:
    """
    Create the 'smart_cart_products' collection with four named vectors
    (barcode_visual, voice_query, nutrition_pdf, audio_waveform), all stored on disk.

    The first three vectors share the Qwen embedding dimension (``vector_dim``).
    The ``audio_waveform`` vector uses the fixed Whisper encoder dimension
    (``WHISPER_EMBED_DIM`` = 512) regardless of what the Qwen model resolves to.

    If the collection already exists this function is a no-op so that
    re-running the ingest script never destroys existing data.

    Args:
        client:     An initialised QdrantClient instance.
        vector_dim: Override the dimension from config (used when the
                    embed model resolves its own output size at runtime).
    """
    dim = vector_dim or VECTOR_DIM

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        logger.info("Collection '%s' already exists — skipping creation.", COLLECTION_NAME)
        return

    # All Qwen-based named vectors share the same cosine distance metric
    # and are persisted on disk (on_disk=True).
    # audio_waveform uses a separate fixed dimension (Whisper encoder output).
    named_vectors = {
        VECTOR_BARCODE_VISUAL: VectorParams(
            size=dim,
            distance=Distance.COSINE,
            on_disk=True,
            hnsw_config=HnswConfigDiff(on_disk=True),
        ),
        VECTOR_VOICE_QUERY: VectorParams(
            size=dim,
            distance=Distance.COSINE,
            on_disk=True,
            hnsw_config=HnswConfigDiff(on_disk=True),
        ),
        VECTOR_NUTRITION_PDF: VectorParams(
            size=dim,
            distance=Distance.COSINE,
            on_disk=True,
            hnsw_config=HnswConfigDiff(on_disk=True),
        ),
        # Native acoustic embedding vector — Whisper encoder output space (512-dim).
        # Stored separately from the Qwen vectors so both spaces remain clean.
        VECTOR_AUDIO_WAVEFORM: VectorParams(
            size=WHISPER_EMBED_DIM,
            distance=Distance.COSINE,
            on_disk=True,
            hnsw_config=HnswConfigDiff(on_disk=True),
        ),
    }

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=named_vectors,
        optimizers_config=OptimizersConfigDiff(memmap_threshold=10_000),
    )
    logger.info(
        "Collection '%s' created — Qwen dim=%d, Whisper audio dim=%d, on_disk vectors.",
        COLLECTION_NAME,
        dim,
        WHISPER_EMBED_DIM,
    )

    _create_payload_indexes(client)


def _create_payload_indexes(client: QdrantClient) -> None:
    """
    Create payload field indexes to enable fast filtered search.
    All fields in the Section 3 payload schema are indexed.
    """
    index_map = {
        "product_id":   PayloadSchemaType.KEYWORD,
        "name":         PayloadSchemaType.TEXT,
        "brand":        PayloadSchemaType.KEYWORD,
        "aisle_number": PayloadSchemaType.INTEGER,
        "price":        PayloadSchemaType.FLOAT,
        "tags":         PayloadSchemaType.KEYWORD,
        "stock_status": PayloadSchemaType.BOOL,
        "store_id":     PayloadSchemaType.INTEGER,
        "description":  PayloadSchemaType.TEXT,
    }

    for field_name, schema_type in index_map.items():
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=schema_type,
        )
        logger.debug("Payload index created: %s (%s)", field_name, schema_type)

    logger.info("All payload indexes created for '%s'.", COLLECTION_NAME)


def verify_collection(client: QdrantClient) -> bool:
    """
    Return True if the collection exists and is accessible.

    Args:
        client: An initialised QdrantClient instance.

    Returns:
        True if the collection exists and responds to info queries.
    """
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(
            "Collection '%s' verified — %d points in collection.",
            COLLECTION_NAME,
            info.points_count or 0,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Collection verification failed: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _client = get_qdrant_client()
    create_collection(_client)
    verify_collection(_client)
    print("Qdrant Edge setup complete.")
