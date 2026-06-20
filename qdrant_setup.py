"""
qdrant_setup.py — Creates and configures the Qdrant Edge collection
for Smart Cart. All named vectors are persisted on disk to preserve
RAM for live neural model execution.
"""

import logging
import os
from typing import Optional


from qdrant_edge import (
    EdgeShard,
    EdgeConfig,
    EdgeVectorParams,
    Distance,
    PayloadSchemaType,
    UpdateOperation,
)

from config import (
    QDRANT_STORAGE_PATH,
    VECTOR_BARCODE_VISUAL,
    VECTOR_VOICE_QUERY,
    VECTOR_NUTRITION_PDF,
    VECTOR_AUDIO_WAVEFORM,
    VECTOR_DIM,
    TEXT_EMBED_DIM,
    WHISPER_EMBED_DIM,
)

logger = logging.getLogger(__name__)


def get_qdrant_client() -> EdgeShard:
    """
    Return a file-backed EdgeShard using the configured storage path.
    If the shard configuration does not exist, initialize it with the 4 named vectors.
    """
    os.makedirs(QDRANT_STORAGE_PATH, exist_ok=True)
    config_file = os.path.join(QDRANT_STORAGE_PATH, "edge_config.json")

    if os.path.exists(config_file):
        logger.info("Loading existing Qdrant Edge shard from: %s", QDRANT_STORAGE_PATH)
        return EdgeShard.load(QDRANT_STORAGE_PATH)
    else:
        logger.info("Initializing new Qdrant Edge shard at: %s", QDRANT_STORAGE_PATH)
        named_vectors = {
            # CLIP image embeddings — 512-dim
            VECTOR_BARCODE_VISUAL: EdgeVectorParams(
                size=VECTOR_DIM,
                distance=Distance.Cosine,
                on_disk=True,
            ),
            # sentence-transformers MiniLM text embeddings — 384-dim
            VECTOR_VOICE_QUERY: EdgeVectorParams(
                size=TEXT_EMBED_DIM,
                distance=Distance.Cosine,
                on_disk=True,
            ),
            VECTOR_NUTRITION_PDF: EdgeVectorParams(
                size=TEXT_EMBED_DIM,
                distance=Distance.Cosine,
                on_disk=True,
            ),
            # Whisper acoustic embedding — 512-dim
            VECTOR_AUDIO_WAVEFORM: EdgeVectorParams(
                size=WHISPER_EMBED_DIM,
                distance=Distance.Cosine,
                on_disk=True,
            ),
        }
        config = EdgeConfig(vectors=named_vectors, on_disk_payload=True)
        return EdgeShard.create(QDRANT_STORAGE_PATH, config)


def create_collection(client: EdgeShard, vector_dim: Optional[int] = None) -> None:
    """
    Create the payload indexes for the collection/shard.
    The vectors schema is already configured during EdgeShard creation.

    Args:
        client:     An initialised EdgeShard instance.
        vector_dim: Unused, kept for interface compatibility.
    """
    _create_payload_indexes(client)


def _create_payload_indexes(client: EdgeShard) -> None:
    """
    Create payload field indexes to enable fast filtered search.
    """
    index_map = {
        "product_id":   PayloadSchemaType.Keyword,
        "name":         PayloadSchemaType.Text,
        "brand":        PayloadSchemaType.Keyword,
        "aisle_number": PayloadSchemaType.Integer,
        "price":        PayloadSchemaType.Float,
        "tags":         PayloadSchemaType.Keyword,
        "stock_status": PayloadSchemaType.Integer,
        "store_id":     PayloadSchemaType.Integer,
        "description":  PayloadSchemaType.Text,
    }

    for field_name, schema_type in index_map.items():
        try:
            client.update(UpdateOperation.create_field_index(field_name, schema_type))
            logger.debug("Payload index created: %s (%s)", field_name, schema_type)
        except Exception as exc:
            logger.error("Failed to create index for field '%s': %s", field_name, exc)

    logger.info("All payload indexes ensured.")


def verify_collection(client: EdgeShard) -> bool:
    """
    Return True if the EdgeShard is accessible.

    Args:
        client: An initialised EdgeShard instance.

    Returns:
        True if the shard responded to info queries.
    """
    try:
        info = client.info()
        logger.info(
            "Qdrant Edge verified — %d points in shard.",
            info.points_count or 0,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Shard verification failed: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _client = get_qdrant_client()
    create_collection(_client)
    verify_collection(_client)
    _client.close()
    print("Qdrant Edge setup complete.")
