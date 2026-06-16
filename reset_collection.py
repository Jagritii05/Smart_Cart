"""
reset_collection.py — Drop and recreate the Qdrant collection.

Run this ONCE after adding a new named vector (e.g., audio_waveform) to
qdrant_setup.py.  It deletes all existing points and vector indexes so
that ingest.py can rebuild the collection with the new schema.

Usage:
    python reset_collection.py [--confirm]

WARNING: This is destructive — all ingested product vectors will be lost.
         You MUST re-run ingest.py afterwards.
"""

import argparse
import logging
import sys

from qdrant_client import QdrantClient

from config import COLLECTION_NAME, QDRANT_STORAGE_PATH
from qdrant_setup import get_qdrant_client, verify_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("reset_collection")


def reset(client: QdrantClient) -> None:
    """
    Delete the existing collection (if present) and recreate it with the
    current schema defined in qdrant_setup.create_collection.

    Args:
        client: An initialised QdrantClient instance.
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        logger.info("Deleting existing collection '%s' …", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)
        logger.info("Collection '%s' deleted.", COLLECTION_NAME)
    else:
        logger.info("Collection '%s' does not exist — nothing to delete.", COLLECTION_NAME)

    # Re-create with the new schema (resolves vector_dim from EmbedService)
    from embed_service import EmbedService          # noqa: PLC0415
    from qdrant_setup import create_collection      # noqa: PLC0415

    logger.info("Loading embedding model to resolve vector dimension …")
    embed_svc = EmbedService.get_instance()
    vector_dim = embed_svc.vector_dim
    logger.info("Vector dim resolved: %d", vector_dim)

    create_collection(client, vector_dim=vector_dim)
    logger.info(
        "Collection '%s' recreated — ready for ingest.py.",
        COLLECTION_NAME,
    )
    verify_collection(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drop and recreate the Qdrant Smart Cart collection."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()

    if not args.confirm:
        print(
            f"\n⚠️  This will DELETE all data in collection '{COLLECTION_NAME}'.\n"
            "   You must re-run  python ingest.py  afterwards to rebuild the index.\n"
        )
        answer = input("Type 'yes' to confirm: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            sys.exit(0)

    client = get_qdrant_client()
    reset(client)
    print(
        f"\n✅  Collection reset complete.\n"
        f"   Next step: python ingest.py\n"
    )
