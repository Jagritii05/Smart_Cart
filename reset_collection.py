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
import os
import shutil
import sys

from qdrant_edge import EdgeShard
from config import QDRANT_STORAGE_PATH
from qdrant_setup import get_qdrant_client, verify_collection, create_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("reset_collection")


def reset(client: EdgeShard) -> None:
    """
    Delete the local Qdrant Edge shard storage directory and recreate it
    with the current schema.

    Args:
        client: An initialised EdgeShard instance.
    """
    logger.info("Closing active Qdrant Edge shard...")
    client.close()

    if os.path.exists(QDRANT_STORAGE_PATH):
        logger.info("Deleting local Qdrant Edge storage directory: %s", QDRANT_STORAGE_PATH)
        try:
            shutil.rmtree(QDRANT_STORAGE_PATH)
            logger.info("Local storage deleted successfully.")
        except Exception as exc:
            logger.error("Failed to delete local storage directory: %s", exc)
            sys.exit(1)
    else:
        logger.info("Storage directory '%s' does not exist.", QDRANT_STORAGE_PATH)

    logger.info("Reinitializing Qdrant Edge shard...")
    new_client = get_qdrant_client()
    create_collection(new_client)
    logger.info("Collection recreated — ready for ingest.py.")
    verify_collection(new_client)
    new_client.close()


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
            f"\n⚠️  This will DELETE all local vector data in '{QDRANT_STORAGE_PATH}'.\n"
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
