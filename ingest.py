"""
ingest.py — Product catalog ingestion script for Smart Cart.

Reads data/products.json, embeds each product via EmbedService across
all four named vectors, and batch-upserts everything into Qdrant Edge.

Named vectors per product:
  barcode_visual  — Qwen vision encoder of the product image
  voice_query     — Qwen text encoder of "<name>. <description>"
  nutrition_pdf   — Qwen text encoder of the nutrition PDF (or voice_query fallback)
  audio_waveform  — Whisper encoder of TTS-synthesised speech (native audio embedding)

Usage:
    python ingest.py [--products PATH] [--store-id INT]
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from qdrant_client.models import PointStruct

from config import (
    DATA_DIR,
    IMAGES_DIR,
    PDFS_DIR,
    COLLECTION_NAME,
    VECTOR_BARCODE_VISUAL,
    VECTOR_VOICE_QUERY,
    VECTOR_NUTRITION_PDF,
    VECTOR_AUDIO_WAVEFORM,
    DEFAULT_STORE_ID,
    AUDIO_SAMPLE_RATE,
)
from audio_embed_service import AudioEmbedService
from embed_service import EmbedService
from qdrant_setup import get_qdrant_client, create_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("ingest")


def load_product_image(product_id: str) -> Optional[np.ndarray]:
    """
    Load the product image at images/{product_id}.jpg as a BGR numpy array.

    Args:
        product_id: UUID string of the product.

    Returns:
        BGR numpy array, or None if the file does not exist or cannot be read.
    """
    try:
        import cv2  # noqa: PLC0415

        path = IMAGES_DIR / f"{product_id}.jpg"
        if not path.exists():
            logger.warning("Image not found: %s — using blank placeholder.", path)
            # Use a grey placeholder frame so the vector is still upserted
            return np.full((224, 224, 3), 128, dtype=np.uint8)
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("cv2 could not read: %s — using placeholder.", path)
            return np.full((224, 224, 3), 128, dtype=np.uint8)
        return img
    except Exception as exc:  # noqa: BLE001
        logger.error("Image load error for %s: %s", product_id, exc)
        return np.full((224, 224, 3), 128, dtype=np.uint8)


def load_pdf_text(product_id: str) -> Optional[str]:
    """
    Extract text from pdfs/{product_id}.pdf using pdfplumber.

    Args:
        product_id: UUID string of the product.

    Returns:
        First 1 000 characters of PDF text, or None if no PDF exists.
    """
    path = PDFS_DIR / f"{product_id}.pdf"
    if not path.exists():
        return None
    try:
        import pdfplumber  # noqa: PLC0415

        with pdfplumber.open(str(path)) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            return " ".join(pages_text)[:1_000] if pages_text else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF parse failed for %s: %s", product_id, exc)
        return None


def synthesise_speech(text: str) -> Optional[np.ndarray]:
    """
    Render *text* to a 16 kHz mono float32 waveform using pyttsx3.

    pyttsx3 is used offline (no network) to produce a WAV byte stream which
    is then decoded and resampled if necessary.  The resulting waveform is
    passed to AudioEmbedService so that the stored ``audio_waveform`` vectors
    and live microphone query vectors share the same Whisper acoustic space.

    Args:
        text: Product description string to synthesise into speech.

    Returns:
        1-D float32 numpy array at 16 000 Hz, or None on failure.
    """
    try:
        import pyttsx3          # noqa: PLC0415
        import scipy.io.wavfile  # noqa: PLC0415
        import scipy.signal      # noqa: PLC0415

        engine = pyttsx3.init()
        engine.setProperty("rate", 165)   # words per minute
        engine.setProperty("volume", 1.0)

        # Save speech to an in-memory WAV buffer
        wav_buffer = io.BytesIO()
        engine.save_to_file(text, "_tmp_tts.wav")
        engine.runAndWait()

        # pyttsx3.save_to_file writes to disk; read back and clean up
        import os  # noqa: PLC0415
        if not os.path.exists("_tmp_tts.wav"):
            logger.warning("pyttsx3 did not produce _tmp_tts.wav — falling back.")
            return None

        sr, data = scipy.io.wavfile.read("_tmp_tts.wav")
        os.remove("_tmp_tts.wav")

        # Convert to float32 mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype != np.float32:
            max_val = float(np.iinfo(data.dtype).max) if np.issubdtype(data.dtype, np.integer) else 1.0
            data = data.astype(np.float32) / max_val

        # Resample to 16 kHz if the TTS engine used a different rate
        if sr != AUDIO_SAMPLE_RATE:
            num_samples = int(len(data) * AUDIO_SAMPLE_RATE / sr)
            data = scipy.signal.resample(data, num_samples)
            data = data.astype(np.float32)

        return data

    except Exception as exc:  # noqa: BLE001
        logger.warning("TTS synthesis failed: %s — will use zero-vector for audio_waveform.", exc)
        return None


def ingest(products_path: Optional[Path] = None, store_id: int = DEFAULT_STORE_ID) -> int:
    """
    Full ingestion pipeline: load → embed → upsert.

    Args:
        products_path: Path to the products JSON file.
                       Defaults to data/products.json.
        store_id:      Store identifier to stamp on every payload.

    Returns:
        Number of successfully upserted SKUs.
    """
    products_path = products_path or (DATA_DIR / "products.json")
    if not products_path.exists():
        logger.error("Products file not found: %s", products_path)
        sys.exit(1)

    with open(products_path, encoding="utf-8") as f:
        products: list[dict] = json.load(f)

    logger.info("Loaded %d products from %s", len(products), products_path)

    # Boot services
    embed_svc = EmbedService()
    audio_svc = AudioEmbedService.get_instance()
    client = get_qdrant_client()

    # Resolve vector dim from the live model and create collection if needed
    vector_dim = embed_svc.vector_dim
    logger.info("Resolved vector dimension: %d", vector_dim)
    create_collection(client, vector_dim=vector_dim)

    points: list[PointStruct] = []
    success_count = 0

    for idx, product in enumerate(products, start=1):
        product_id: str = product["product_id"]
        name: str = product.get("name", "")
        description: str = product.get("description", "")

        logger.info(
            "[%d/%d] Embedding '%s' (id=%s) …",
            idx, len(products), name, product_id,
        )

        try:
            # ── barcode_visual vector ────────────────────────────────────────────
            frame = load_product_image(product_id)
            vis_vec = embed_svc.embed(frame, modality="image")

            # ── voice_query vector ───────────────────────────────────────────────
            voice_text = f"{name}. {description}".strip()
            voice_vec = embed_svc.embed(voice_text, modality="text")

            # ── nutrition_pdf vector ────────────────────────────────────────────
            pdf_text = load_pdf_text(product_id)
            if pdf_text:
                pdf_vec = embed_svc.embed(pdf_text, modality="pdf_chunk")
            else:
                # Fall back to voice_query embedding so the named vector
                # is always populated (Qdrant requires all named vectors).
                pdf_vec = voice_vec
                logger.debug("No PDF for %s — using voice_query vec as fallback.", product_id)

            # ── audio_waveform vector (native Whisper acoustic embedding) ──────
            speech_text = f"{name}. {description}".strip()
            waveform = synthesise_speech(speech_text)
            if waveform is not None:
                audio_vec = audio_svc.embed_waveform(waveform)
            else:
                # Graceful fallback: embed a silent waveform so the vector
                # slot is always populated.  Silent audio has a well-defined
                # Whisper encoder output (near-zero energy frames), so it
                # will rank last in any cosine similarity search.
                logger.warning(
                    "TTS failed for '%s' — storing silent-audio fallback vector.",
                    product_id,
                )
                silence = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)  # 1 s of silence
                audio_vec = audio_svc.embed_waveform(silence)

            # Stamp store_id onto the payload (override if already present)
            payload = {
                "product_id":   product_id,
                "name":         name,
                "brand":        product.get("brand", ""),
                "aisle_number": int(product.get("aisle_number", 0)),
                "price":        float(product.get("price", 0.0)),
                "tags":         product.get("tags", []),
                "stock_status": bool(product.get("stock_status", True)),
                "store_id":     store_id,
                "description":  description,
            }

            point = PointStruct(
                id=_uuid_to_int(product_id),
                vector={
                    VECTOR_BARCODE_VISUAL: vis_vec,
                    VECTOR_VOICE_QUERY:    voice_vec,
                    VECTOR_NUTRITION_PDF:  pdf_vec,
                    VECTOR_AUDIO_WAVEFORM: audio_vec,
                },
                payload=payload,
            )
            points.append(point)
            success_count += 1

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to embed product '%s': %s — skipping.", product_id, exc)

    # Batch upsert
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(
            "Batch upsert complete — %d/%d SKUs ingested into '%s'.",
            success_count,
            len(products),
            COLLECTION_NAME,
        )
    else:
        logger.warning("No valid points to upsert.")

    return success_count


def _uuid_to_int(uuid_str: str) -> int:
    """
    Convert a UUID string to a stable unsigned 64-bit integer for Qdrant.

    Args:
        uuid_str: Standard UUID string (with or without hyphens).

    Returns:
        Unsigned integer derived from the UUID's lower 64 bits.
    """
    import uuid  # noqa: PLC0415

    try:
        return uuid.UUID(uuid_str).int & 0xFFFFFFFFFFFFFFFF
    except ValueError:
        # Fallback: hash the string
        return abs(hash(uuid_str)) & 0xFFFFFFFFFFFFFFFF


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Cart catalog ingestion")
    parser.add_argument(
        "--products",
        type=Path,
        default=None,
        help="Path to products.json (default: data/products.json)",
    )
    parser.add_argument(
        "--store-id",
        type=int,
        default=DEFAULT_STORE_ID,
        help="Store identifier to stamp on all products (default: 1)",
    )
    args = parser.parse_args()

    total = ingest(products_path=args.products, store_id=args.store_id)
    print(f"\nIngestion complete -- {total} SKUs in Qdrant Edge.")
