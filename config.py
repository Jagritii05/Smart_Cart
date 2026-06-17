"""
config.py — Central configuration for the Smart Cart kiosk system.
All constants, model names, paths, and defaults are defined here.
"""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # loads HF_TOKEN from .env before any transformers import

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.resolve()
DATA_DIR: Path = BASE_DIR / "data"
IMAGES_DIR: Path = DATA_DIR / "images"
PDFS_DIR: Path = DATA_DIR / "pdfs"
QDRANT_STORAGE_PATH: str = str(BASE_DIR / "qdrant_storage")

# ─── Model Identifiers ─────────────────────────────────────────────────────────
# Qwen3-VL-2B-Instruct: multimodal VLM with unified text+image token space.
# Text and image inputs share the same 1536-dim hidden state space, enabling
# cross-modal cosine similarity without separate projection heads.
EMBED_MODEL_NAME: str = "Qwen/Qwen3-VL-2B-Instruct"
REASONING_MODEL_NAME: str = "google/gemma-3-4b-it"   # Gemma 3 4B — requires HF token + accepted terms
WHISPER_MODEL_NAME: str = "openai/whisper-base"       # Whisper-base for audio encoder + ASR

# ─── Qdrant Collection ─────────────────────────────────────────────────────────
COLLECTION_NAME: str = "smart_cart_products"

# Named vector keys (must match qdrant_setup.py and retriever_service.py)
VECTOR_BARCODE_VISUAL: str = "barcode_visual"
VECTOR_VOICE_QUERY: str = "voice_query"
VECTOR_NUTRITION_PDF: str = "nutrition_pdf"
VECTOR_AUDIO_WAVEFORM: str = "audio_waveform"

# Qwen3-VL-2B-Instruct hidden_size = 1536. This is the dimension of mean-pooled
# last hidden states used as cross-modal embeddings.
# Whisper-base encoder outputs 512 dims — audio_waveform vector stays at 512.
VECTOR_DIM: int = 2048
WHISPER_EMBED_DIM: int = 512

# ─── Qdrant / Retrieval Defaults ──────────────────────────────────────────────
DEFAULT_STORE_ID: int = 1
DEFAULT_TOP_K: int = 3

# ─── Audio Settings ────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE: int = 16_000   # Hz — required by Whisper
AUDIO_MEL_N_MELS: int = 128
AUDIO_MEL_HOP_LENGTH: int = 160   # 10 ms @ 16 kHz
AUDIO_MEL_WIN_LENGTH: int = 400   # 25 ms @ 16 kHz

# ─── Device Detection ──────────────────────────────────────────────────────────
def _detect_device() -> str:
    """Return the best available torch device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE: str = _detect_device()
USE_CUDA: bool = DEVICE == "cuda"
USE_MPS: bool = DEVICE == "mps"

# CLIP is small (~600 MB) so it stays on CPU, leaving GPU headroom for Gemma.
EMBED_DEVICE: str = "cpu"
REASONING_DEVICE: str = DEVICE

# ─── Quantization Policy ───────────────────────────────────────────────────────
# 4-bit NF4 on CUDA (requires bitsandbytes).  MPS and CPU use float32.
# bitsandbytes does not support MPS, so no 8-bit path is needed.
GEMMA_LOAD_IN_4BIT: bool = REASONING_DEVICE == "cuda"
GEMMA_LOAD_IN_8BIT: bool = False  # bitsandbytes MPS support not available

# ─── Server ────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("SMART_CART_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("SMART_CART_PORT", "8000"))

# ─── TTS ───────────────────────────────────────────────────────────────────────
TTS_RATE: int = 165     # words per minute for pyttsx3
TTS_VOLUME: float = 1.0
