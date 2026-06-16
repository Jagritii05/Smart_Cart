"""
config.py — Central configuration for the Smart Cart kiosk system.
All constants, model names, paths, and defaults are defined here.
"""

import os
import torch
from pathlib import Path

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.resolve()
DATA_DIR: Path = BASE_DIR / "data"
IMAGES_DIR: Path = DATA_DIR / "images"
PDFS_DIR: Path = DATA_DIR / "pdfs"
QDRANT_STORAGE_PATH: str = str(BASE_DIR / "qdrant_storage")

# ─── Model Identifiers ─────────────────────────────────────────────────────────
EMBED_MODEL_NAME: str = "Qwen/Qwen2.5-VL-3B-Instruct"    # Qwen2.5-VL-3B — smallest true Qwen2.5-VL variant
REASONING_MODEL_NAME: str = "google/gemma-3-4b-it"        # Gemma 3 4B (HF ID)
WHISPER_MODEL_NAME: str = "openai/whisper-base"           # Whisper-base for native audio encoder embeddings

# ─── Qdrant Collection ─────────────────────────────────────────────────────────
COLLECTION_NAME: str = "smart_cart_products"

# Named vector keys (must match qdrant_setup.py and retriever_service.py)
VECTOR_BARCODE_VISUAL: str = "barcode_visual"
VECTOR_VOICE_QUERY: str = "voice_query"
VECTOR_NUTRITION_PDF: str = "nutrition_pdf"
VECTOR_AUDIO_WAVEFORM: str = "audio_waveform"  # Native Whisper encoder embeddings

# Vector dimension is determined at runtime from the loaded model.
# This sentinel value triggers dynamic resolution in qdrant_setup.py.
VECTOR_DIM: int = 2048  # Qwen2.5-VL-3B hidden size; overridden at runtime if needed

# Whisper encoder output dimension — whisper-base has 512 hidden dims.
# The audio_waveform Qdrant named-vector is sized to this constant.
WHISPER_EMBED_DIM: int = 512

# ─── Qdrant / Retrieval Defaults ──────────────────────────────────────────────
DEFAULT_STORE_ID: int = 1
DEFAULT_TOP_K: int = 3

# ─── Audio Settings ────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE: int = 16_000          # Hz — required by Whisper / Qwen audio
AUDIO_MEL_N_MELS: int = 128
AUDIO_MEL_HOP_LENGTH: int = 160          # 10 ms @ 16 kHz
AUDIO_MEL_WIN_LENGTH: int = 400          # 25 ms @ 16 kHz

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

# Hybrid Execution Settings (to avoid VRAM OOM on 6GB GPUs like RTX 3050)
# - Run embedding (Qwen2.5-VL) on CPU to preserve VRAM.
# - Run reasoning (Gemma-3) on GPU (CUDA) for fast generation.
EMBED_DEVICE: str = "cpu"
REASONING_DEVICE: str = DEVICE  # Will be "cuda" if CUDA is available, otherwise "cpu"

# ─── Quantization Policy ───────────────────────────────────────────────────────
# Gemma 4 QAT quantization: 4-bit on CUDA, 8-bit otherwise (CPU/MPS get float32)
GEMMA_LOAD_IN_4BIT: bool = REASONING_DEVICE == "cuda"
GEMMA_LOAD_IN_8BIT: bool = (REASONING_DEVICE != "cuda") and (REASONING_DEVICE != "cpu" and not USE_MPS)
# Qwen2.5-VL uses float16 on GPU, float32 on CPU
EMBED_TORCH_DTYPE = torch.float16 if EMBED_DEVICE == "cuda" else torch.float32

# ─── Server ────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("SMART_CART_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("SMART_CART_PORT", "8000"))

# ─── TTS ───────────────────────────────────────────────────────────────────────
TTS_RATE: int = 165     # words per minute for pyttsx3
TTS_VOLUME: float = 1.0
