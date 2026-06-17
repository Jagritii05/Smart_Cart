"""
embed_service.py — Unified multimodal embedding service using Qwen3-VL-2B.

Qwen/Qwen3-VL-2B-Instruct processes text and images through the same
transformer backbone, giving 1536-dim hidden-state embeddings in a shared
cosine space.  Mean-pooling the last hidden states + L2 normalisation is
the standard extraction recipe for decoder-based VLMs.

Why Qwen3-VL over CLIP for a retail kiosk:
  - Reads product labels, barcodes, and packaging text in-image (OCR-aware)
  - Handles real-world camera frames (blur, angle, mixed lighting)
  - Text and image share the same transformer token space — no separate
    projection heads or modality-gap bridging needed.

Audio waveforms are routed to AudioEmbedService (Whisper encoder, 512-dim).
"""

import logging
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image

from config import EMBED_MODEL_NAME, EMBED_DEVICE

logger = logging.getLogger(__name__)

ModalityType = Literal["image", "audio", "text", "pdf_chunk"]


class EmbedService:
    """
    Singleton embedding service backed by Qwen3-VL-2B-Instruct.

    Text and image modalities share a 1536-dim cosine embedding space.
    Audio is handled by AudioEmbedService (Whisper encoder, 512-dim).
    """

    _instance: Optional["EmbedService"] = None

    @classmethod
    def get_instance(cls) -> "EmbedService":
        """Return the process-wide EmbedService singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: PLC0415

        logger.info(
            "Loading Qwen3-VL model '%s' on device '%s' …", EMBED_MODEL_NAME, EMBED_DEVICE
        )
        self.processor = AutoProcessor.from_pretrained(
            EMBED_MODEL_NAME,
            trust_remote_code=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            EMBED_MODEL_NAME,
            torch_dtype=torch.float32,
        )
        self.model = self.model.to(EMBED_DEVICE)
        self.model.eval()

        # hidden_size is the dimension of mean-pooled last hidden states.
        # Qwen3-VL-2B-Instruct: hidden_size = 1536.
        self._vector_dim: int = self.model.config.text_config.hidden_size
        logger.info(
            "Qwen3-VL loaded — hidden_size=%d, device=%s",
            self._vector_dim,
            EMBED_DEVICE,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(
        self,
        input_data: "np.ndarray | str",
        modality: ModalityType,
    ) -> list[float]:
        """
        Embed *input_data* into a normalised L2 float32 vector.

        Args:
            input_data: BGR numpy frame (image), float32 waveform (audio), or str.
            modality:   One of "image", "audio", "text", "pdf_chunk".

        Returns:
            L2-normalised list[float] of length ``vector_dim`` (1536).
        """
        if modality == "image":
            return self._embed_image(input_data)  # type: ignore[arg-type]
        if modality in ("text", "pdf_chunk"):
            return self._embed_text(str(input_data))
        if modality == "audio":
            return self._embed_audio(input_data)  # type: ignore[arg-type]
        raise ValueError(f"Unsupported modality: {modality!r}")

    @property
    def vector_dim(self) -> int:
        """Output embedding dimension — 1536 for Qwen3-VL-2B."""
        return self._vector_dim

    # ── Private helpers ───────────────────────────────────────────────────────

    def _embed_text(self, text: str) -> list[float]:
        """
        Tokenise *text* and encode it through the language backbone.

        Uses the LM transformer layers only (no vision encoder) for speed.
        Mean-pools over non-padding tokens then L2-normalises.
        """
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(EMBED_DEVICE)
        attention_mask = inputs["attention_mask"].to(EMBED_DEVICE)

        with torch.inference_mode():
            # model.model = language backbone (Qwen3-VL transformer layers)
            out = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden = out.last_hidden_state  # [1, seq_len, 1536]
        mask = attention_mask.unsqueeze(-1).float()
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self._l2_normalise(emb.squeeze(0).float().cpu()).tolist()

    def _embed_image(self, bgr_frame: np.ndarray) -> list[float]:
        """
        Embed a camera frame through the full Qwen3-VL model.

        The vision encoder reads the image (including labels, barcodes, and
        packaging text) and merges visual tokens into the transformer sequence.
        Mean-pooling the last hidden state captures the full visual meaning.

        Args:
            bgr_frame: H×W×3 uint8 numpy array in BGR channel order (OpenCV).

        Returns:
            L2-normalised 1536-dim float32 vector.
        """
        rgb = bgr_frame[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb.astype(np.uint8))

        # Standard Qwen3-VL chat template with embedded image
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": "product"},
            ],
        }]
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(
            text=text_prompt,
            images=[pil_img],
            return_tensors="pt",
        )
        inputs = {k: v.to(EMBED_DEVICE) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Last hidden state contains fused visual + text token representations
        last_hidden = outputs.hidden_states[-1]  # [1, seq_len, 1536]
        emb = last_hidden.mean(dim=1)
        return self._l2_normalise(emb.squeeze(0).float().cpu()).tolist()

    def _embed_audio(self, waveform: np.ndarray) -> list[float]:
        """Route audio to AudioEmbedService (Whisper encoder, 512-dim)."""
        from audio_embed_service import AudioEmbedService  # noqa: PLC0415
        return AudioEmbedService.get_instance().embed_waveform(waveform)

    @staticmethod
    def _l2_normalise(tensor: torch.Tensor) -> torch.Tensor:
        """Return a unit-norm version of *tensor*."""
        norm = tensor.norm(p=2)
        if norm < 1e-12:
            return tensor
        return tensor / norm
