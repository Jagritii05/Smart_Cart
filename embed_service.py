"""
embed_service.py — Unified multimodal embedding service using CLIP.

openai/clip-vit-base-patch32 was trained with contrastive image-text alignment:
images and text for the same concept land in the same region of a shared 512-dim
cosine space.  This makes cross-modal product search accurate without any
bridging or separate adapters.

Audio waveforms are routed to AudioEmbedService (Whisper encoder, also 512-dim).
"""

import logging
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import EMBED_MODEL_NAME, EMBED_DEVICE

logger = logging.getLogger(__name__)

ModalityType = Literal["image", "audio", "text", "pdf_chunk"]


class EmbedService:
    """
    Singleton embedding service backed by CLIP (openai/clip-vit-base-patch32).

    All text and image modalities share a single 512-dim cosine embedding space.
    Audio is handled by the companion AudioEmbedService (Whisper, 512-dim).
    """

    _instance: Optional["EmbedService"] = None

    @classmethod
    def get_instance(cls) -> "EmbedService":
        """Return the process-wide EmbedService singleton, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        logger.info("Loading CLIP model '%s' on device '%s' …", EMBED_MODEL_NAME, EMBED_DEVICE)
        self.processor = CLIPProcessor.from_pretrained(EMBED_MODEL_NAME)
        self.model = CLIPModel.from_pretrained(
            EMBED_MODEL_NAME,
            torch_dtype=torch.float32,
        )
        self.model = self.model.to(EMBED_DEVICE)
        self.model.eval()
        # CLIP ViT-B/32 projects both images and text to 512 dims
        self._vector_dim: int = self.model.config.projection_dim
        logger.info("CLIP loaded — projection_dim=%d, device=%s", self._vector_dim, EMBED_DEVICE)

    def embed(
        self,
        input_data: "np.ndarray | str",
        modality: ModalityType,
    ) -> list[float]:
        """
        Embed *input_data* into a normalised L2 float32 vector.

        Args:
            input_data: A numpy BGR frame (image), float32 waveform (audio),
                        or str (text / pdf_chunk).
            modality:   One of "image", "audio", "text", "pdf_chunk".

        Returns:
            L2-normalised list[float] of length ``vector_dim`` (512).
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
        """Output embedding dimension — 512 for CLIP ViT-B/32."""
        return self._vector_dim

    def _embed_image(self, bgr_frame: np.ndarray) -> list[float]:
        """
        Embed an OpenCV BGR frame through CLIP's vision encoder.

        Args:
            bgr_frame: H×W×3 uint8 numpy array in BGR channel order.

        Returns:
            L2-normalised 512-dim float32 vector.
        """
        rgb = bgr_frame[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb.astype(np.uint8))
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(EMBED_DEVICE)
        with torch.inference_mode():
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            image_features = self.model.visual_projection(vision_out.pooler_output)
        return self._l2_normalise(image_features.squeeze(0).float().cpu()).tolist()

    def _embed_text(self, text: str) -> list[float]:
        """
        Tokenise *text* and encode it through CLIP's text encoder.

        CLIP's context window is 77 tokens; longer strings are truncated.
        Product descriptions in this catalog are all well within that limit.

        Args:
            text: Input string — product name, description, or customer query.

        Returns:
            L2-normalised 512-dim float32 vector.
        """
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        input_ids = inputs["input_ids"].to(EMBED_DEVICE)
        attention_mask = inputs["attention_mask"].to(EMBED_DEVICE)
        with torch.inference_mode():
            text_out = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_features = self.model.text_projection(text_out.pooler_output)
        return self._l2_normalise(text_features.squeeze(0).float().cpu()).tolist()

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
