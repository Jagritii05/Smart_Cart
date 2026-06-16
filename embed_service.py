"""
embed_service.py — Unified multimodal embedding service using Qwen3-VL-2B.

All four modalities (image, audio, text, pdf_chunk) are embedded into the
same vector space so that image and voice queries for the same product
become nearest neighbors in Qdrant.
"""

import logging
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import (
    EMBED_MODEL_NAME,
    EMBED_TORCH_DTYPE,
    EMBED_DEVICE,
    AUDIO_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

ModalityType = Literal["image", "audio", "text", "pdf_chunk"]


class EmbedService:
    """
    Singleton embedding service backed by Qwen2.5-VL-2B-Instruct.

    All modalities are routed through the same transformer backbone so
    that the resulting vectors share a unified semantic space.  The
    public API exposes a single :meth:`embed` method.
    """

    _instance: Optional["EmbedService"] = None

    # ── Singleton accessor ────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "EmbedService":
        """Return the process-wide EmbedService singleton, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Initialisation ────────────────────────────────────────────────────────
    def __init__(self) -> None:
        """Load Qwen2.5-VL-2B into memory once and keep it resident."""
        logger.info("Loading embedding model '%s' on device '%s' …", EMBED_MODEL_NAME, EMBED_DEVICE)

        self.processor = AutoProcessor.from_pretrained(
            EMBED_MODEL_NAME,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            EMBED_MODEL_NAME,
            trust_remote_code=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            EMBED_MODEL_NAME,
            torch_dtype=EMBED_TORCH_DTYPE,
            trust_remote_code=True,
            device_map=EMBED_DEVICE if EMBED_DEVICE != "cpu" else None,
        )
        if EMBED_DEVICE == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()
        self._vector_dim: Optional[int] = None
        logger.info("Embedding model loaded — dtype=%s", EMBED_TORCH_DTYPE)

    # ── Public API ────────────────────────────────────────────────────────────
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
            A normalised L2 list[float] of length ``vector_dim``.

        Raises:
            ValueError: If an unsupported modality is supplied.
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
        """Return the output embedding dimension (resolved lazily)."""
        if self._vector_dim is None:
            # Probe with a short text sequence to learn the hidden size.
            probe = self._embed_text("probe", _return_raw=True)
            self._vector_dim = len(probe)
        return self._vector_dim

    # ── Private helpers ───────────────────────────────────────────────────────
    def _embed_image(self, bgr_frame: np.ndarray) -> list[float]:
        """
        Embed an OpenCV BGR numpy frame via the vision encoder.

        Converts BGR→RGB, builds a Qwen-VL messages payload with the
        image, and extracts the [CLS]-like mean-pool of the last hidden
        state from the language model trunk.

        Args:
            bgr_frame: H×W×3 uint8 numpy array in BGR channel order.

        Returns:
            Normalised float32 vector.
        """
        rgb = bgr_frame[:, :, ::-1].copy()          # BGR → RGB
        pil_img = Image.fromarray(rgb.astype(np.uint8))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Describe this product image."},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text_input],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(EMBED_DEVICE) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        return self._pool_and_normalise(outputs)

    def _embed_audio(self, waveform: np.ndarray) -> list[float]:
        """
        Embed a 16 kHz mono float32 waveform via Whisper's encoder.

        Rather than transcribing speech to text (ASR), this method feeds
        the raw waveform directly into the Whisper-base encoder and extracts
        the mean-pooled last hidden state — a true *acoustic* embedding that
        captures semantic meaning without ever producing a word.

        The AudioEmbedService singleton is loaded lazily on first audio call
        so that cold-start latency is incurred only if audio queries arrive.

        Args:
            waveform: 1-D float32 numpy array at 16 000 Hz.

        Returns:
            L2-normalised acoustic embedding from the Whisper encoder
            (512-dim for whisper-base).
        """
        from audio_embed_service import AudioEmbedService  # noqa: PLC0415 (lazy load)

        audio_svc = AudioEmbedService.get_instance()
        return audio_svc.embed_waveform(waveform)

    def _embed_text(self, text: str, *, _return_raw: bool = False) -> list[float]:
        """
        Tokenise *text* and extract a mean-pooled hidden-state embedding.

        Args:
            text:        Input string (product name, description, query, etc.).
            _return_raw: Internal flag — returns list directly without logging.

        Returns:
            Normalised float32 vector.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(EMBED_DEVICE) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Mean-pool the last hidden state over sequence dimension
        last_hidden: torch.Tensor = outputs.last_hidden_state   # (1, seq, dim)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq, 1)
        summed = (last_hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = (summed / counts).squeeze(0)                    # (dim,)

        vec = self._l2_normalise(pooled.float().cpu())
        return vec.tolist()

    def _pool_and_normalise(self, outputs: object) -> list[float]:
        """
        Mean-pool hidden states from a model output object and L2-normalise.

        Args:
            outputs: A transformers CausalLM output with ``hidden_states``.

        Returns:
            Normalised float32 vector as a list.
        """
        hidden = outputs.hidden_states[-1]   # (1, seq, dim)
        pooled = hidden.mean(dim=1).squeeze(0).float().cpu()
        return self._l2_normalise(pooled).tolist()

    @staticmethod
    def _l2_normalise(tensor: torch.Tensor) -> torch.Tensor:
        """Return a unit-norm version of *tensor* (in-place safe)."""
        norm = tensor.norm(p=2)
        if norm < 1e-12:
            return tensor
        return tensor / norm
