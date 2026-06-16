"""
audio_embed_service.py — Native Multimodal Audio Embedding + ASR via Whisper.

Two public methods:

    embed_waveform(waveform)  →  512-dim acoustic embedding (encoder-only)
    transcribe(waveform)      →  transcribed text string (encoder + decoder)

embed_waveform is used at *ingest time* to store acoustic vectors in Qdrant.
transcribe is used at *query time* to convert spoken customer intent into
text that can be matched via the semantic Qwen text-embedding path.
"""

import logging
from typing import Optional

import numpy as np
import torch

from config import WHISPER_MODEL_NAME, AUDIO_SAMPLE_RATE, EMBED_DEVICE

logger = logging.getLogger(__name__)


class AudioEmbedService:
    """
    Singleton audio service backed by OpenAI Whisper-base.

    Public API:
        embed_waveform(waveform) → list[float]   512-dim L2-normalised vector
        transcribe(waveform)     → str            ASR transcription of speech
    """

    _instance: Optional["AudioEmbedService"] = None

    @classmethod
    def get_instance(cls) -> "AudioEmbedService":
        """Return the process-wide AudioEmbedService singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self) -> None:
        """
        Load the Whisper encoder model and feature extractor.

        WhisperModel (base class) has both encoder and decoder weights but
        no LM head, so RAM usage is lower than WhisperForConditionalGeneration.
        The transcribe() method lazy-loads the generation model on first call.
        """
        logger.info(
            "Loading Whisper audio encoder '%s' on device '%s' …",
            WHISPER_MODEL_NAME,
            EMBED_DEVICE,
        )

        from transformers import WhisperFeatureExtractor, WhisperModel  # noqa: PLC0415

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            WHISPER_MODEL_NAME
        )
        self.model = WhisperModel.from_pretrained(
            WHISPER_MODEL_NAME,
            torch_dtype=torch.float32,
        )
        self.model = self.model.to(EMBED_DEVICE)
        self.model.eval()

        self._embed_dim: int = self.model.config.d_model
        logger.info(
            "Whisper encoder loaded — encoder_dim=%d, device=%s",
            self._embed_dim,
            EMBED_DEVICE,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_waveform(self, waveform: np.ndarray) -> list[float]:
        """
        Embed a raw audio waveform into a normalised 512-dim acoustic vector.

        Used at ingest time to store acoustic vectors in Qdrant's
        ``audio_waveform`` named vector index.

        Args:
            waveform: 1-D float32 numpy array sampled at 16 000 Hz.

        Returns:
            L2-normalised float32 list of length 512.
        """
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        waveform = waveform.astype(np.float32)

        peak = float(np.abs(waveform).max())
        if peak > 1e-6:
            waveform = waveform / peak

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        input_features: torch.Tensor = inputs["input_features"].to(EMBED_DEVICE)

        with torch.inference_mode():
            encoder_outputs = self.model.encoder(
                input_features,
                output_hidden_states=False,
                return_dict=True,
            )

        last_hidden: torch.Tensor = encoder_outputs.last_hidden_state
        pooled = last_hidden.mean(dim=1).squeeze(0).float().cpu()
        vec = self._l2_normalise(pooled)
        return vec.tolist()

    def transcribe(self, waveform: np.ndarray) -> str:
        """
        Transcribe a raw audio waveform to text using Whisper ASR.

        Runs the full encoder+decoder pipeline to convert spoken customer
        intent into a text string.  The text is then passed to the Qwen
        semantic text-embedding path, which gives far more accurate product
        matching than the acoustic-only approach.

        The generation model (WhisperForConditionalGeneration) is lazy-loaded
        on the first call and cached on the instance so subsequent calls are
        fast.

        Args:
            waveform: 1-D float32 numpy array sampled at 16 000 Hz.

        Returns:
            Transcribed text string.  Returns empty string on failure.
        """
        from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: PLC0415

        # Lazy-load the full generation model once
        if not hasattr(self, "_gen_model"):
            logger.info("Loading Whisper generation model for ASR transcription …")
            self._gen_model = WhisperForConditionalGeneration.from_pretrained(
                WHISPER_MODEL_NAME,
                torch_dtype=torch.float32,
            ).to(EMBED_DEVICE)
            self._gen_model.eval()
            self._processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
            logger.info("Whisper generation model ready.")

        try:
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=-1)
            waveform = waveform.astype(np.float32)

            peak = float(np.abs(waveform).max())
            if peak > 1e-6:
                waveform = waveform / peak

            inputs = self._processor(
                waveform,
                sampling_rate=AUDIO_SAMPLE_RATE,
                return_tensors="pt",
            )
            input_features = inputs["input_features"].to(EMBED_DEVICE)

            with torch.inference_mode():
                predicted_ids = self._gen_model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    max_new_tokens=128,
                )

            text: str = self._processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            logger.info("Whisper transcription: %r", text)
            return text

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Whisper transcription failed: %s — falling back to empty string", exc
            )
            return ""

    @property
    def embed_dim(self) -> int:
        """Return the encoder output dimension (512 for whisper-base)."""
        return self._embed_dim

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalise(tensor: torch.Tensor) -> torch.Tensor:
        """Return a unit-norm copy of *tensor*."""
        norm = tensor.norm(p=2)
        if norm < 1e-12:
            return tensor
        return tensor / norm
