"""
agent_service.py — Agentic RAG loop powered by Gemma 4.

Orchestrates: embed → retrieve → build context → Gemma inference → TTS.
"""

import logging
import threading
from typing import Literal, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import (
    REASONING_MODEL_NAME,
    REASONING_DEVICE,
    USE_CUDA,
    GEMMA_LOAD_IN_4BIT,
    GEMMA_LOAD_IN_8BIT,
    TTS_RATE,
    TTS_VOLUME,
)
from embed_service import EmbedService
from retriever_service import RetrieverService

logger = logging.getLogger(__name__)

InputType = Literal["text", "audio", "image"]

# System prompt injected into every Gemma call
_SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable retail assistant at an Indian supermarket kiosk. "
    "You have been given the top product matches from our store database. "
    "Based only on the provided product data, give the customer a clear, helpful, "
    "spoken-style recommendation in 2-3 sentences. "
    "Mention aisle number and price in INR. "
    "If the customer asked for a dietary alternative, explain why your suggestion fits. "
    "Never mention model names, databases, or technical details."
)


class AgentService:
    """
    Agentic RAG orchestrator that combines embedding, retrieval, and
    Gemma-powered language generation into a single conversational turn.

    The full loop is:
      1. Embed user input via EmbedService.
      2. Retrieve top-K matching products via RetrieverService.
      3. Build a RAG context string.
      4. Call Gemma to generate a spoken-style recommendation.
      5. Speak the response aloud via pyttsx3 (non-blocking thread).
    """

    _instance: Optional["AgentService"] = None

    @classmethod
    def get_instance(
        cls,
        embed_service: EmbedService,
        retriever_service: RetrieverService,
    ) -> "AgentService":
        """Return the process-wide AgentService singleton."""
        if cls._instance is None:
            cls._instance = cls(embed_service, retriever_service)
        return cls._instance

    def __init__(
        self,
        embed_service: EmbedService,
        retriever_service: RetrieverService,
    ) -> None:
        """
        Load Gemma into memory and wire up embed + retrieval services.

        Args:
            embed_service:     Loaded EmbedService instance.
            retriever_service: Loaded RetrieverService instance.
        """
        self._embed = embed_service
        self._retriever = retriever_service
        self._tts_engine = self._init_tts()
        self._llm, self._tokenizer = self._load_gemma()
        logger.info("AgentService ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def respond(
        self,
        user_input: "np.ndarray | str",
        input_type: InputType,
        aisle_number: Optional[int] = None,
        products: Optional[list[dict]] = None,
    ) -> str:
        """
        Execute the full agentic RAG loop and return a spoken recommendation.

        Args:
            user_input:   A text string, audio waveform, or BGR image frame.
            input_type:   One of "text", "audio", or "image".
            aisle_number: Optional aisle filter restriction (visual queries only).
            products:     Pre-fetched product list from the endpoint. When
                          provided the internal retrieval step is skipped so
                          Gemma always reasons about the exact same products
                          that are shown to the customer.

        Returns:
            Natural-language recommendation string from Gemma.
        """
        # Step 1 & 2 — retrieve relevant products (skipped when pre-fetched)
        if products is None:
            products = self._retrieve(user_input, input_type, aisle_number=aisle_number)

        if not products:
            fallback = (
                "I'm sorry, I couldn't find a matching product in our store right now. "
                "Please ask a store associate for help."
            )
            self._speak_async(fallback)
            return fallback

        # Step 3 — build RAG context
        query_str = user_input if input_type == "text" else f"[{input_type} query]"
        if aisle_number is not None:
            query_str += f" (filtered to Aisle {aisle_number})"
        context = self._build_context(products, query_str)

        # Step 4 — call Gemma
        response = self._call_gemma(context)

        # Step 5 — speak aloud (only for audio queries)
        if input_type == "audio":
            self._speak_async(response)

        return response

    # ── Private helpers ───────────────────────────────────────────────────────

    def _retrieve(
        self,
        user_input: "np.ndarray | str",
        input_type: InputType,
        aisle_number: Optional[int] = None,
    ) -> list[dict]:
        """Route the user input to the correct retrieval method."""
        if input_type == "image":
            return self._retriever.search_by_image(user_input, aisle_number=aisle_number)  # type: ignore[arg-type]
        if input_type == "audio":
            return self._retriever.search_by_voice(user_input)  # type: ignore[arg-type]
        # Default: text
        return self._retriever.search_by_text(str(user_input))

    def _build_context(self, products: list[dict], query: str) -> str:
        """
        Format retrieved products into a RAG prompt context block.

        Args:
            products: List of product dicts from RetrieverService.
            query:    Original query string for context awareness.

        Returns:
            Formatted context string to prepend to the Gemma prompt.
        """
        lines = [f"Customer query: {query}\n\nTop matching products from our store:\n"]
        for i, p in enumerate(products, start=1):
            tags_str = ", ".join(p.get("tags", [])) or "none"
            lines.append(
                f"{i}. {p['name']} by {p['brand']}\n"
                f"   Aisle: {p['aisle_number']} | Price: ₹{p['price']:.2f}\n"
                f"   Tags: {tags_str}\n"
                f"   Description: {p['description']}\n"
                f"   Match score: {p['score']}\n"
            )
        return "\n".join(lines)

    def _call_gemma(self, context: str) -> str:
        """
        Generate a response from Gemma using the RAG context.

        Args:
            context: RAG context string built from retrieved products.

        Returns:
            Stripped response string from Gemma.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(REASONING_DEVICE)

        with torch.no_grad():
            output_ids = self._llm.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        logger.debug("Gemma response: %s", response[:120])
        return response

    def _load_gemma(self) -> tuple:
        """
        Load Gemma with QAT quantization appropriate for the current device.

        Returns:
            Tuple of (model, tokenizer).
        """
        logger.info(
            "Loading reasoning model '%s' — 4bit=%s, 8bit=%s …",
            REASONING_MODEL_NAME,
            GEMMA_LOAD_IN_4BIT,
            GEMMA_LOAD_IN_8BIT,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            REASONING_MODEL_NAME, trust_remote_code=True
        )

        if GEMMA_LOAD_IN_4BIT and REASONING_DEVICE == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                REASONING_MODEL_NAME,
                quantization_config=quant_config,
                device_map="auto",
            )
        else:
            # CPU / MPS: device_map is not supported on these backends.
            # Load to CPU first (safe), then move to target device.
            model = AutoModelForCausalLM.from_pretrained(
                REASONING_MODEL_NAME,
                torch_dtype=torch.float32,
            )
            model = model.to(REASONING_DEVICE)

        model.eval()
        logger.info("Reasoning model loaded.")
        return model, tokenizer

    def _init_tts(self) -> bool:
        """
        Check if pyttsx3 TTS engine is available on the system.

        Returns:
            True if available, False otherwise.
        """
        try:
            import pyttsx3  # noqa: PLC0415
            engine = pyttsx3.init()
            engine.setProperty("rate", TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)
            del engine
            logger.info("pyttsx3 TTS engine availability verified.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS unavailable: %s — proceeding without audio output.", exc)
            return False

    def _speak_async(self, text: str) -> None:
        """
        Speak *text* in a daemon thread so the server is never blocked.

        Args:
            text: The string to synthesise and play through the speakers.
        """
        if not self._tts_engine:
            return

        def _run() -> None:
            import sys  # noqa: PLC0415
            is_win = sys.platform == "win32"
            if is_win:
                import ctypes  # noqa: PLC0415
                try:
                    ctypes.windll.ole32.CoInitialize(None)
                except Exception:
                    pass
            try:
                # Re-init per thread because pyttsx3 is not thread-safe
                import pyttsx3  # noqa: PLC0415
                eng = pyttsx3.init()
                try:
                    eng.endLoop()
                except Exception:
                    pass
                eng.setProperty("rate", TTS_RATE)
                eng.setProperty("volume", TTS_VOLUME)
                eng.say(text)
                eng.runAndWait()
                # Clean up references to release locks
                del eng
            except Exception as exc:  # noqa: BLE001
                logger.warning("TTS playback error: %s", exc)
            finally:
                if is_win:
                    try:
                        ctypes.windll.ole32.CoUninitialize()
                    except Exception:
                        pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()


