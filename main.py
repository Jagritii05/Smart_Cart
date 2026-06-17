"""
main.py — FastAPI kiosk server for Smart Cart.

Startup loads both neural models once. All blocking inference calls are
dispatched to a thread-pool executor to keep the asyncio event loop free.

Endpoints:
  POST      /query/text                   — text query → Gemma recommendation
  POST      /query/image                  — image upload → Gemma recommendation
  POST      /query/audio                  — WAV upload  → Gemma recommendation
  PATCH     /products/{product_id}/stock  — real-time inventory update
  GET       /health                       — liveness / readiness probe
  WebSocket /ws/kiosk                     — real-time kiosk state push
"""

import asyncio
import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()  # picks up HF_TOKEN from .env so Gemma download works without env var juggling
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent_service import AgentService
from audio_embed_service import AudioEmbedService
from config import (
    API_HOST,
    API_PORT,
    AUDIO_SAMPLE_RATE,
    COLLECTION_NAME,
    DEFAULT_STORE_ID,
)
from embed_service import EmbedService
from models import (
    HealthResponse,
    ProductResult,
    QueryResponse,
    StockUpdateRequest,
    StockUpdateResponse,
    TextQueryRequest,
)
from qdrant_setup import get_qdrant_client, verify_collection
from retriever_service import RetrieverService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("smart_cart.main")

# ─── Process-wide singletons ───────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=2)
_embed_svc: Optional[EmbedService] = None
_audio_embed_svc: Optional[AudioEmbedService] = None
_retriever_svc: Optional[RetrieverService] = None
_agent_svc: Optional[AgentService] = None
_qdrant_client = None
_models_loaded: bool = False


# ─── WebSocket Connection Manager ─────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.append(websocket)
        logger.info("WebSocket client connected — total=%d", len(self.active))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active:
            self.active.remove(websocket)
        logger.info("WebSocket client disconnected — total=%d", len(self.active))

    async def broadcast(self, data: dict) -> None:
        """Send JSON payload to all connected clients."""
        message = json.dumps(data)
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


_ws_manager = ConnectionManager()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and initialise Qdrant on startup; release on shutdown."""
    global _embed_svc, _audio_embed_svc, _retriever_svc, _agent_svc, _qdrant_client, _models_loaded

    logger.info("Smart Cart kiosk starting — loading models …")

    loop = asyncio.get_running_loop()

    # 1. Load CLIP embedding model (CPU)
    _embed_svc = await loop.run_in_executor(
        _executor, EmbedService.get_instance
    )

    # 2. Load Whisper audio encoder (CPU) — warm-up before first audio query
    _audio_embed_svc = await loop.run_in_executor(
        _executor, AudioEmbedService.get_instance
    )

    # 3. Init Qdrant client
    _qdrant_client = await loop.run_in_executor(_executor, get_qdrant_client)

    # 4. Build collection if it doesn't exist yet
    await loop.run_in_executor(
        _executor,
        partial(
            _safe_create_collection,
            _qdrant_client,
            _embed_svc.vector_dim,
        ),
    )

    # 5. Build retriever
    _retriever_svc = RetrieverService(_qdrant_client, _embed_svc)

    # 6. Load Gemma + wire up agent
    _agent_svc = await loop.run_in_executor(
        _executor,
        partial(AgentService.get_instance, _embed_svc, _retriever_svc),
    )

    _models_loaded = True
    logger.info("All models loaded (CLIP + Whisper + Gemma) — kiosk server is ready.")

    yield

    # Graceful shutdown
    logger.info("Shutting down Smart Cart kiosk …")
    _executor.shutdown(wait=False)


def _safe_create_collection(client, vector_dim: int) -> None:
    """Create the Qdrant collection if it does not already exist."""
    from qdrant_setup import create_collection  # noqa: PLC0415
    create_collection(client, vector_dim=vector_dim)


# ─── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Cart Kiosk API",
    description="Fully offline AI-powered retail assistant — zero cloud, zero cost.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_ready() -> None:
    """Raise 503 if models are still loading."""
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading. Retry in a moment.")


def _to_product_results(products: list[dict]) -> list[ProductResult]:
    """Convert raw retriever dicts to validated ProductResult models."""
    results = []
    for p in products:
        try:
            results.append(ProductResult(**p))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse product result: %s — %s", p, exc)
    return results


async def _run_in_executor(fn, *args):
    """Dispatch a blocking callable to the thread-pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, partial(fn, *args))


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """
    Liveness and readiness probe.

    Returns model load status and Qdrant connectivity.
    """
    qdrant_ok = False
    if _qdrant_client is not None:
        try:
            qdrant_ok = await _run_in_executor(verify_collection, _qdrant_client)
        except Exception:  # noqa: BLE001
            qdrant_ok = False

    return HealthResponse(
        status="ok",
        models_loaded=_models_loaded,
        qdrant_ok=qdrant_ok,
    )


@app.post("/query/text", response_model=QueryResponse, tags=["Query"])
async def query_text(body: TextQueryRequest) -> QueryResponse:
    """
    Accept a natural-language text query and return a Gemma recommendation.

    Body fields:
    - **query**: Customer's question (e.g. "show me a gluten-free bread")
    - **store_id**: Store identifier (default 1)
    - **tags**: Optional dietary tag filters
    """
    _require_ready()

    def _run():
        products = _retriever_svc.search_by_text(
            body.query, tags=body.tags, store_id=body.store_id
        )
        # Pass pre-fetched products so Gemma reasons about the same items the UI shows
        response = _agent_svc.respond(body.query, input_type="text", products=products)
        return response, products

    try:
        response, products = await _run_in_executor(_run)
    except Exception as exc:
        logger.error("Text query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Text query processing error: {exc}") from exc

    return QueryResponse(
        response=response,
        products=_to_product_results(products),
    )


@app.post("/query/image", response_model=QueryResponse, tags=["Query"])
async def query_image(
    file: UploadFile = File(...),
    aisle_number: Optional[int] = Form(None),
) -> QueryResponse:
    """
    Accept an image upload (JPEG/PNG) and return a Gemma recommendation.

    The image is decoded to a BGR numpy array and passed through the
    barcode_visual vector index, optionally filtered to a specific aisle.
    """
    _require_ready()

    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported image type '{file.content_type}'. Use JPEG or PNG.",
        )

    raw_bytes = await file.read()

    def _run():
        import cv2  # noqa: PLC0415

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode uploaded image. Ensure it is a valid JPEG/PNG.")

        products = _retriever_svc.search_by_image(frame, aisle_number=aisle_number)
        # Pass pre-fetched products so Gemma reasons about the same items the UI shows
        response = _agent_svc.respond(
            frame, input_type="image", aisle_number=aisle_number, products=products
        )
        return response, products

    try:
        response, products = await _run_in_executor(_run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Image query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Image query processing error: {exc}") from exc

    return QueryResponse(
        response=response,
        products=_to_product_results(products),
    )


@app.post("/query/audio", response_model=QueryResponse, tags=["Query"])
async def query_audio(file: UploadFile = File(...)) -> QueryResponse:
    """
    Accept a WAV file upload (16 kHz mono) and return a Gemma recommendation.

    Pipeline:
      1. Decode WAV bytes → float32 mono waveform
      2. Transcribe waveform → text via Whisper ASR (encoder + decoder)
      3. Embed transcribed text via Qwen text-embedding path
      4. Query Qdrant ``voice_query`` vector index for semantic product matches
      5. Feed products + query into Gemma retail agent → spoken recommendation

    This two-step approach (ASR → text search) is far more accurate than the
    encoder-only acoustic path because the Qwen text embeddings capture
    *semantic intent* rather than raw phonetics.
    """
    _require_ready()

    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported audio type '{file.content_type}'. Upload a 16 kHz mono WAV file.",
        )

    raw_bytes = await file.read()

    def _run():
        import scipy.io.wavfile as wavfile  # noqa: PLC0415

        sr, data = wavfile.read(io.BytesIO(raw_bytes))

        # Convert to float32 mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype != np.float32:
            max_val = float(np.iinfo(data.dtype).max) if np.issubdtype(data.dtype, np.integer) else 1.0
            data = data.astype(np.float32) / max_val

        # Resample to 16 kHz if needed
        if sr != AUDIO_SAMPLE_RATE:
            import scipy.signal as signal  # noqa: PLC0415
            num_samples = int(len(data) * AUDIO_SAMPLE_RATE / sr)
            data = signal.resample(data, num_samples)

        # ── Step 1: Transcribe speech → text ─────────────────────────────────
        transcribed_text = _audio_embed_svc.transcribe(data)

        if not transcribed_text:
            # Whisper produced no text — audio too short, silent, or noisy
            return (
                "I couldn't make out what you said. Please hold the button, "
                "speak clearly toward the microphone, and try again.",
                [],
            )

        logger.info("Voice query transcribed: %r", transcribed_text)

        # ── Step 2: Semantic text search ─────────────────────────────────────
        products = _retriever_svc.search_by_text(transcribed_text)

        # ── Step 3: Gemma retail agent ────────────────────────────────────────
        # Pass pre-fetched products so Gemma reasons about the same items the UI
        # shows, and use input_type="audio" only to trigger TTS playback.
        response = _agent_svc.respond(
            transcribed_text, input_type="audio", products=products
        )

        return response, products

    try:
        response, products = await _run_in_executor(_run)
    except Exception as exc:
        logger.error("Audio query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Audio query processing error: {exc}") from exc

    return QueryResponse(
        response=response,
        products=_to_product_results(products),
    )


@app.get(
    "/products",
    response_model=list[ProductResult],
    tags=["Inventory"],
)
async def get_products() -> list[ProductResult]:
    """
    Fetch all products with their current real-time stock status from Qdrant.
    """
    _require_ready()

    def _run():
        res, _ = _qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        products = []
        for point in res:
            payload = point.payload or {}
            products.append({
                "product_id":   payload.get("product_id", ""),
                "name":         payload.get("name", "Unknown"),
                "brand":        payload.get("brand", "Unknown"),
                "aisle_number": payload.get("aisle_number", 0),
                "price":        payload.get("price", 0.0),
                "tags":         payload.get("tags", []),
                "description":  payload.get("description", ""),
                "stock_status": payload.get("stock_status", False),
                "score":        1.0,
            })
        # Sort alphabetically by name
        products.sort(key=lambda x: x["name"])
        return products

    try:
        results = await _run_in_executor(_run)
        return _to_product_results(results)
    except Exception as exc:
        logger.error("Failed to scroll products: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve product list from vector DB: {exc}",
        ) from exc


@app.patch(
    "/products/{product_id}/stock",
    response_model=StockUpdateResponse,
    tags=["Inventory"],
)
async def update_stock(product_id: str, body: StockUpdateRequest) -> StockUpdateResponse:
    """
    Update the stock_status payload field for a product without re-embedding.

    Uses Qdrant's set_payload API so the change is instantaneous and does
    not require any neural model inference.

    Args:
    - **product_id**: UUID of the product to update.
    - **stock_status**: True = in stock, False = out of stock.
    """
    _require_ready()

    def _run():
        from ingest import _uuid_to_int  # noqa: PLC0415

        point_id = _uuid_to_int(product_id)
        _qdrant_client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"stock_status": body.stock_status},
            points=[point_id],
        )

    try:
        await _run_in_executor(_run)
    except Exception as exc:
        logger.error("Stock update failed for %s: %s", product_id, exc, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail=f"Could not update stock for product '{product_id}': {exc}",
        ) from exc

    logger.info("Stock updated — product_id=%s  stock_status=%s", product_id, body.stock_status)
    # Broadcast stock change to all connected kiosk clients
    await _ws_manager.broadcast({
        "event": "stock_update",
        "product_id": product_id,
        "stock_status": body.stock_status,
    })
    return StockUpdateResponse(updated=True)


# ─── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/kiosk")
async def kiosk_websocket(websocket: WebSocket):
    """
    Real-time kiosk state push endpoint.

    On connect, immediately sends the current system state (models loaded,
    Qdrant status). Subsequently, broadcasts are triggered by stock updates
    and other server-side events.
    """
    await _ws_manager.connect(websocket)

    # Send current state snapshot to newly connected client
    qdrant_ok = False
    if _qdrant_client is not None:
        try:
            qdrant_ok = await _run_in_executor(verify_collection, _qdrant_client)
        except Exception:  # noqa: BLE001
            qdrant_ok = False

    await websocket.send_text(json.dumps({
        "event": "state_snapshot",
        "models_loaded": _models_loaded,
        "qdrant_ok": qdrant_ok,
    }))

    try:
        # Keep connection alive — client can also send messages (e.g. aisle_change)
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                event = msg.get("event")
                if event == "aisle_change":
                    # Broadcast to all clients so multiple screens stay in sync
                    await _ws_manager.broadcast({
                        "event": "aisle_change",
                        "aisle_number": msg.get("aisle_number"),
                    })
                elif event == "ping":
                    await websocket.send_text(json.dumps({"event": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        _ws_manager.disconnect(websocket)


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        workers=1,  # Single worker — models are process-scoped singletons
    )
