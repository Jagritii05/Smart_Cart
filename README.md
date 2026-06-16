# 🛒 Smart Cart — Offline AI Retail Kiosk

A fully offline, zero-cloud, zero-API-cost intelligent shopping assistant that runs entirely on local edge hardware. Camera → voice → text queries resolve against a local Qdrant Edge vector database and a quantised Gemma reasoning model. No internet required after the initial model download.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| Storage | 40 GB free | 80 GB SSD |
| GPU | None (CPU mode) | NVIDIA RTX 3090+ (CUDA) |
| OS | Windows 10/11, Ubuntu 22.04, macOS 13+ | Linux with CUDA |
| Python | 3.11+ | 3.11 |

> **CPU-only mode** runs but is slow for Gemma inference (~2–4 min/response). GPU recommended for kiosk use.

---

## Architecture Overview

```
Customer Input
    │
    ├─ Camera Frame ──────────────────────┐
    ├─ WAV Audio (voice) ─────────────────┤
    └─ Text Query ────────────────────────┤
                                          ▼
                               ┌─────────────────────┐
                               │   EmbedService      │
                               │  Qwen2.5-VL-2B      │
                               │ (image/audio/text)  │
                               └──────────┬──────────┘
                                          │  unified vector
                                          ▼
                               ┌─────────────────────┐
                               │   Qdrant Edge       │
                               │  (file-backed,      │
                               │   no server)        │
                               │                     │
                               │  named vectors:     │
                               │  • barcode_visual   │
                               │  • voice_query      │
                               │  • nutrition_pdf    │
                               └──────────┬──────────┘
                                          │  top-3 products
                                          ▼
                               ┌─────────────────────┐
                               │   AgentService      │
                               │  Gemma 3 4B (RAG)   │
                               └──────────┬──────────┘
                                          │
                               ┌──────────┴──────────┐
                               ▼                     ▼
                        JSON Response           pyttsx3 TTS
                         (FastAPI)             (spoken aloud)
```

---

## Project Structure

```
smart_cart/
├── main.py              # FastAPI app + startup model loading
├── embed_service.py     # EmbedService — all-modality embedder
├── retriever_service.py # RetrieverService — Qdrant search layer
├── agent_service.py     # AgentService — Gemma RAG orchestrator
├── ingest.py            # Catalog ingestion script
├── qdrant_setup.py      # Collection creation + index setup
├── models.py            # Pydantic request/response schemas
├── config.py            # All constants and device detection
├── requirements.txt     # Pinned dependencies
├── README.md            # This file
└── data/
    ├── products.json    # 10 sample Indian supermarket products
    ├── images/          # Product images: {product_id}.jpg
    └── pdfs/            # Nutrition PDFs: {product_id}.pdf (optional)
```

---

## Setup Instructions

### 1. Create and activate a Python 3.11 virtual environment

```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

**CPU-only (Windows or macOS without Apple Silicon GPU):**
```bash
pip install -r requirements.txt
```

**CUDA (Linux / Windows with NVIDIA GPU — recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes>=0.42.0
pip install -r requirements.txt
```

**macOS Apple Silicon (MPS):**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

> `flash-attn` is auto-skipped on Windows/macOS. The system will still work — attention will use the standard implementation.

### 3. Download models from Hugging Face

The models download automatically the first time you start the server or run `ingest.py`. Ensure you have an internet connection for this one-time download:

| Model | Size on disk | Notes |
|-------|-------------|-------|
| `Qwen/Qwen2.5-VL-2B-Instruct` | ~5 GB | Embedding model |
| `google/gemma-3-4b-it` | ~9 GB | Reasoning model |

To pre-download (optional, avoids timeout at first startup):
```python
from transformers import AutoProcessor, AutoTokenizer
AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-2B-Instruct", trust_remote_code=True)
AutoTokenizer.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True)
```

### 4. (Optional) Add product images and PDFs

- Place product images at `data/images/{product_id}.jpg`
- Place nutrition PDFs at `data/pdfs/{product_id}.pdf`

If files are missing, placeholders are used automatically — ingestion never fails due to missing assets.

### 5. Ingest the product catalog

```bash
cd smart_cart
python ingest.py
```

Expected output:
```
2024-...  INFO  ingest — Loaded 10 products from data/products.json
2024-...  INFO  ingest — Resolved vector dimension: 1536
2024-...  INFO  ingest — [1/10] Embedding 'Amul Taaza Full Cream Milk 1L' ...
...
2024-...  INFO  ingest — Batch upsert complete — 10/10 SKUs ingested into 'smart_cart_products'.

✅  Ingestion complete — 10 SKUs in Qdrant Edge.
```

Custom catalog or store:
```bash
python ingest.py --products /path/to/my_products.json --store-id 2
```

### 6. Start the kiosk server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Server output:
```
INFO  smart_cart.main — Smart Cart kiosk starting — loading models …
INFO  smart_cart.main — All models loaded — kiosk server is ready.
INFO  uvicorn — Application startup complete.
```

---

## API Reference — Sample curl Commands

### GET /health — Liveness probe
```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Expected response:
```json
{
  "status": "ok",
  "models_loaded": true,
  "qdrant_ok": true
}
```

---

### POST /query/text — Text query
```bash
curl -s -X POST http://localhost:8000/query/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me a gluten-free alternative to wheat bread",
    "store_id": 1,
    "tags": ["gluten-free"]
  }' | python -m json.tool
```

Expected response:
```json
{
  "response": "Great news! We have Slurrp Farm Millet Pancake Mix in Aisle 5 for ₹289.00 — it's made entirely from jowar and ragi with zero wheat, making it naturally gluten-free. You might also love our True Elements Rolled Oats in Aisle 5 for ₹349.00, which are certified gluten-free and packed with fibre.",
  "products": [
    {
      "name": "Slurrp Farm Millet Pancake Mix 200g",
      "brand": "Slurrp Farm",
      "aisle_number": 5,
      "price": 289.0,
      "tags": ["gluten-free", "vegan", "no-maida", "kids-friendly"],
      "description": "...",
      "stock_status": true,
      "score": 0.9134
    }
  ]
}
```

---

### POST /query/image — Image upload
```bash
curl -s -X POST http://localhost:8000/query/image \
  -F "file=@/path/to/product_photo.jpg" | python -m json.tool
```

---

### POST /query/audio — Audio (WAV) upload
```bash
curl -s -X POST http://localhost:8000/query/audio \
  -F "file=@/path/to/query.wav" | python -m json.tool
```

> WAV file must be 16 kHz, mono. Use `ffmpeg` to convert:
> ```bash
> ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
> ```

---

### PATCH /products/{product_id}/stock — Real-time inventory update

Mark a product as out of stock:
```bash
curl -s -X PATCH \
  http://localhost:8000/products/a1b2c3d4-0001-0001-0001-000000000001/stock \
  -H "Content-Type: application/json" \
  -d '{"stock_status": false}' | python -m json.tool
```

Expected response:
```json
{"updated": true}
```

Restore to in-stock:
```bash
curl -s -X PATCH \
  http://localhost:8000/products/a1b2c3d4-0001-0001-0001-000000000001/stock \
  -H "Content-Type: application/json" \
  -d '{"stock_status": true}' | python -m json.tool
```

---

## Interactive API Docs

After the server starts, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SMART_CART_HOST` | `0.0.0.0` | Server bind address |
| `SMART_CART_PORT` | `8000` | Server port |

---

## Offline Operation

Once models are downloaded (one-time internet required), the system operates **fully offline**:

- Qdrant Edge runs as a file-backed client — no network, no server process
- Qwen2.5-VL-2B and Gemma are cached in `~/.cache/huggingface/hub/`
- pyttsx3 uses OS TTS engines (SAPI on Windows, espeak on Linux, NSSpeechSynthesizer on macOS)

---

## Adding New Products

1. Add entries to `data/products.json` following the existing schema.
2. Place the product image at `data/images/{product_id}.jpg`.
3. (Optional) Place the nutrition PDF at `data/pdfs/{product_id}.pdf`.
4. Run `python ingest.py` — existing products are not duplicated (Qdrant upsert is idempotent).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Enable 4-bit by ensuring bitsandbytes is installed; or switch to CPU mode |
| `flash-attn` build fails on Windows | It is skipped automatically — no action needed |
| TTS is silent | On Linux, install espeak: `sudo apt install espeak` |
| `ModuleNotFoundError: pyttsx3` | `pip install pyttsx3` |
| Server reports 503 on first request | Models are still loading — wait 30–60 seconds and retry |
| Qdrant collection empty after ingest | Check that `python ingest.py` completed without errors |
