# 🛒 Smart Cart — Offline AI Retail Kiosk

> A fully offline, multimodal AI shopping assistant for supermarket kiosks.  
> Ask in plain English. Get aisle number, price, and a spoken recommendation.  
> No cloud. No API keys. No internet after setup.

---

<!-- SCREENSHOT: Add your kiosk UI screenshot here (localhost:3000 with a search result) -->
<!-- ![Kiosk UI](docs/screenshots/kiosk-ui.png) -->

---

## What It Does

You walk up to the kiosk. You type (or say, or scan) "dairy free milk." The kiosk tells you:

> *"For a dairy-free option, I'd recommend So Good Almond Milk Unsweetened in Aisle 7 for ₹249. It's vegan, gluten-free, and zero lactose — perfect if you're avoiding dairy entirely!"*

That's it. No staff needed. No keyword matching. The system understands what you mean, not just what you typed.

Built on **CLIP** for semantic embedding, **Qdrant Edge** for local vector search, and **Gemma 3 4B** for natural language generation — all running on your hardware.

---

## Screenshots

### Kiosk UI — Text Search
<!-- SCREENSHOT: localhost:3000 showing a text query and Gemma's response -->
<!-- ![Text Search](docs/screenshots/text-search.png) -->
*Add your screenshot here: open localhost:3000, search for something, screenshot the result.*

---

### Semantic Retrieval Scores (CLIP)
<!-- SCREENSHOT: Terminal output from test_clip.py showing similarity scores -->
<!-- ![CLIP Retrieval](docs/screenshots/clip-retrieval.png) -->
*Add your screenshot here: the terminal output showing "dairy free milk → So Good Almond Milk 0.6955"*

---

### API Docs (Swagger UI)
<!-- SCREENSHOT: localhost:8000/docs showing all endpoints -->
<!-- ![Swagger UI](docs/screenshots/swagger-ui.png) -->
*Add your screenshot here: open localhost:8000/docs in browser.*

---

### Gemma Response (curl)
<!-- SCREENSHOT: Terminal showing the curl POST /query/text response -->
<!-- ![Gemma Response](docs/screenshots/gemma-response.png) -->
*Add your screenshot here: the curl output with the full Gemma recommendation JSON.*

---

## Architecture

```
Customer Input (text / voice / camera)
        │
        ├─ Text ──────┐
        ├─ Audio ─────┤──► EmbedService
        └─ Image ─────┘    (CLIP ViT-B/32 + Whisper-base)
                                │
                                │  512-dim unified vector
                                ▼
                        ┌──────────────┐
                        │  Qdrant Edge │  ← file-backed, no server
                        │  4 named     │
                        │  vectors per │
                        │  product     │
                        └──────┬───────┘
                               │  top-3 products
                               ▼
                        ┌──────────────┐
                        │  AgentService│  ← Gemma 3 4B (RAG)
                        │  + TTS       │
                        └──────┬───────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             JSON Response          pyttsx3 TTS
              (FastAPI)            (spoken aloud)
```

---

## Tech Stack

| Layer | Component | Notes |
|-------|-----------|-------|
| Embedding | CLIP ViT-B/32 | 512-dim unified text+image space |
| Audio | Whisper-base | Same 512-dim encoder — no bridging |
| Reasoning | Gemma 3 4B | Local LLM, no API key needed |
| Vector DB | Qdrant Edge | File-backed, runs as a Python library |
| Backend | FastAPI + Uvicorn | Async, WebSocket, thread-pool inference |
| Frontend | Next.js 14 | Camera, mic, text, cart, admin dashboard |
| Language | Python 3.11 + TypeScript | Type-safe end to end |

---

## Project Structure

```
Smart_Cart/
├── main.py              # FastAPI server — endpoints + lifespan model loading
├── embed_service.py     # CLIP text + image + audio embeddings
├── retriever_service.py # Qdrant search — text, image, voice
├── agent_service.py     # Gemma RAG orchestrator + TTS
├── audio_embed_service.py # Whisper encoder + ASR transcription
├── ingest.py            # One-time product catalogue ingestion
├── qdrant_setup.py      # Collection creation + named vector config
├── models.py            # Pydantic request/response schemas
├── config.py            # All constants, device detection, quantisation flags
├── requirements.txt     # Pinned dependencies
└── data/
    ├── products.json    # 10 sample Indian supermarket products
    ├── images/          # Product images: {product_id}.jpg
    └── pdfs/            # Nutrition PDFs: {product_id}.pdf (optional)

kiosk-ui/                # Next.js 14 kiosk frontend
├── src/app/page.tsx     # Main kiosk page — search, camera, voice, cart
└── src/lib/api.ts       # Typed fetch wrappers for FastAPI
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Jagritii05/Smart_Cart.git
cd Smart_Cart
python3.11 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 2. Install Python dependencies

```bash
# macOS / CPU
pip install -r requirements.txt

# CUDA (Linux / Windows with NVIDIA GPU — enables 4-bit Gemma)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes>=0.42.0
pip install -r requirements.txt
```

### 3. Add your HuggingFace token

Gemma 3 requires accepting the terms on HuggingFace and providing a token.

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

Get your token at: https://huggingface.co/settings/tokens  
Accept Gemma terms at: https://huggingface.co/google/gemma-3-4b-it

### 4. Ingest the product catalogue

```bash
python3.11 ingest.py
```

Expected output:
```
INFO  ingest — Loaded 10 products from data/products.json
INFO  ingest — [1/10] a1b2c3d4-... Amul Taaza Full Cream Milk 1L ✓
...
INFO  ingest — Batch upsert complete — 10/10 SKUs ingested.
✅  Ingestion complete — 10 SKUs in Qdrant Edge.
```

### 5. Start the backend

```bash
python3.11 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete.` — Gemma takes 3–5 minutes to load on first run.

### 6. Start the kiosk UI

```bash
cd kiosk-ui
npm install
npm run dev
```

Open **http://localhost:3000** and start searching.

---

## API Reference

Full interactive docs at **http://localhost:8000/docs** once the server is running.

### Health check
```bash
curl http://localhost:8000/health
# {"status":"ok","models_loaded":true,"qdrant_ok":true}
```

### Text query
```bash
curl -X POST http://localhost:8000/query/text \
  -H "Content-Type: application/json" \
  -d '{"query": "where is milk"}' \
  --max-time 600
```

Response:
```json
{
  "response": "Okay, looking for milk? I'd recommend the Amul Taaza Full Cream Milk — it's in Aisle 3 and costs just ₹68.00. It's fresh, creamy cow's milk packed with calcium and protein!",
  "products": [
    {
      "name": "Amul Taaza Full Cream Milk 1L",
      "brand": "Amul",
      "aisle_number": 3,
      "price": 68.0,
      "tags": ["vegetarian", "dairy"],
      "score": 0.5361
    }
  ]
}
```

### Image query (barcode / product photo)
```bash
curl -X POST http://localhost:8000/query/image \
  -F "file=@product_photo.jpg"
```

### Voice query (WAV file)
```bash
curl -X POST http://localhost:8000/query/audio \
  -F "file=@query.wav"
# WAV must be 16kHz mono — convert with: ffmpeg -i input.mp3 -ar 16000 -ac 1 out.wav
```

---

## Sample Queries to Try

| Query | Expected Result |
|-------|----------------|
| `where is milk` | Amul Taaza Full Cream Milk, Aisle 3 |
| `dairy free milk` | So Good Almond Milk, Aisle 7 |
| `vegan protein source` | Down To Earth Red Lentils, Aisle 9 |
| `something for kids breakfast` | Soulfull Ragi Bites or Slurrp Farm Pancake Mix |
| `keto friendly fat` | Conscious Food Coconut Oil, Aisle 8 |
| `iron rich supplement` | Wellbeing Nutrition Moringa Powder, Aisle 11 |
| `probiotic food` | Epigamia Greek Yogurt, Aisle 3 |

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---------|-------------|
| RAM | 16 GB | 32 GB |
| Storage | 15 GB free | 40 GB SSD |
| GPU | None (MPS/CPU) | NVIDIA RTX 3080+ (CUDA) |
| Python | 3.11 | 3.11 |
| OS | macOS 13+, Ubuntu 22.04, Windows 10 | Linux + CUDA |

> **Note on inference speed:** Gemma 3 4B on Apple Silicon MPS (float32) takes 2–5 min/response. On CUDA with 4-bit NF4 quantisation it takes under 10 seconds. The retrieval step (CLIP + Qdrant) is always under 100ms regardless of hardware.

---

## How Semantic Search Works

The original version of this project had a bug: ask "where is milk" and it returned pancake mix. The fix was replacing the generative embedding model with CLIP.

CLIP (Contrastive Language-Image Pretraining) is trained to pull matching text-image pairs together in the same 512-dimensional space. "Dairy free milk" and the product entry for almond milk end up near each other. Pancake mix does not.

Similarity scores after the fix:

| Query | Top Result | Score |
|-------|-----------|-------|
| `where is milk` | Amul Taaza Full Cream Milk | 0.605 |
| `dairy free milk` | So Good Almond Milk | 0.695 |
| `gluten-free bread alternative` | True Elements Rolled Oats | 0.617 |

No keyword overlap needed. Pure semantic similarity.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `curl: Failed to connect to localhost port 8000` | Server not started yet, or still loading Gemma (wait 3-5 min) |
| UI shows "API error 500" | Check uvicorn terminal for traceback |
| `ModuleNotFoundError` | Make sure you're using `.venv/bin/python3.11`, not system python |
| Qdrant lock error | `rm -f qdrant_storage/.lock` then restart server |
| TTS silent on macOS | Normal — pyttsx3 uses NSSpeechSynthesizer, check System Preferences → Accessibility |
| `torch_dtype is deprecated` | Safe to ignore — cosmetic warning from Transformers |

---

## Adding Your Own Products

1. Add entries to `data/products.json` following the existing schema
2. Drop product images at `data/images/{product_id}.jpg` (optional but recommended for image search)
3. Run `python3.11 ingest.py` — existing products are not duplicated (Qdrant upsert is idempotent)

---

## References

- [Qdrant Edge Docs](https://qdrant.tech/documentation/edge/)
- [CLIP Paper — Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- [CLIP ViT-B/32 on HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32)
- [Gemma 3 4B on HuggingFace](https://huggingface.co/google/gemma-3-4b-it)
- [Whisper-base on HuggingFace](https://huggingface.co/openai/whisper-base)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

---

*Built as part of a Modern 2026 AI Stack tutorial. Full article on Medium.*
