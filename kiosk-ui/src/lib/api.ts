/**
 * lib/api.ts — Typed fetch wrappers for the Smart Cart FastAPI backend.
 * All calls go through Next.js proxy rewrites (/api/* → localhost:8000/*).
 */

export const API_BASE = "/api";
export const WS_URL =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8000/ws/kiosk`
    : "ws://localhost:8000/ws/kiosk";

// ─── Response Types ────────────────────────────────────────────────────────────

export interface ProductResult {
  product_id?: string;
  name: string;
  brand: string;
  aisle_number: number;
  price: number;
  tags: string[];
  description: string;
  stock_status: boolean;
  score: number;
}

export interface QueryResponse {
  response: string;
  products: ProductResult[];
}

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  qdrant_ok: boolean;
}

export interface StockUpdateResponse {
  updated: boolean;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ─── Endpoints ────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse<HealthResponse>(res);
}

export async function getProducts(): Promise<ProductResult[]> {
  const res = await fetch(`${API_BASE}/products`);
  return handleResponse<ProductResult[]>(res);
}

export async function queryText(
  query: string,
  storeId = 1,
  tags?: string[]
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, store_id: storeId, tags }),
  });
  return handleResponse<QueryResponse>(res);
}

export async function queryImage(
  imageBlob: Blob,
  aisleNumber?: number | null
): Promise<QueryResponse> {
  const form = new FormData();
  form.append("file", imageBlob, "snapshot.jpg");
  if (aisleNumber != null) {
    form.append("aisle_number", String(aisleNumber));
  }
  const res = await fetch(`${API_BASE}/query/image`, {
    method: "POST",
    body: form,
  });
  return handleResponse<QueryResponse>(res);
}

export async function queryAudio(wavBlob: Blob): Promise<QueryResponse> {
  const form = new FormData();
  form.append("file", wavBlob, "query.wav");
  const res = await fetch(`${API_BASE}/query/audio`, {
    method: "POST",
    body: form,
  });
  return handleResponse<QueryResponse>(res);
}

export async function updateStock(
  productId: string,
  stockStatus: boolean
): Promise<StockUpdateResponse> {
  const res = await fetch(`${API_BASE}/products/${productId}/stock`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stock_status: stockStatus }),
  });
  return handleResponse<StockUpdateResponse>(res);
}
