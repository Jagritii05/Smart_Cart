"""
models.py — Pydantic request and response schemas for the Smart Cart API.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ─── Request Schemas ───────────────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    """Body for POST /query/text."""

    query: str = Field(..., min_length=1, description="Customer's natural-language query.")
    store_id: int = Field(default=1, ge=1, description="Store identifier for multi-tenant filtering.")
    tags: Optional[list[str]] = Field(
        default=None,
        description="Optional dietary tag filters (e.g. ['vegan', 'gluten-free']).",
    )


class StockUpdateRequest(BaseModel):
    """Body for PATCH /products/{product_id}/stock."""

    stock_status: bool = Field(..., description="True = in stock; False = out of stock.")


# ─── Response Schemas ──────────────────────────────────────────────────────────

class ProductResult(BaseModel):
    """Single product match returned by retrieval."""

    product_id: str = ""
    name: str
    brand: str
    aisle_number: int
    price: float
    tags: list[str]
    description: str
    stock_status: bool
    score: float


class QueryResponse(BaseModel):
    """Standard response for all /query/* endpoints."""

    response: str = Field(..., description="Gemma-generated natural-language recommendation.")
    products: list[ProductResult] = Field(
        ..., description="Top-K products retrieved from Qdrant."
    )


class StockUpdateResponse(BaseModel):
    """Response for PATCH /products/{product_id}/stock."""

    updated: bool = True


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    models_loaded: bool
    qdrant_ok: bool
