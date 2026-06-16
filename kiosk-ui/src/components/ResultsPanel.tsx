"use client";

import { ProductResult } from "../lib/api";
import ProductCard from "./ProductCard";

interface ResultsPanelProps {
  results: ProductResult[];
  aiResponse: string;
  isSearching: boolean;
  onAddToCart?: (product: ProductResult) => void;
  isAdmin?: boolean;
  onToggleStock?: (id: string, currentStock: boolean) => Promise<void>;
  updatingStockId?: string | null;
}

export default function ResultsPanel({
  results,
  aiResponse,
  isSearching,
  onAddToCart,
  isAdmin = false,
  onToggleStock,
  updatingStockId = null,
}: ResultsPanelProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px", height: "100%" }}>
      {/* AI Assistant Explanation Box */}
      {(aiResponse || isSearching) && (
        <div
          className="glass-card"
          style={{
            padding: "20px",
            borderRadius: "var(--radius-lg)",
            borderLeft: "4px solid var(--accent-primary)",
            background: "rgba(99, 102, 241, 0.04)",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            animation: "fadeInUp 0.3s ease-out",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div
              style={{
                width: "24px",
                height: "24px",
                borderRadius: "50%",
                background: "var(--bg-elevated)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "0.8rem",
              }}
            >
              🤖
            </div>
            <span
              style={{
                fontSize: "0.78rem",
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "var(--text-secondary)",
              }}
            >
              Gemma Retail Agent
            </span>
          </div>

          {isSearching && !aiResponse ? (
            <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "8px 0" }}>
              <div className="spinner" style={{ width: "18px", height: "18px" }} />
              <span style={{ fontSize: "0.95rem", color: "var(--text-secondary)", fontStyle: "italic" }}>
                Synthesizing products and thinking of recommendation...
              </span>
            </div>
          ) : (
            <p
              style={{
                fontSize: "0.98rem",
                color: "var(--text-primary)",
                lineHeight: "1.5",
                whiteSpace: "pre-line",
              }}
            >
              {aiResponse}
            </p>
          )}
        </div>
      )}

      {/* Results Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>
          {isSearching ? "Searching catalog..." : `Catalog Matches (${results.length})`}
        </h3>
        {isSearching && (
          <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
            Please wait...
          </span>
        )}
      </div>

      {/* Products Grid */}
      <div style={{ flex: 1, overflowY: "auto", paddingRight: "4px", minHeight: "280px" }}>
        {isSearching && results.length === 0 ? (
          /* Skeleton Loader Grid */
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
              gap: "16px",
            }}
          >
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="glass-card"
                style={{
                  height: "220px",
                  padding: "16px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "16px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <div className="skeleton" style={{ width: "70px", height: "18px" }} />
                  <div className="skeleton" style={{ width: "60px", height: "18px" }} />
                </div>
                <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                  <div className="skeleton" style={{ width: "48px", height: "48px", borderRadius: "12px" }} />
                  <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "6px" }}>
                    <div className="skeleton" style={{ width: "40%", height: "12px" }} />
                    <div className="skeleton" style={{ width: "80%", height: "16px" }} />
                    <div className="skeleton" style={{ width: "30%", height: "16px" }} />
                  </div>
                </div>
                <div className="skeleton" style={{ width: "100%", height: "32px" }} />
                <div className="skeleton" style={{ width: "100%", height: "24px" }} />
              </div>
            ))}
          </div>
        ) : results.length > 0 ? (
          /* Actual Results Grid */
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
              gap: "16px",
            }}
          >
            {results.map((product, index) => (
              <div key={product.product_id ?? index} style={{ animationDelay: `${index * 0.05}s` }}>
                <ProductCard
                  product={product}
                  onAddToCart={onAddToCart}
                  isAdmin={isAdmin}
                  onToggleStock={onToggleStock}
                  isUpdatingStock={updatingStockId === (product.product_id ?? "")}
                />
              </div>
            ))}
          </div>
        ) : (
          /* Empty State */
          <div
            className="glass-card"
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "240px",
              padding: "24px",
              textAlign: "center",
              color: "var(--text-muted)",
              borderStyle: "dashed",
            }}
          >
            <span style={{ fontSize: "2rem", marginBottom: "12px" }}>📦</span>
            <h4 style={{ color: "var(--text-primary)", fontWeight: 600, marginBottom: "4px" }}>
              No items matching criteria
            </h4>
            <p style={{ fontSize: "0.85rem", maxWidth: "300px" }}>
              Try speaking or typing a product query, or scan an item label to populate matching items here.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
