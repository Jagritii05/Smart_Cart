"use client";

import { ProductResult } from "../lib/api";

interface ProductCardProps {
  product: ProductResult;
  onAddToCart?: (product: ProductResult) => void;
  isAdmin?: boolean;
  onToggleStock?: (id: string, currentStock: boolean) => Promise<void>;
  isUpdatingStock?: boolean;
}

export default function ProductCard({
  product,
  onAddToCart,
  isAdmin = false,
  onToggleStock,
  isUpdatingStock = false,
}: ProductCardProps) {
  const isOutOfStock = !product.stock_status;

  // Render a visual thumbnail based on first letter of name
  const letter = product.name ? product.name.charAt(0).toUpperCase() : "?";

  // Match background gradient based on aisle to give a premium visual distinction
  const getThumbnailGradient = (aisle: number) => {
    switch (aisle) {
      case 1:
        return "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)";
      case 2:
        return "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)";
      case 3:
        return "linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)";
      case 4:
        return "linear-gradient(135deg, #fccd4d 0%, #fb9e3c 100%)";
      default:
        return "linear-gradient(135deg, #cfd9df 0%, #e2ebf0 100%)";
    }
  };

  return (
    <div
      className="glass-card"
      style={{
        display: "flex",
        flexDirection: "column",
        padding: "16px",
        borderRadius: "var(--radius-lg)",
        gap: "12px",
        position: "relative",
        animation: "fadeInUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) both",
        opacity: isOutOfStock && !isAdmin ? 0.6 : 1,
        transition: "transform var(--transition-fast), opacity var(--transition-fast)",
      }}
    >
      {/* Top row: Score + Aisle Badge */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span
          className="badge badge-green"
          style={{
            fontSize: "0.68rem",
            textTransform: "none",
            backgroundColor: "rgba(158, 255, 158, 0.08)",
          }}
        >
          Match: {(product.score * 100).toFixed(0)}%
        </span>
        <span
          className={product.aisle_number === 5 ? "badge badge-rose" : "badge badge-indigo"}
          style={{ fontSize: "0.68rem" }}
        >
          Aisle {product.aisle_number}
        </span>
      </div>

      {/* Product visual + info */}
      <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
        {/* Item Circle Thumbnail */}
        <div
          style={{
            width: "48px",
            height: "48px",
            borderRadius: "var(--radius-md)",
            background: getThumbnailGradient(product.aisle_number),
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 800,
            fontSize: "1.2rem",
            color: "var(--bg-base)",
            boxShadow: "0 4px 10px rgba(0, 0, 0, 0.2)",
            flexShrink: 0,
          }}
        >
          {letter}
        </div>

        {/* Text Details */}
        <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <span
            style={{
              fontSize: "0.72rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "var(--text-muted)",
              fontWeight: 700,
            }}
          >
            {product.brand}
          </span>
          <h4
            style={{
              fontSize: "0.95rem",
              fontWeight: 600,
              color: "var(--text-primary)",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              marginTop: "2px",
            }}
            title={product.name}
          >
            {product.name}
          </h4>
          <span
            style={{
              fontSize: "1.1rem",
              fontWeight: 700,
              color: "var(--text-primary)",
              marginTop: "4px",
            }}
          >
            ${product.price.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Description */}
      <p
        style={{
          fontSize: "0.8rem",
          color: "var(--text-secondary)",
          lineHeight: "1.4",
          display: "-webkit-box",
          WebkitLineClamp: 2,
          WebkitBoxOrient: "vertical",
          overflow: "hidden",
          textOverflow: "ellipsis",
          height: "36px",
        }}
      >
        {product.description}
      </p>

      {/* Tags */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px", minHeight: "20px" }}>
        {product.tags && product.tags.map((tag) => (
          <span
            key={tag}
            className="badge badge-muted"
            style={{
              fontSize: "0.62rem",
              padding: "2px 6px",
              textTransform: "capitalize",
            }}
          >
            {tag}
          </span>
        ))}
      </div>

      <div className="divider" style={{ margin: "4px 0" }} />

      {/* Bottom actions */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px" }}>
        {isAdmin ? (
          <>
            <span
              className={product.stock_status ? "badge badge-green" : "badge badge-rose"}
              style={{ fontSize: "0.72rem" }}
            >
              {product.stock_status ? "In Stock" : "Out of Stock"}
            </span>
            <button
              className={`btn btn-sm ${product.stock_status ? "btn-danger" : "btn-success"}`}
              style={{ flex: 1 }}
              onClick={() => onToggleStock?.(product.product_id ?? "", product.stock_status)}
              disabled={isUpdatingStock}
            >
              {isUpdatingStock ? "Updating..." : product.stock_status ? "Set Out" : "Set In"}
            </button>
          </>
        ) : (
          <>
            <span
              className={product.stock_status ? "badge badge-green" : "badge badge-rose"}
              style={{ fontSize: "0.72rem" }}
            >
              {product.stock_status ? "Available" : "Sold Out"}
            </span>
            {product.stock_status ? (
              <button
                className="btn btn-secondary btn-sm"
                style={{ flex: 1, borderColor: "var(--accent-primary-glow)" }}
                onClick={() => onAddToCart?.(product)}
              >
                <span>🛒</span> Add to Cart
              </button>
            ) : (
              <button
                className="btn btn-secondary btn-sm"
                style={{ flex: 1 }}
                disabled
              >
                Unavailable
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
