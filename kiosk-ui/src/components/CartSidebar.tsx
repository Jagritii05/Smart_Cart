"use client";

import { ProductResult } from "../lib/api";

interface CartItem {
  product: ProductResult;
  quantity: number;
}

interface CartSidebarProps {
  cart: CartItem[];
  onUpdateQuantity: (productId: string, delta: number) => void;
  onClear: () => void;
}

export default function CartSidebar({ cart, onUpdateQuantity, onClear }: CartSidebarProps) {
  const total = cart.reduce((sum, item) => sum + item.product.price * item.quantity, 0);

  return (
    <div
      className="glass-card"
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        padding: "20px",
        gap: "16px",
        borderRadius: "var(--radius-lg)",
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "1.2rem" }}>🛒</span>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Your Smart Cart</h3>
        </div>
        {cart.length > 0 && (
          <button
            onClick={onClear}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--accent-rose)",
              fontSize: "0.8rem",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Clear All
          </button>
        )}
      </div>

      <div className="divider" style={{ margin: 0 }} />

      {/* Cart List */}
      <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "12px" }}>
        {cart.length > 0 ? (
          cart.map((item) => (
            <div
              key={item.product.product_id ?? item.product.name}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "10px",
                background: "var(--bg-glass)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-md)",
                animation: "fadeIn 0.2s ease-out",
              }}
            >
              <div style={{ display: "flex", flexDirection: "column", overflow: "hidden", flex: 1 }}>
                <span
                  style={{
                    fontSize: "0.875rem",
                    fontWeight: 600,
                    color: "var(--text-primary)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                  title={item.product.name}
                >
                  {item.product.name}
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "2px" }}>
                  ${item.product.price.toFixed(2)} each
                </span>
              </div>

              {/* Quantity Controls */}
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginLeft: "12px" }}>
                <button
                  className="btn btn-secondary btn-icon"
                  style={{
                    width: "28px",
                    height: "28px",
                    padding: 0,
                    minWidth: "28px",
                    fontSize: "0.75rem",
                    borderRadius: "var(--radius-sm)",
                  }}
                  onClick={() => onUpdateQuantity(item.product.product_id ?? "", -1)}
                >
                  -
                </button>
                <span style={{ fontSize: "0.875rem", fontWeight: 700, minWidth: "16px", textAlign: "center" }}>
                  {item.quantity}
                </span>
                <button
                  className="btn btn-secondary btn-icon"
                  style={{
                    width: "28px",
                    height: "28px",
                    padding: 0,
                    minWidth: "28px",
                    fontSize: "0.75rem",
                    borderRadius: "var(--radius-sm)",
                  }}
                  onClick={() => onUpdateQuantity(item.product.product_id ?? "", 1)}
                >
                  +
                </button>
              </div>
            </div>
          ))
        ) : (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              color: "var(--text-muted)",
              gap: "10px",
              padding: "24px 0",
              textAlign: "center",
            }}
          >
            <span style={{ fontSize: "1.8rem" }}>🛒</span>
            <span style={{ fontSize: "0.85rem", maxWidth: "200px" }}>
              Your cart is empty. Scan products or select from catalog to begin checkout.
            </span>
          </div>
        )}
      </div>

      <div className="divider" style={{ margin: 0 }} />

      {/* Summary / Total */}
      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>Est. Total:</span>
          <span style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text-primary)" }}>
            ${total.toFixed(2)}
          </span>
        </div>

        <button
          className="btn btn-primary btn-lg"
          style={{
            width: "100%",
            height: "48px",
            background: "linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))",
            color: "white",
            border: "none",
          }}
          disabled={cart.length === 0}
          onClick={() => {
            alert(`Kiosk checkout complete! Simulated payment of $${total.toFixed(2)} successful.`);
            onClear();
          }}
        >
          Checkout Session
        </button>
      </div>
    </div>
  );
}
