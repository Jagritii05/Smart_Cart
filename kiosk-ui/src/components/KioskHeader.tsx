"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { WsStatus } from "../hooks/useWebSocket";

interface KioskHeaderProps {
  wsStatus: WsStatus;
  modelsLoaded: boolean;
  qdrantOk: boolean;
  cartItemCount: number;
  onCartToggle: () => void;
}

export default function KioskHeader({ wsStatus, modelsLoaded, qdrantOk, cartItemCount, onCartToggle }: KioskHeaderProps) {
  const pathname = usePathname();
  const isAdmin = pathname === "/admin";

  const getWsStatusLabel = () => {
    switch (wsStatus) {
      case "connected":
        return "Connected";
      case "connecting":
        return "Connecting";
      case "disconnected":
        return "Offline";
    }
  };

  const getWsStatusDotClass = () => {
    switch (wsStatus) {
      case "connected":
        return "status-dot online";
      case "connecting":
        return "status-dot loading";
      case "disconnected":
        return "status-dot offline";
    }
  };

  return (
    <header
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "16px 24px",
        height: "80px",
        borderBottom: "1px solid var(--border-subtle)",
        background: "var(--bg-surface)",
        zIndex: 50,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "var(--radius-sm)",
              background: "var(--text-primary)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: 800,
              fontSize: "1.2rem",
              color: "var(--bg-base)",
            }}
          >
            S
          </div>
          <span style={{ fontSize: "1.25rem", fontWeight: 700, letterSpacing: "-0.02em" }}>
            Smart<span style={{ color: "var(--accent-secondary)" }}>Cart</span> Kiosk
          </span>
        </Link>
        <span className="badge badge-muted" style={{ fontSize: "0.68rem" }}>
          Offline-First AI
        </span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        {/* WebSocket Connection indicator */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            background: "var(--bg-glass)",
            padding: "6px 12px",
            borderRadius: "var(--radius-full)",
            border: "1px solid var(--border-subtle)",
            fontSize: "0.85rem",
          }}
        >
          <span className={getWsStatusDotClass()} />
          <span style={{ color: "var(--text-secondary)" }}>Server:</span>
          <span style={{ fontWeight: 600 }}>{getWsStatusLabel()}</span>
        </div>

        {/* AI Models indicator */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            background: "var(--bg-glass)",
            padding: "6px 12px",
            borderRadius: "var(--radius-full)",
            border: "1px solid var(--border-subtle)",
            fontSize: "0.85rem",
          }}
        >
          <span
            className={
              modelsLoaded
                ? "status-dot online"
                : qdrantOk
                ? "status-dot loading"
                : "status-dot offline"
            }
          />
          <span style={{ color: "var(--text-secondary)" }}>AI Core:</span>
          <span style={{ fontWeight: 600 }}>
            {modelsLoaded ? "Ready" : qdrantOk ? "Loading..." : "Offline"}
          </span>
        </div>

        <div className="divider" style={{ width: "1px", height: "24px", margin: "0 8px" }} />

        {!isAdmin && (
          <button onClick={onCartToggle} className="btn btn-primary btn-sm">
            🛒 Cart ({cartItemCount})
          </button>
        )}

        {isAdmin ? (
          <Link href="/" className="btn btn-secondary btn-sm">
            Kiosk Screen
          </Link>
        ) : (
          <Link href="/admin" className="btn btn-secondary btn-sm">
            Admin Panel
          </Link>
        )}
      </div>
    </header>
  );
}
