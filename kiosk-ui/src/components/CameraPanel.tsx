"use client";

import { useEffect, useState } from "react";
import { useCamera } from "../hooks/useCamera";

interface CameraPanelProps {
  onScan: (blob: Blob) => Promise<void>;
  isSearching: boolean;
}

export default function CameraPanel({ onScan, isSearching }: CameraPanelProps) {
  const { videoRef, status, error, startCamera, stopCamera, captureSnapshot } = useCamera();
  const [flash, setFlash] = useState(false);

  // Automatically start camera on mount, stop on unmount
  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [startCamera, stopCamera]);

  const handleScan = async () => {
    if (status !== "active") return;
    const blob = captureSnapshot();
    if (!blob) return;

    // Trigger visual flash
    setFlash(true);
    setTimeout(() => setFlash(false), 200);

    await onScan(blob);
  };

  return (
    <div
      className="glass-card"
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
        position: "relative",
      }}
    >
      {/* Visual Flash effect */}
      {flash && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "white",
            opacity: 0.8,
            zIndex: 10,
            pointerEvents: "none",
            transition: "opacity 0.2s ease-out",
          }}
        />
      )}

      {/* Header */}
      <div
        style={{
          padding: "16px 20px",
          borderBottom: "1px solid var(--border-subtle)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              backgroundColor: status === "active" ? "var(--accent-secondary)" : "var(--accent-amber)",
            }}
          />
          <span style={{ fontWeight: 600, fontSize: "0.95rem" }}>Live Vision Scanner</span>
        </div>
        <span className="badge badge-indigo" style={{ fontSize: "0.65rem" }}>
          WebRTC
        </span>
      </div>

      {/* Stream Area */}
      <div
        style={{
          flex: 1,
          backgroundColor: "#03060b",
          position: "relative",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
          minHeight: "240px",
        }}
      >
        {status === "active" ? (
          <div style={{ width: "100%", height: "100%", position: "relative" }}>
            <video
              ref={videoRef}
              muted
              playsInline
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                transform: "scaleX(-1)", // Mirror effect for kiosk
              }}
            />

            {/* Futuristic Scanning Overlay Target */}
            <div
              style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: "220px",
                height: "220px",
                border: "2px dashed rgba(255, 255, 255, 0.25)",
                borderRadius: "var(--radius-md)",
                pointerEvents: "none",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {/* Corner brackets */}
              <div
                style={{
                  position: "absolute",
                  top: "-4px",
                  left: "-4px",
                  width: "24px",
                  height: "24px",
                  borderTop: "3px solid var(--accent-secondary)",
                  borderLeft: "3px solid var(--accent-secondary)",
                  borderRadius: "4px 0 0 0",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  top: "-4px",
                  right: "-4px",
                  width: "24px",
                  height: "24px",
                  borderTop: "3px solid var(--accent-secondary)",
                  borderRight: "3px solid var(--accent-secondary)",
                  borderRadius: "0 4px 0 0",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  bottom: "-4px",
                  left: "-4px",
                  width: "24px",
                  height: "24px",
                  borderBottom: "3px solid var(--accent-secondary)",
                  borderLeft: "3px solid var(--accent-secondary)",
                  borderRadius: "0 0 0 4px",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  bottom: "-4px",
                  right: "-4px",
                  width: "24px",
                  height: "24px",
                  borderBottom: "3px solid var(--accent-secondary)",
                  borderRight: "3px solid var(--accent-secondary)",
                  borderRadius: "0 0 4px 0",
                }}
              />

              {/* Scanning red laser line */}
              <div
                style={{
                  position: "absolute",
                  left: "6%",
                  width: "88%",
                  height: "2px",
                  backgroundColor: "var(--accent-rose)",
                  boxShadow: "0 0 10px var(--accent-rose), 0 0 20px var(--accent-rose)",
                  animation: "laserMove 3s infinite linear",
                }}
              />
            </div>
            
            <style jsx global>{`
              @keyframes laserMove {
                0% { top: 10%; }
                50% { top: 90%; }
                100% { top: 10%; }
              }
            `}</style>
          </div>
        ) : status === "loading" ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
            <div className="spinner" style={{ width: "32px", height: "32px" }} />
            <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Initializing WebRTC stream...
            </span>
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "16px",
              padding: "24px",
              textAlign: "center",
            }}
          >
            <div
              style={{
                width: "48px",
                height: "48px",
                borderRadius: "50%",
                background: "rgba(235, 94, 85, 0.1)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--accent-rose)",
              }}
            >
              📷
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <span style={{ fontSize: "0.95rem", fontWeight: 600 }}>Camera Offline</span>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", maxWidth: "260px" }}>
                {error || "Kiosk camera feed is currently disabled or needs permissions."}
              </span>
            </div>
            <button className="btn btn-secondary btn-sm" onClick={startCamera}>
              Enable Camera
            </button>
          </div>
        )}
      </div>

      {/* Footer controls */}
      <div
        style={{
          padding: "16px 20px",
          borderTop: "1px solid var(--border-subtle)",
          display: "flex",
          gap: "12px",
          alignItems: "center",
          background: "var(--bg-glass)",
        }}
      >
        <button
          className="btn btn-primary btn-lg"
          style={{ flex: 1, height: "48px" }}
          onClick={handleScan}
          disabled={status !== "active" || isSearching}
        >
          {isSearching ? (
            <>
              <div className="spinner" style={{ borderTopColor: "white", width: "16px", height: "16px" }} />
              Scanning...
            </>
          ) : (
            <>
              <span>📷</span>
              Scan Barcode / Item
            </>
          )}
        </button>
        {status === "active" && (
          <button
            className="btn btn-secondary btn-icon"
            onClick={stopCamera}
            title="Pause camera"
            style={{ width: "48px", height: "48px" }}
          >
            ⏹️
          </button>
        )}
      </div>
    </div>
  );
}
