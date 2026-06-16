"use client";

import { useMicrophone } from "../hooks/useMicrophone";

interface MicrophonePanelProps {
  onRecordComplete: (blob: Blob) => Promise<void>;
  isSearching: boolean;
}

export default function MicrophonePanel({ onRecordComplete, isSearching }: MicrophonePanelProps) {
  const { status, errorMessage, startRecording, stopRecording } = useMicrophone();

  const handleToggleRecord = async () => {
    if (status === "recording") {
      const blob = stopRecording();
      if (blob) {
        await onRecordComplete(blob);
      }
    } else {
      await startRecording();
    }
  };

  const isRecording = status === "recording";

  return (
    <div
      className="glass-card"
      style={{
        display: "flex",
        flexDirection: "column",
        padding: "20px",
        gap: "16px",
        height: "100%",
        justifyContent: "space-between",
        position: "relative",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              backgroundColor: isRecording ? "var(--accent-rose)" : "var(--text-muted)",
              boxShadow: isRecording ? "0 0 10px var(--accent-rose)" : "none",
            }}
          />
          <span style={{ fontWeight: 600, fontSize: "0.95rem" }}>Voice Assistant</span>
        </div>
        <span className="badge badge-amber" style={{ fontSize: "0.65rem" }}>
          16kHz WAV Mono
        </span>
      </div>

      {/* Recording Display & Waveform */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "120px",
          position: "relative",
        }}
      >
        {isRecording ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "16px" }}>
            {/* Waveform bars */}
            <div style={{ display: "flex", alignItems: "center", gap: "4px", height: "40px" }}>
              {[...Array(9)].map((_, i) => {
                // Different delay for each bar to look natural
                const delay = `${i * 0.15}s`;
                return (
                  <div
                    key={i}
                    style={{
                      width: "4px",
                      backgroundColor: "var(--accent-primary)",
                      borderRadius: "var(--radius-full)",
                      animation: "waveform 1.2s ease-in-out infinite",
                      animationDelay: delay,
                      height: "8px",
                    }}
                  />
                );
              })}
            </div>
            <span style={{ fontSize: "0.85rem", color: "var(--accent-rose)", fontWeight: 500 }}>
              Listening closely... Tap button to search
            </span>
          </div>
        ) : isSearching ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
            <div className="spinner" style={{ width: "28px", height: "28px" }} />
            <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Processing voice query...
            </span>
          </div>
        ) : (
          <div style={{ textAlign: "center", display: "flex", flexDirection: "column", gap: "6px" }}>
            <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
              Ask a question like:
            </span>
            <span
              style={{
                fontSize: "0.95rem",
                color: "var(--text-primary)",
                fontStyle: "italic",
                fontWeight: 500,
              }}
            >
              &ldquo;Where can I find organic apples?&rdquo;
            </span>
          </div>
        )}
      </div>

      {errorMessage && (
        <div
          style={{
            padding: "8px 12px",
            borderRadius: "var(--radius-sm)",
            backgroundColor: "rgba(235, 94, 85, 0.1)",
            border: "1px solid rgba(235, 94, 85, 0.2)",
            color: "var(--accent-rose)",
            fontSize: "0.78rem",
            textAlign: "center",
          }}
        >
          {errorMessage}
        </div>
      )}

      {/* Button controls */}
      <button
        className={`btn ${isRecording ? "btn-danger" : "btn-primary"}`}
        style={{
          width: "100%",
          height: "48px",
          boxShadow: isRecording ? "0 0 20px rgba(235, 94, 85, 0.4)" : undefined,
        }}
        onClick={handleToggleRecord}
        disabled={isSearching}
      >
        {isRecording ? (
          <>
            <span>⏹️</span> Stop & Send
          </>
        ) : (
          <>
            <span>🎤</span> Hold to Speak
          </>
        )}
      </button>

      <style jsx global>{`
        @keyframes waveform {
          0%, 100% { height: 8px; }
          50% { height: 36px; }
        }
      `}</style>
    </div>
  );
}
