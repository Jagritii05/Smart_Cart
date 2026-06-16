"use client";

import { useState, useEffect, useRef } from "react";

interface TextSearchProps {
  onSearch: (query: string) => void;
  isSearching: boolean;
}

export default function TextSearch({ onSearch, isSearching }: TextSearchProps) {
  const [query, setQuery] = useState("");
  const initialMount = useRef(true);

  // Auto-search (debounce) removed to prevent high latency with local Gemma model.
  // Search only fires on form submit.

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="glass-card"
      style={{
        display: "flex",
        alignItems: "center",
        padding: "8px 12px",
        gap: "12px",
        width: "100%",
      }}
    >
      <div style={{ fontSize: "1.2rem", color: "var(--text-muted)", paddingLeft: "8px" }}>🔍</div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Type a product name, category, or customer request (e.g. 'high protein cereal')..."
        style={{
          flex: 1,
          border: "none",
          background: "transparent",
          color: "var(--text-primary)",
          fontSize: "1rem",
          outline: "none",
          height: "40px",
        }}
      />
      {query && (
        <button
          type="button"
          onClick={() => setQuery("")}
          style={{
            background: "transparent",
            border: "none",
            color: "var(--text-muted)",
            fontSize: "1.1rem",
            padding: "4px 8px",
            cursor: "pointer",
          }}
          title="Clear search"
        >
          ✕
        </button>
      )}
      <button
        type="submit"
        className="btn btn-primary btn-sm"
        disabled={isSearching}
        style={{ height: "40px", padding: "0 20px" }}
      >
        {isSearching ? <div className="spinner" style={{ width: "16px", height: "16px", borderTopColor: "white" }} /> : "Search"}
      </button>
    </form>
  );
}
