"use client";

import { useState, useCallback } from "react";
import KioskHeader from "../components/KioskHeader";
import CameraPanel from "../components/CameraPanel";
import MicrophonePanel from "../components/MicrophonePanel";
import AisleSelector from "../components/AisleSelector";
import TextSearch from "../components/TextSearch";
import ResultsPanel from "../components/ResultsPanel";
import CartSidebar from "../components/CartSidebar";
import { useWebSocket } from "../hooks/useWebSocket";
import { WS_URL, ProductResult, queryText, queryImage, queryAudio } from "../lib/api";

interface CartItem {
  product: ProductResult;
  quantity: number;
}

type TabMode = "text" | "camera" | "voice";

export default function KioskPage() {
  const { status: wsStatus, kioskState, sendAisleChange } = useWebSocket(WS_URL);
  
  const [results, setResults] = useState<ProductResult[]>([]);
  const [aiResponse, setAiResponse] = useState<string>("");
  const [isSearching, setIsSearching] = useState(false);
  const [cart, setCart] = useState<CartItem[]>([]);
  
  // New Layout States
  const [activeTab, setActiveTab] = useState<TabMode>("text");
  const [isCartOpen, setIsCartOpen] = useState(false);

  const activeAisle = kioskState.active_aisle;

  // Search by Text
  const handleTextSearch = useCallback(async (query: string) => {
    if (!query.trim()) return;
    setIsSearching(true);
    setAiResponse("");   // clear stale message immediately
    setResults([]);
    try {
      const res = await queryText(query);
      setResults(res.products);
      setAiResponse(res.response);
    } catch (err) {
      console.error("Text search error:", err);
      setAiResponse("Sorry, there was an issue processing your search. Please try again.");
    } finally {
      setIsSearching(false);
    }
  }, []);

  // Search by Image (Camera snapshot)
  const handleCameraScan = useCallback(async (blob: Blob) => {
    setIsSearching(true);
    setAiResponse("");   // clear stale message immediately
    setResults([]);
    try {
      const res = await queryImage(blob, activeAisle);
      setResults(res.products);
      setAiResponse(res.response);
    } catch (err) {
      console.error("Image scan error:", err);
      setAiResponse("Sorry, the scanner couldn't identify a barcode in that frame. Please try again.");
    } finally {
      setIsSearching(false);
    }
  }, [activeAisle]);

  // Search by Voice (Audio WAV recording)
  const handleVoiceQuery = useCallback(async (blob: Blob) => {
    setIsSearching(true);
    setAiResponse("");   // clear stale message immediately
    setResults([]);
    try {
      const res = await queryAudio(blob);
      setResults(res.products);
      setAiResponse(res.response);
    } catch (err) {
      console.error("Audio query error:", err);
      setAiResponse("Sorry, I couldn't hear or process that voice command. Please speak clearly and try again.");
    } finally {
      setIsSearching(false);
    }
  }, []);

  // Cart Management
  const handleAddToCart = useCallback((product: ProductResult) => {
    setCart((prevCart) => {
      const existing = prevCart.find((item) => (item.product.product_id ?? item.product.name) === (product.product_id ?? product.name));
      if (existing) {
        return prevCart.map((item) =>
          item.product.product_id === product.product_id || item.product.name === product.name
            ? { ...item, quantity: item.quantity + 1 }
            : item
        );
      }
      return [...prevCart, { product, quantity: 1 }];
    });
  }, []);

  const handleUpdateQuantity = useCallback((productId: string, delta: number) => {
    setCart((prevCart) => {
      return prevCart
        .map((item) => {
          if ((item.product.product_id ?? item.product.name) === productId) {
            const newQty = item.quantity + delta;
            return { ...item, quantity: newQty };
          }
          return item;
        })
        .filter((item) => item.quantity > 0);
    });
  }, []);

  const handleClearCart = useCallback(() => {
    setCart([]);
  }, []);

  const cartItemCount = cart.reduce((acc, item) => acc + item.quantity, 0);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        maxHeight: "100vh",
        overflow: "hidden",
        position: "relative",
      }}
    >
      <KioskHeader
        wsStatus={wsStatus}
        modelsLoaded={kioskState.models_loaded}
        qdrantOk={kioskState.qdrant_ok}
        cartItemCount={cartItemCount}
        onCartToggle={() => setIsCartOpen(true)}
      />

      <main
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          padding: "20px 24px",
          gap: "24px",
          overflow: "hidden",
          maxWidth: "1200px",
          margin: "0 auto",
          width: "100%",
        }}
      >
        {/* Tab Navigation */}
        <div className="tab-bar">
          <button 
            className={`tab-btn ${activeTab === "text" ? "active" : ""}`}
            onClick={() => setActiveTab("text")}
          >
            🔍 Text Search
          </button>
          <button 
            className={`tab-btn ${activeTab === "camera" ? "active" : ""}`}
            onClick={() => setActiveTab("camera")}
          >
            📷 Barcode Scanner
          </button>
          <button 
            className={`tab-btn ${activeTab === "voice" ? "active" : ""}`}
            onClick={() => setActiveTab("voice")}
          >
            🎤 Voice Assistant
          </button>
        </div>

        {/* Input Area (Swaps based on Tab) */}
        <div style={{ flexShrink: 0, minHeight: "150px" }}>
          {activeTab === "text" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <TextSearch onSearch={handleTextSearch} isSearching={isSearching} />
              <div style={{ maxWidth: "300px" }}>
                <AisleSelector currentAisle={activeAisle} onChange={sendAisleChange} />
              </div>
            </div>
          )}
          {activeTab === "camera" && (
            <div style={{ display: "flex", gap: "20px", alignItems: "flex-start" }}>
              <div style={{ width: "320px", flexShrink: 0 }}>
                <CameraPanel onScan={handleCameraScan} isSearching={isSearching} />
              </div>
              <div style={{ width: "300px" }}>
                <AisleSelector currentAisle={activeAisle} onChange={sendAisleChange} />
              </div>
            </div>
          )}
          {activeTab === "voice" && (
            <div>
              <MicrophonePanel onRecordComplete={handleVoiceQuery} isSearching={isSearching} />
            </div>
          )}
        </div>

        {/* Full-width Results Panel */}
        <div style={{ flex: 1, overflow: "hidden" }}>
          <ResultsPanel
            results={results}
            aiResponse={aiResponse}
            isSearching={isSearching}
            onAddToCart={handleAddToCart}
          />
        </div>
      </main>

      {/* Slide-out Cart Drawer Overlay */}
      <div 
        className={`drawer-overlay ${isCartOpen ? "open" : ""}`}
        onClick={() => setIsCartOpen(false)}
      >
        <div 
          className="drawer"
          onClick={(e) => e.stopPropagation()} // Prevent clicking inside from closing it
        >
          {/* Drawer Header */}
          <div style={{ 
            padding: "16px 24px", 
            borderBottom: "1px solid var(--border-subtle)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center"
          }}>
            <h2 style={{ fontSize: "1.2rem", fontWeight: 600 }}>Your Smart Cart</h2>
            <button 
              onClick={() => setIsCartOpen(false)}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--text-muted)",
                fontSize: "1.2rem",
                cursor: "pointer",
                padding: "4px"
              }}
            >
              ✕
            </button>
          </div>
          
          {/* Cart Contents */}
          <div style={{ flex: 1, overflow: "hidden" }}>
            <CartSidebar
              cart={cart}
              onUpdateQuantity={handleUpdateQuantity}
              onClear={handleClearCart}
            />
          </div>
        </div>
      </div>

    </div>
  );
}
