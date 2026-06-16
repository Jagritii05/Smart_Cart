"use client";

import { useEffect, useState, useCallback } from "react";
import KioskHeader from "../../components/KioskHeader";
import AisleSelector from "../../components/AisleSelector";
import ProductCard from "../../components/ProductCard";
import { useWebSocket } from "../../hooks/useWebSocket";
import { WS_URL, ProductResult, getProducts, updateStock } from "../../lib/api";

export default function AdminPage() {
  const { status: wsStatus, kioskState, sendAisleChange } = useWebSocket(WS_URL);
  
  const [products, setProducts] = useState<ProductResult[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updatingStockId, setUpdatingStockId] = useState<string | null>(null);

  const activeAisle = kioskState.active_aisle;

  // Fetch all products
  const fetchProducts = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await getProducts();
      setProducts(res);
    } catch (err: any) {
      console.error("Failed to load products:", err);
      setError(err.message || "Failed to load products from vector DB.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProducts();
  }, [fetchProducts]);

  // Handle stock toggling
  const handleToggleStock = useCallback(async (productId: string, currentStock: boolean) => {
    setUpdatingStockId(productId);
    try {
      const newStock = !currentStock;
      await updateStock(productId, newStock);
      
      // Update local state instantly
      setProducts((prevProducts) =>
        prevProducts.map((p) =>
          p.product_id === productId ? { ...p, stock_status: newStock } : p
        )
      );
    } catch (err) {
      console.error("Failed to update stock:", err);
      alert("Failed to update stock status. Please try again.");
    } finally {
      setUpdatingStockId(null);
    }
  }, []);

  // Filter products by aisle and search query
  const filteredProducts = products.filter((p) => {
    const matchesAisle = activeAisle === null || p.aisle_number === activeAisle;
    const matchesSearch =
      p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.brand.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesAisle && matchesSearch;
  });

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        maxHeight: "100vh",
        overflow: "hidden",
      }}
    >
      <KioskHeader
        wsStatus={wsStatus}
        modelsLoaded={kioskState.models_loaded}
        qdrantOk={kioskState.qdrant_ok}
      />

      <main
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          padding: "20px 24px",
          gap: "20px",
          overflow: "hidden",
        }}
      >
        {/* Page title and description */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h2 style={{ fontSize: "1.5rem", fontWeight: 700 }}>Admin Dashboard</h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Manage real-time inventory levels and active kiosk aisle locations.
            </p>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={fetchProducts} disabled={isLoading}>
            {isLoading ? "Refreshing..." : "🔄 Refresh Inventory"}
          </button>
        </div>

        {/* Workspace controls */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "300px 1fr",
            gap: "20px",
            flex: 1,
            overflow: "hidden",
          }}
        >
          {/* Controls Panel */}
          <div
            className="glass-card"
            style={{
              padding: "20px",
              display: "flex",
              flexDirection: "column",
              gap: "20px",
              height: "fit-content",
            }}
          >
            <h3 style={{ fontSize: "1rem", fontWeight: 600 }}>Active Aisle Control</h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: "1.4" }}>
              Setting the active aisle here will instantly update the barcode/visual search filters across all kiosk screens connected to this server.
            </p>
            <AisleSelector currentAisle={activeAisle} onChange={sendAisleChange} />
          </div>

          {/* Catalog Listing */}
          <div
            className="glass-card"
            style={{
              padding: "20px",
              display: "flex",
              flexDirection: "column",
              gap: "16px",
              overflow: "hidden",
            }}
          >
            {/* Search filter input */}
            <div style={{ display: "flex", gap: "12px", width: "100%" }}>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter catalog by product name, brand, or ingredients..."
                className="input"
                style={{ flex: 1 }}
              />
            </div>

            {/* Scrollable list */}
            <div style={{ flex: 1, overflowY: "auto", paddingRight: "4px" }}>
              {isLoading ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "200px",
                    gap: "12px",
                  }}
                >
                  <div className="spinner" style={{ width: "28px", height: "28px" }} />
                  <span style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
                    Reading current database state...
                  </span>
                </div>
              ) : error ? (
                <div
                  style={{
                    padding: "24px",
                    textAlign: "center",
                    color: "var(--accent-rose)",
                  }}
                >
                  ⚠️ Error: {error}
                </div>
              ) : filteredProducts.length > 0 ? (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
                    gap: "16px",
                  }}
                >
                  {filteredProducts.map((product) => (
                    <ProductCard
                      key={product.product_id}
                      product={product}
                      isAdmin={true}
                      onToggleStock={handleToggleStock}
                      isUpdatingStock={updatingStockId === product.product_id}
                    />
                  ))}
                </div>
              ) : (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "100%",
                    color: "var(--text-muted)",
                    padding: "40px",
                  }}
                >
                  No products matched the filter query.
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
