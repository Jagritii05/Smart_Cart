"use client";

interface AisleSelectorProps {
  currentAisle: number | null;
  onChange: (aisle: number | null) => void;
}

export default function AisleSelector({ currentAisle, onChange }: AisleSelectorProps) {
  // Let's support Aisles 1 to 5 as defined in the dataset, plus "All Aisles"
  const aisles = [
    { value: null, label: "All Aisles" },
    { value: 1, label: "Aisle 1 — Snacks & Beverages" },
    { value: 2, label: "Aisle 2 — Pantry Essentials" },
    { value: 3, label: "Aisle 3 — Organic & Produce" },
    { value: 4, label: "Aisle 4 — Bakery & Dairy" },
    { value: 5, label: "Aisle 5 — Visual-Only Barcodes" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      <label className="section-label" style={{ marginBottom: "0" }}>
        Active Aisle Filter
      </label>
      <div style={{ position: "relative" }}>
        <select
          value={currentAisle === null ? "" : currentAisle.toString()}
          onChange={(e) => {
            const val = e.target.value;
            onChange(val === "" ? null : parseInt(val, 10));
          }}
          className="input"
          style={{
            appearance: "none",
            width: "100%",
            cursor: "pointer",
            paddingRight: "40px",
            fontSize: "0.875rem",
            background: "var(--bg-glass) url(\"data:image/svg+xml;utf8,<svg fill='white' height='24' viewBox='0 0 24 24' width='24' xmlns='http://www.w3.org/2000/svg'><path d='M7 10l5 5 5-5z'/><path d='M0 0h24v24H0z' fill='none'/></svg>\") no-repeat right 12px center",
          }}
        >
          {aisles.map((aisle) => (
            <option
              key={aisle.value ?? "all"}
              value={aisle.value ?? ""}
              style={{ background: "var(--bg-surface)", color: "var(--text-primary)" }}
            >
              {aisle.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
