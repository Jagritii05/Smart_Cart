"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export type WsStatus = "connecting" | "connected" | "disconnected";

export interface KioskState {
  models_loaded: boolean;
  qdrant_ok: boolean;
  active_aisle: number | null;
}

interface WsMessage {
  event: string;
  [key: string]: unknown;
}

export function useWebSocket(wsUrl: string) {
  const [status, setStatus] = useState<WsStatus>("connecting");
  const [kioskState, setKioskState] = useState<KioskState>({
    models_loaded: false,
    qdrant_ok: false,
    active_aisle: null,
  });
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    setStatus("connecting");

    ws.onopen = () => {
      setStatus("connected");
      // Start keepalive ping every 25 s
      const ping = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ event: "ping" }));
        } else {
          clearInterval(ping);
        }
      }, 25000);
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string) as WsMessage;
        switch (msg.event) {
          case "state_snapshot":
            setKioskState((prev) => ({
              ...prev,
              models_loaded: msg.models_loaded as boolean,
              qdrant_ok: msg.qdrant_ok as boolean,
            }));
            break;
          case "aisle_change":
            setKioskState((prev) => ({
              ...prev,
              active_aisle: (msg.aisle_number as number) ?? null,
            }));
            break;
          case "stock_update":
            // Handled by the stock_status field on individual product cards
            break;
          default:
            break;
        }
      } catch {
        /* ignore parse errors */
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [wsUrl]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendAisleChange = useCallback((aisleNumber: number | null) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ event: "aisle_change", aisle_number: aisleNumber })
      );
    }
    setKioskState((prev) => ({ ...prev, active_aisle: aisleNumber }));
  }, []);

  return { status, kioskState, sendAisleChange };
}
