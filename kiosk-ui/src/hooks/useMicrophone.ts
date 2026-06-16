"use client";

import { useState, useRef, useCallback } from "react";

export type MicrophoneStatus = "idle" | "ready" | "recording" | "error";

export function useMicrophone() {
  const [status, setStatus] = useState<MicrophoneStatus>("idle");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const pcmBuffersRef = useRef<Float32Array[]>([]);

  const startRecording = useCallback(async () => {
    try {
      setStatus("ready");
      setAudioBlob(null);
      setErrorMessage("");
      pcmBuffersRef.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      // Use 16000Hz sample rate for Whisper model
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;

      // 4096 buffer size, 1 input channel, 1 output channel
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        // Clone the input buffer array
        pcmBuffersRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
      
      setStatus("recording");
    } catch (err: any) {
      console.error("Microphone access failed:", err);
      setStatus("error");
      setErrorMessage(err.message || "Failed to access microphone");
    }
  }, []);

  const stopRecording = useCallback((): Blob | null => {
    if (status !== "recording") return null;

    // Disconnect and stop tracks
    if (processorRef.current && sourceRef.current) {
      sourceRef.current.disconnect();
      processorRef.current.disconnect();
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }

    // Merge PCM buffers
    const buffers = pcmBuffersRef.current;
    let totalLength = 0;
    for (const b of buffers) {
      totalLength += b.length;
    }

    const mergedSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const b of buffers) {
      mergedSamples.set(b, offset);
      offset += b.length;
    }

    // Encode to WAV (16kHz mono)
    const wavBlob = encodeWAV(mergedSamples, 16000);
    setAudioBlob(wavBlob);
    setStatus("idle");

    return wavBlob;
  }, [status]);

  return {
    status,
    audioBlob,
    errorMessage,
    startRecording,
    stopRecording,
  };
}

// ─── WAV Encoder Helper ──────────────────────────────────────────────────────

function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  /* RIFF identifier */
  writeString(view, 0, "RIFF");
  /* file length */
  view.setUint32(4, 36 + samples.length * 2, true);
  /* RIFF type */
  writeString(view, 8, "WAVE");
  /* format chunk identifier */
  writeString(view, 12, "fmt ");
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, "data");
  /* data chunk length */
  view.setUint32(40, samples.length * 2, true);

  floatTo16BitPCM(view, 44, samples);

  return new Blob([view], { type: "audio/wav" });
}

function floatTo16BitPCM(output: DataView, offset: number, input: Float32Array) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}
