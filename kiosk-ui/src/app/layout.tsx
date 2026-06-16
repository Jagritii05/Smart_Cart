import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Smart Cart AI Kiosk",
  description: "Premium offline-first retail kiosk assistant powered by Whisper, Qwen, and Gemma.",
};

export default function RootLayout({
  children,
  ...props
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-gradient-radial" style={{ minHeight: "100vh" }}>
        {children}
      </body>
    </html>
  );
}
