// Decodes a QR code in a camera frame (qr-scan.ts), off the page's thread.
// The pairing code is dark on a white quiet zone (desktop/qr.ts), so inverted
// codes are not tried: that halves the cost.

import jsQR from "jsqr";

export interface QrRequest { data: Uint8ClampedArray; width: number; height: number }
export interface QrReply { text: string | null; ms: number }

self.onmessage = (e: MessageEvent<QrRequest>) => {
  const t0 = performance.now();
  const { data, width, height } = e.data;
  let text: string | null = null;
  try {
    text = jsQR(data, width, height, { inversionAttempts: "dontInvert" })?.data || null;
  } catch {
    text = null;
  }
  (self as unknown as Worker).postMessage({ text, ms: performance.now() - t0 } satisfies QrReply);
};
