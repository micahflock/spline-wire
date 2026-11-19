// Looks for a QR code in the live camera's frames, beside the chain overlay:
// so the desktop page's pairing code can be scanned without leaving the camera.
//
// Only while the overlay tracks no chain (nobody points at a chain and a code at
// once), and a few times a second, not every frame: the whole frame is drawn small (as
// camera-view.ts draws it: drawImage's source rectangle is unreliable on iPhone)
// and decoded in a worker, one frame at a time. A code at arm's length from a
// screen spans a fifth of the frame or more, 5 px a module or more at this size.

import type { QrReply, QrRequest } from "./qr-worker";

/** The longer side of the frame the decoder reads. */
export const QR_SIZE = 1024;
/** Between frames handed to the decoder. */
export const QR_GAP_MS = 400;

export class QrScan {
  /** Decoding time of the last frame (ms), for the camera's stats. */
  lastMs = 0;
  /** The page's own time for the last frame (draw and read back, ms). */
  grabMs = 0;
  frames = 0;
  private worker: Worker | null = null;
  private busy = false;
  private last = -Infinity;
  private readonly seen = new Set<string>();
  private readonly canvas = document.createElement("canvas");
  private readonly g = this.canvas.getContext("2d", { willReadFrequently: true });

  /** `found` is called once per distinct code while this scan lives. */
  constructor(private readonly found: (text: string) => void) {}

  /** Called for each video frame; does nothing most of the time. Returns
   * whether it took this frame (the page's time it spent is grabMs). */
  frame(video: HTMLVideoElement, now: number): boolean {
    if (this.busy || now - this.last < QR_GAP_MS || !this.g || !video.videoWidth) return false;
    this.last = now;
    const t0 = performance.now();
    const scale = Math.min(1, QR_SIZE / Math.max(video.videoWidth, video.videoHeight));
    const w = Math.round(video.videoWidth * scale), h = Math.round(video.videoHeight * scale);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.g.drawImage(video, 0, 0, w, h);
    const { data } = this.g.getImageData(0, 0, w, h);
    this.grabMs = performance.now() - t0;
    if (!this.worker) {
      this.worker = new Worker(new URL("./qr-worker.ts", import.meta.url), { type: "module" });
      this.worker.onmessage = (e: MessageEvent<QrReply>) => {
        this.busy = false;
        this.frames++;
        this.lastMs = e.data.ms;
        const text = e.data.text;
        if (text && !this.seen.has(text)) {
          this.seen.add(text);
          this.found(text);
        }
      };
      this.worker.onerror = () => this.close();    // no worker: no scanning, the camera goes on
    }
    this.busy = true;
    this.worker.postMessage({ data, width: w, height: h } satisfies QrRequest, [data.buffer]);
    return true;
  }

  close(): void {
    this.worker?.terminate();
    this.worker = null;
    this.busy = true;                                // a closed scan stays closed
    this.canvas.width = this.canvas.height = 0;
  }
}
