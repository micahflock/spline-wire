// The live view's frame grab and detector, off the page's thread
// (docs/open-items.md, live view resolution; camera/grab.ts is the page's side).
//
// The page hands over the video's current frame as a VideoFrame (taking it
// costs the page nothing), with the region to draw (overlay/region.ts). Here
// it is drawn small, read back, turned gray and searched for the chain's
// fiducials: the work that, on the page's thread, took a 12 MP frame's
// conversion (~40 ms in headless Chromium) plus the detector.
//
// Messages: GrabRequest in (the frame transferred), GrabReply out. The frame
// is always closed here: a VideoFrame left open holds the camera's buffer.

import { detectBlobs, toGray } from "../overlay/blobs";
import { CanvasTrial, transformFor, type CanvasKind, type GrabReply, type GrabRequest } from "./grab";

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<GrabRequest>) => void) | null;
  postMessage(msg: GrabReply, transfer?: Transferable[]): void;
};

// One canvas of each kind (grab.ts, CanvasKind), tried in turn, the faster kept.
const canvases: Partial<Record<CanvasKind, { canvas: OffscreenCanvas; g: OffscreenCanvasRenderingContext2D }>> = {};
let trial: CanvasTrial | null = null;
let gray = new Uint8Array(0);

function workCanvas(kind: CanvasKind, width: number, height: number): OffscreenCanvasRenderingContext2D {
  let c = canvases[kind];
  if (!c || c.canvas.width < width || c.canvas.height < height) {
    // Grown, never shrunk: the regions change size from frame to frame.
    const canvas = new OffscreenCanvas(Math.max(c?.canvas.width ?? 0, width), Math.max(c?.canvas.height ?? 0, height));
    const g = canvas.getContext("2d", { willReadFrequently: kind === "cpu" });
    if (!g) throw new Error("no 2D context on an OffscreenCanvas");
    c = canvases[kind] = { canvas, g };
  }
  return c.g;
}

scope.onmessage = (e) => {
  const { id, frame, region: r, shape, turn } = e.data;
  try {
    const t0 = performance.now();
    trial ??= new CanvasTrial(e.data.canvas ?? null);
    const kind = trial.next();
    const g = workCanvas(kind, r.workWidth, r.workHeight);
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";            // area-like averaging, as on the page
    // Upright: the frame may be the sensor's way round, not the video's (grab.ts, Turn).
    // The whole frame under one transform, no source rectangle (grab.ts, transformFor).
    g.setTransform(...transformFor(r, turn ?? 0, frame.displayWidth, frame.displayHeight));
    g.drawImage(frame, 0, 0);
    g.setTransform(1, 0, 0, 1, 0, 0);
    frame.close();
    const n = r.workWidth * r.workHeight;
    if (gray.length < n) gray = new Uint8Array(n);
    const work = toGray(g.getImageData(0, 0, r.workWidth, r.workHeight).data, r.workWidth, r.workHeight, gray);
    const t1 = performance.now();
    trial.record(kind, t1 - t0);
    // The other kind's canvas goes once the choice is made.
    const other = canvases[kind === "gpu" ? "cpu" : "gpu"];
    if (trial.chosen === kind && other) {
      other.canvas.width = other.canvas.height = 0;
      delete canvases[kind === "gpu" ? "cpu" : "gpu"];
    }
    const canvas = trial.describe();
    if (e.data.gray) {
      // The check (grab.ts): the pixels themselves, to compare with the page's own grab.
      const copy = work.data.slice();
      scope.postMessage({ id, gray: copy, grabMs: t1 - t0, canvas }, [copy.buffer]);
      return;
    }
    const blobs = detectBlobs(work, r, shape);
    scope.postMessage({ id, blobs, grabMs: t1 - t0, blobMs: performance.now() - t1, canvas });
  } catch (err) {
    scope.postMessage({ id, error: err instanceof Error ? `${err.name}: ${err.message}` : String(err) });
  } finally {
    try {
      frame.close();
    } catch {
      /* closed already */
    }
  }
};
