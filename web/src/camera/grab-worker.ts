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
import { drawFor, type GrabReply, type GrabRequest } from "./grab";

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<GrabRequest>) => void) | null;
  postMessage(msg: GrabReply, transfer?: Transferable[]): void;
};

let canvas: OffscreenCanvas | null = null;
let g: OffscreenCanvasRenderingContext2D | null = null;
let gray = new Uint8Array(0);

scope.onmessage = (e) => {
  const { id, frame, region: r, shape, turn } = e.data;
  try {
    const t0 = performance.now();
    if (!canvas || !g || canvas.width < r.workWidth || canvas.height < r.workHeight) {
      // Grown, never shrunk: the regions change size from frame to frame.
      canvas = new OffscreenCanvas(Math.max(canvas?.width ?? 0, r.workWidth), Math.max(canvas?.height ?? 0, r.workHeight));
      g = canvas.getContext("2d", { willReadFrequently: true });
      if (!g) throw new Error("no 2D context on an OffscreenCanvas");
    }
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";            // area-like averaging, as on the page
    // Upright: the frame may be the sensor's way round, not the video's (grab.ts, Turn).
    const d = drawFor(r, turn ?? 0, frame.displayWidth, frame.displayHeight);
    g.setTransform(...d.m);
    g.drawImage(frame, d.sx, d.sy, d.sw, d.sh, 0, 0, d.dw, d.dh);
    g.setTransform(1, 0, 0, 1, 0, 0);
    frame.close();
    const n = r.workWidth * r.workHeight;
    if (gray.length < n) gray = new Uint8Array(n);
    const work = toGray(g.getImageData(0, 0, r.workWidth, r.workHeight).data, r.workWidth, r.workHeight, gray);
    const t1 = performance.now();
    if (e.data.gray) {
      // The check (grab.ts): the pixels themselves, to compare with the page's own grab.
      const copy = work.data.slice();
      scope.postMessage({ id, gray: copy, grabMs: t1 - t0 }, [copy.buffer]);
      return;
    }
    const blobs = detectBlobs(work, r, shape);
    scope.postMessage({ id, blobs, grabMs: t1 - t0, blobMs: performance.now() - t1 });
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
