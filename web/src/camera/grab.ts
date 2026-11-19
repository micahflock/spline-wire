// Where the live view grabs its frames: in a worker when the browser can, else
// on the page (docs/open-items.md, live view resolution, step 3).
//
// Measured in headless Chromium (no GPU) with a 12 MP camera: drawing a
// region of the video into a canvas and reading it back costs the page ~40 ms
// per new frame, the conversion of the camera's frame; createImageBitmap(video)
// costs the page as long (it copies the frame before it resolves); taking a
// VideoFrame of the video costs ~0 ms, and the worker can draw it
// (grab-worker.ts). So: a VideoFrame to the worker, which draws, reads back and
// finds the blobs, and the page only draws the overlay.
//
// That needs VideoFrame, OffscreenCanvas with a 2D context in a worker, and a
// VideoFrame of the same pixels the page sees: Chrome and Safari 16.4+ have
// all three, Firefox on Android may lack VideoFrame. So first a check: the
// page grabs a frame as before, and the worker draws the same region of the
// same frame; their gray copies must agree (same orientation, same pixels
// within a few gray levels). A plain picture cannot tell a turned or mirrored
// frame: the check waits for one with contrast (the chain's region has it). Any failure
// (no VideoFrame, another size, the check, an error or no answer from the
// worker) makes the page grab from then on, as it did before. ?grab=main in
// the page's address skips the worker, to compare.

import type { Shape } from "../overlay/blobs";
import type { Region } from "../overlay/region";
import type { Blob } from "../overlay/types";

export interface GrabRequest {
  id: number;
  frame: VideoFrame;
  region: Region;
  shape: Shape;
  /** Reply with the gray work frame instead of its blobs (the check). */
  gray?: boolean;
}

export interface GrabReply {
  id: number;
  blobs?: Blob[];
  gray?: Uint8Array;
  grabMs?: number;
  blobMs?: number;
  error?: string;
}

export interface GrabWorkerLike {
  postMessage(msg: GrabRequest, transfer: Transferable[]): void;
  terminate(): void;
  onmessage: ((e: MessageEvent<GrabReply>) => void) | null;
  onerror: ((e: ErrorEvent) => void) | null;
}

export type GrabMode = "check" | "worker" | "main";

export const GRAB = {
  /** The worker's copy and the page's may differ by this much, mean absolute gray level. */
  maxDifference: 6,
  /** A frame with less contrast than this (gray levels, standard deviation) cannot
   * show a turned or mirrored picture: the check waits for a busier one. */
  minContrast: 8,
  /** A worker that has not answered in this long is given up on, ms. */
  timeoutMs: 2000,
  /** After a check that could not tell, the page grabs this many frames before the next check. */
  recheckFrames: 15,
};

/** Why this browser cannot try the worker, or null if it can. */
export function whyNotWorker(search: string, g: object = globalThis): string | null {
  if (new URLSearchParams(search).get("grab") === "main") return "?grab=main";
  const has = (name: string) => typeof (g as Record<string, unknown>)[name] === "function";
  if (!has("VideoFrame")) return "no VideoFrame here";
  if (!has("OffscreenCanvas")) return "no OffscreenCanvas here";
  if (!has("Worker")) return "no workers here";
  return null;
}

export class FrameGrab {
  mode: GrabMode;
  /** Why the page grabs, when it does. */
  why: string;
  private worker: GrabWorkerLike | null = null;
  private next = 1;
  /** Frames the page grabs before the next check (after an unsure one). */
  private wait = 0;
  private pending: { id: number; resolve: (r: GrabReply) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> } | null = null;

  /** `blocked`: why the worker cannot be tried here (whyNotWorker), or null. */
  constructor(private readonly make: () => GrabWorkerLike, blocked: string | null, private readonly timeoutMs = GRAB.timeoutMs) {
    this.mode = blocked === null ? "check" : "main";
    this.why = blocked ?? "";
  }

  /** Where frames are grabbed, for the camera view's info line. */
  describe(): string {
    return this.mode === "worker" ? "grab in a worker"
      : this.mode === "check" ? "grab on the page (checking the worker)"
        : `grab on the page: ${this.why}`;
  }

  /** A frame is with the worker: the page waits for it rather than send another. */
  get busy(): boolean {
    return this.pending !== null;
  }

  /** Send a frame to the worker (it takes the frame, and closes it). Rejects if
   * the worker fails or does not answer; the page then grabs (fallBack). */
  run(frame: VideoFrame, region: Region, shape: Shape, gray = false): Promise<GrabReply> {
    const w = this.worker ?? this.start();
    const id = this.next++;
    return new Promise<GrabReply>((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending?.id !== id) return;
        this.pending = null;
        reject(new Error(`the frame worker did not answer in ${this.timeoutMs / 1000} s`));
      }, this.timeoutMs);
      this.pending = { id, resolve, reject, timer };
      try {
        w.postMessage({ id, frame, region, shape, gray }, [frame as unknown as Transferable]);
      } catch (e) {
        clearTimeout(timer);
        this.pending = null;
        try {
          frame.close();
        } catch {
          /* closed already */
        }
        reject(e as Error);
      }
    }).then((reply) => {
      if (reply.error !== undefined) throw new Error(reply.error);
      return reply;
    });
  }

  /** Should this frame go to the worker (as a check, or to be searched)? While
   * a check is put off, the page grabs. */
  wants(): boolean {
    if (this.mode === "worker") return true;
    if (this.mode === "main") return false;
    if (this.wait > 0) {
      this.wait--;
      return false;
    }
    return true;
  }

  /** The check's verdict (compareGrays): the worker grabs from now on, the page
   * does, or the check is tried again on a busier frame a little later. */
  checked(verdict: "same" | "different" | "unsure"): void {
    if (this.mode !== "check") return;
    if (verdict === "same") this.mode = "worker";
    else if (verdict === "different") this.fallBack("the worker's copy of a frame differs");
    else this.wait = GRAB.recheckFrames;
  }

  /** From now on the page grabs. */
  fallBack(why: string): void {
    this.mode = "main";
    this.why = why;
    this.close();
  }

  close(): void {
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.reject(new Error("the frame worker was closed"));
      this.pending = null;
    }
    this.worker?.terminate();
    this.worker = null;
  }

  private start(): GrabWorkerLike {
    const w = this.make();
    w.onmessage = (e) => {
      const p = this.pending;
      if (!p || p.id !== e.data.id) return;
      clearTimeout(p.timer);
      this.pending = null;
      p.resolve(e.data);
    };
    w.onerror = (e) => {
      const p = this.pending;
      this.pending = null;
      if (p) {
        clearTimeout(p.timer);
        p.reject(new Error(`the frame worker stopped: ${e.message}`));
      }
    };
    this.worker = w;
    return w;
  }
}

/** The check's verdict on the worker's gray copy of a frame against the page's:
 * "same", "different" (another orientation, or other pixels), or "unsure" (too
 * plain a picture to tell). */
export function compareGrays(page: Uint8Array, worker: Uint8Array): "same" | "different" | "unsure" {
  if (page.length !== worker.length || page.length === 0) return "different";
  let sum = 0, sq = 0, diff = 0;
  for (let i = 0; i < page.length; i++) {
    const v = page[i]!;
    sum += v;
    sq += v * v;
    diff += Math.abs(v - worker[i]!);
  }
  const n = page.length, mean = sum / n;
  if (Math.sqrt(Math.max(0, sq / n - mean * mean)) < GRAB.minContrast) return "unsure";
  return diff / n <= GRAB.maxDifference ? "same" : "different";
}
