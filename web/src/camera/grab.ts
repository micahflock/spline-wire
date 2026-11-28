// Where the live view grabs its frames: in a worker when the browser can, else
// on the page (experiments/live_view/README.md, Findings: the grab in a worker).
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
// same frame; their gray copies must correlate (the same picture the same
// way round; a little lighter or darker is allowed). A plain picture cannot tell a turned or mirrored
// frame: the check waits for one with contrast (the chain's region has it). Any failure
// (no VideoFrame, another size, the check, an error or no answer from the
// worker) makes the page grab from then on, as it did before. ?grab=main in
// the page's address skips the worker, to compare.

import type { Shape } from "../overlay/blobs";
import type { Region } from "../overlay/region";
import type { Blob } from "../overlay/types";

/** How far the video, as shown, is turned from the VideoFrame of it, clockwise
 * in degrees. Safari on iPhone gives a portrait video's frames in the sensor's
 * landscape (VideoFrame 4032×3024 for a 3024×4032 video, owner's iPhone,
 * 2026-11-19), with no rotation said: 90 or 270, the check tells which. */
export type Turn = 0 | 90 | 180 | 270;

export interface GrabRequest {
  id: number;
  frame: VideoFrame;
  region: Region;
  shape: Shape;
  turn: Turn;
  /** Reply with the gray work frame instead of its blobs (the check). */
  gray?: boolean;
  /** Draw into this kind of canvas only (?canvas= in the page's address), not the faster one. */
  canvas?: CanvasKind;
}

export interface GrabReply {
  id: number;
  blobs?: Blob[];
  gray?: Uint8Array;
  grabMs?: number;
  blobMs?: number;
  /** Which canvas the worker drew into, and its trial (CanvasTrial.describe). */
  canvas?: string;
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
  /** The worker's copy and the page's must correlate at least this well (Pearson
   * r). On the owner's iPhone (Safari, 2026-11-19) the right turn gave r 1.00 at
   * a mean 8 gray levels lighter or darker, the wrong ones r -0.08 and 0.02: the
   * worker's colour conversion differs a little from the page's, which a fixed
   * gray-level limit took for another picture. The detector thresholds each pixel
   * against its own neighbourhood, so an even shift of a few levels is nothing to it. */
  minCorrelation: 0.95,
  /** A frame with less contrast than this (gray levels, standard deviation) cannot
   * show a turned or mirrored picture: the check waits for a busier one. */
  minContrast: 8,
  /** A worker that has not answered in this long is given up on, ms. */
  timeoutMs: 2000,
  /** After a check that could not tell, the page grabs this many frames before the next check. */
  recheckFrames: 15,
};

/** The worker's work canvas: "gpu" (a plain 2D canvas: the browser may scale
 * the frame on the GPU and read back only the small copy) or "cpu"
 * (willReadFrequently: kept in memory, so a 12 MP frame may be copied out of
 * the GPU and scaled on the CPU first). Which is cheaper depends on the
 * browser; the worker drew into a "cpu" canvas only until 2026-11-21, ~48 ms a
 * frame on the owner's iPhone (Safari, build 22). */
export type CanvasKind = "gpu" | "cpu";

/** The worker's choice of canvas: the first frames take turns between the two
 * kinds (TRIAL each), then the one with the lower median time is kept. */
export class CanvasTrial {
  static readonly TRIAL = 6;
  private readonly times: Record<CanvasKind, number[]> = { gpu: [], cpu: [] };
  chosen: CanvasKind | null;

  constructor(forced: CanvasKind | null = null) {
    this.chosen = forced;
  }

  /** The canvas to draw the next frame into. */
  next(): CanvasKind {
    return this.chosen ?? (this.times.gpu.length <= this.times.cpu.length ? "gpu" : "cpu");
  }

  /** A frame drawn into `kind` in `ms` (draw, read back, gray). */
  record(kind: CanvasKind, ms: number): void {
    const t = this.times[kind];
    t.push(ms);
    if (t.length > CanvasTrial.TRIAL) t.shift();
    if (this.chosen === null && this.times.gpu.length >= CanvasTrial.TRIAL && this.times.cpu.length >= CanvasTrial.TRIAL) {
      this.chosen = middle(this.times.gpu) <= middle(this.times.cpu) ? "gpu" : "cpu";
    }
  }

  /** For the info line: "gpu canvas (gpu 12, cpu 45 ms)", "trying canvases", or "cpu canvas" when forced. */
  describe(): string {
    const ms = (k: CanvasKind) => (this.times[k].length ? middle(this.times[k]).toFixed(0) : "–");
    const both = `gpu ${ms("gpu")}, cpu ${ms("cpu")} ms`;
    if (this.chosen === null) return `trying canvases (${both})`;
    return this.times.gpu.length && this.times.cpu.length ? `${this.chosen} canvas (${both})` : `${this.chosen} canvas`;
  }
}

function middle(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  return s[s.length >> 1]!;
}

/** The canvas ?canvas= in the page's address asks for, or null. */
export function canvasAsked(search: string): CanvasKind | null {
  const c = new URLSearchParams(search).get("canvas");
  return c === "gpu" || c === "cpu" ? c : null;
}

/** Why this browser cannot try the worker, or null if it can. */
export function whyNotWorker(search: string, g: object = globalThis): string | null {
  if (new URLSearchParams(search).get("grab") === "main") return "?grab=main";
  const has = (name: string) => typeof (g as Record<string, unknown>)[name] === "function";
  if (!has("VideoFrame")) return "no VideoFrame here";
  if (!has("OffscreenCanvas")) return "no OffscreenCanvas here";
  if (!has("Worker")) return "no workers here";
  return null;
}

/** The turns that could take a frame of frameW × frameH to a video of videoW ×
 * videoH, likeliest first (the frame's own `rotation`, where the browser gives
 * one, before the rest); null if none can. The same size could also be the
 * frame upside down; a frame across could be turned either way, or (a browser
 * that turns it when drawing) not at all. */
export function turnsFor(frameW: number, frameH: number, videoW: number, videoH: number, hint?: unknown): Turn[] | null {
  let turns: Turn[];
  if (frameW === videoW && frameH === videoH) turns = [0, 180];
  else if (frameW === videoH && frameH === videoW) turns = [90, 270, 0];
  else return null;
  if (typeof hint === "number" && (turns as number[]).includes(hint)) turns = [hint as Turn, ...turns.filter((t) => t !== hint)];
  return turns;
}

/** The transform (setTransform's a, b, c, d, e, f) under which drawImage(frame,
 * 0, 0) puts the region `r` of the video as shown upright into r.workWidth ×
 * r.workHeight, from a frame of frameW × frameH turned by `turn`: the whole
 * frame turned, scaled and shifted, the rest falling off the canvas. Not a
 * source rectangle: Safari on iPhone took drawImage's source rectangle of a
 * camera video in the wrong place (owner's screenshots, 2026-11-19: on the page,
 * and then in this worker, where whole frames were right and regions shifted). */
export function transformFor(r: Region, turn: Turn, frameW: number, frameH: number): [number, number, number, number, number, number] {
  const kx = 1 / r.scaleX, ky = 1 / r.scaleY;
  switch (turn) {
    // Work (x, y) = ((shown x - r.x) / scaleX, (shown y - r.y) / scaleY), with shown from frame (u, v):
    case 0: return [kx, 0, 0, ky, -r.x * kx, -r.y * ky];                                // shown (u, v)
    case 90: return [0, ky, -kx, 0, (frameH - r.x) * kx, -r.y * ky];                    // shown (frameH - v, u)
    case 180: return [-kx, 0, 0, -ky, (frameW - r.x) * kx, (frameH - r.y) * ky];        // shown (frameW - u, frameH - v)
    case 270: return [0, -ky, kx, 0, -r.x * kx, (frameW - r.y) * ky];                   // shown (v, frameW - u)
  }
}

export class FrameGrab {
  mode: GrabMode;
  /** The turns still to check, the one in use first (turnsFor). */
  private turns: Turn[] = [];
  /** Turns the check found different, with their differences. */
  private tried: string[] = [];
  /** Why the page grabs, when it does. */
  why: string;
  private worker: GrabWorkerLike | null = null;
  private next = 1;
  /** Frames the page grabs before the next check (after an unsure one). */
  private wait = 0;
  private pending: { id: number; resolve: (r: GrabReply) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> } | null = null;

  /** `blocked`: why the worker cannot be tried here (whyNotWorker), or null. */
  constructor(private readonly make: () => GrabWorkerLike, blocked: string | null, private readonly timeoutMs = GRAB.timeoutMs,
    private readonly canvasKind: CanvasKind | null = null) {
    this.mode = blocked === null ? "check" : "main";
    this.why = blocked ?? "";
  }

  /** The worker's canvas, as its last reply described it (CanvasTrial.describe). */
  canvas = "";

  /** Where frames are grabbed, for the camera view's info line. */
  describe(): string {
    return this.mode === "worker" ? `grab in a worker${this.turn ? ` (frames turned ${this.turn}°)` : ""}${this.canvas ? `, ${this.canvas}` : ""}`
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
        w.postMessage({ id, frame, region, shape, gray, turn: this.turn, ...(this.canvasKind ? { canvas: this.canvasKind } : {}) },
          [frame as unknown as Transferable]);
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
      if (reply.canvas) this.canvas = reply.canvas;
      return reply;
    });
  }

  /** The turn frames are drawn with. */
  get turn(): Turn {
    return this.turns[0] ?? 0;
  }

  /** The turns a frame could need (turnsFor), told once, by the first frame. */
  turnsKnown(): boolean {
    return this.turns.length > 0;
  }

  setTurns(turns: Turn[]): void {
    if (!this.turns.length) this.turns = [...turns];
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
  checked(verdict: "same" | "different" | "unsure", difference?: number, correlation?: number): void {
    if (this.mode !== "check") return;
    if (verdict === "same") this.mode = "worker";
    else if (verdict === "unsure") this.wait = GRAB.recheckFrames;
    else {
      // Kept for the message: how far each turn's copy was from the page's (mean
      // gray levels), and how alike (correlation: near 1 for the same picture
      // made lighter or darker, near 0 for one turned or shifted).
      if (difference !== undefined) {
        const d = Number.isFinite(difference) ? difference.toFixed(0) : "–";
        const r = correlation !== undefined && Number.isFinite(correlation) ? ` r ${correlation.toFixed(2)}` : "";
        this.tried.push(`${this.turn}° ${d}${r}`);
      }
      if (this.turns.length > 1) this.turns.shift();      // try the frame turned the next way
      else this.fallBack(`the worker's copy of a frame differs${this.tried.length ? ` (${this.tried.join(", ")})` : ""}`);
    }
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
  let sum = 0, sq = 0;
  for (let i = 0; i < page.length; i++) {
    const v = page[i]!;
    sum += v;
    sq += v * v;
  }
  const n = page.length, mean = sum / n;
  if (Math.sqrt(Math.max(0, sq / n - mean * mean)) < GRAB.minContrast) return "unsure";
  return grayCorrelation(page, worker) >= GRAB.minCorrelation ? "same" : "different";
}

/** How alike two gray copies are, -1 to 1 (Pearson correlation; NaN if their
 * sizes differ or either is flat): about 1 for the same picture even if one is
 * lighter, darker or of more contrast, low for a picture turned or shifted. */
export function grayCorrelation(a: Uint8Array, b: Uint8Array): number {
  if (a.length !== b.length || a.length === 0) return NaN;
  let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!, y = b[i]!;
    sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
  }
  const n = a.length, va = saa - (sa * sa) / n, vb = sbb - (sb * sb) / n;
  return va > 0 && vb > 0 ? (sab - (sa * sb) / n) / Math.sqrt(va * vb) : NaN;
}

/** Mean absolute difference of two gray copies, gray levels (Infinity if their sizes differ). */
export function grayDifference(a: Uint8Array, b: Uint8Array): number {
  if (a.length !== b.length || a.length === 0) return Infinity;
  let d = 0;
  for (let i = 0; i < a.length; i++) d += Math.abs(a[i]! - b[i]!);
  return d / a.length;
}
