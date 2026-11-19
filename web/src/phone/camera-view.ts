// The phone page's camera (web-build.md D16, step 5): a full-screen live view
// of the rear camera with the chain traced over it, a coach that says what to
// fix, and the still taken when the chain has been steady for a moment (or
// when the person presses the shutter, which always works).
//
// The decisions are in src/camera/ (which camera, focal lengths for frames),
// src/overlay/ (blobs, coach, the loop) and lenses.ts; this file is the
// browser's half: the stream, the frames, the drawing and the still.
//
// The still. ImageCapture.takePhoto() at the largest photo size gives the full
// sensor with its EXIF (S2: iPhone 4032×3024 in 1.3 s, Pixel 0.6 s). Where it
// is missing (Firefox) or fails, the still is the full-resolution video frame,
// which has no EXIF: it is measured with the frame's focal length
// (lenses.frameFocal) and marked so the person can tell.
//
// The stream is opened once, at the full 4:3 sensor size, so the still's focal
// length applies to the frames unchanged (S3: iPhone 3024×4032) and so that no
// second request at another size is needed: Firefox on a Pixel refused several
// (NotAllowedError). It is released as soon as the still is taken, the view is
// closed or the page is hidden: iOS keeps the camera light on otherwise.

import { chooseCamera, labelled, rearCameras, type CameraInfo } from "../camera/choose";
import { compareGrays, FrameGrab, whyNotWorker, type GrabWorkerLike } from "../camera/grab";
import type { Size } from "../camera/focal";
import type { FrameAnalysis, FramePoints } from "../core/protocol";
import type { ChainSpec, Point } from "../core/types";
import { toGray } from "../overlay/blobs";
import type { Region } from "../overlay/region";
import { OverlaySession, type OverlayView } from "../overlay/session";
import { followChain } from "../overlay/trace";
import { pinsText, sayFor } from "../overlay/words";
import { frameFocal, ultraWides, type FrameFocal } from "./lenses";
import type { Settings } from "./settings";

export interface Still {
  file: File;
  /** "photo": takePhoto, with the camera's EXIF; "frame": a video frame, none. */
  kind: "photo" | "frame";
  camera: string;
  facing: string | null;
  /** For a frame: the focal length it is to be measured with (null: estimate it). */
  frameFocal: FrameFocal | null;
  /** The chain the overlay last saw, in video pixels, and the video's size: to compare fields of view. */
  frameChain: Point[];
  frameSize: Size;
  /** From the shutter to the still's bytes. */
  ms: number;
}

export interface CameraDeps {
  chain(): Promise<ChainSpec>;
  settings(): Settings;
  saveCamera(camera: CameraInfo): void;
  engineReady(): boolean;
  analyse(points: FramePoints): Promise<FrameAnalysis>;
  /** The still, after the camera has been released. */
  still(still: Still): void;
  /** Open the file input's camera (a fresh tap: the browser wants one). */
  systemCamera(): void;
  /** The person chose Upload a photo: the page switches mode (the view is closed first). */
  upload(): void;
  /** Pick a photo to learn the focal length from (settings). */
  teach(): void;
  /** The view closed by itself (not the person's Close, Upload or back gesture): why, in a few words. */
  closedBy?(why: string): void;
}

interface PhotoCaps {
  imageWidth?: { max?: number };
  imageHeight?: { max?: number };
}
interface ImageCaptureLike {
  takePhoto(options?: { imageWidth?: number; imageHeight?: number }): Promise<Blob>;
  getPhotoCapabilities(): Promise<PhotoCaps>;
}
type VideoWithCallback = HTMLVideoElement & { requestVideoFrameCallback?(cb: (now: number) => void): number };

/** Asked for once: the full 4:3 sensor of a phone (iPhone and Samsung 4032×3024,
 * Pixel 4080×3072, delivered portrait). Ideal, not exact: a camera that cannot
 * still opens. */
const STREAM = { width: { ideal: 4032 }, height: { ideal: 3024 }, frameRate: { ideal: 30 } };

type WorkMode = "gpu" | "cpu";
interface WorkCanvas {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  g: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
}
/** Frames each work canvas is tried on before choosing. */
const TRIAL_FRAMES = 8;

/** How long a page hidden while the camera was asked for (a permission
 * prompt) may take to show again before the camera is given back, ms. */
const VISIBLE_AGAIN_MS = 1500;

/** Time per frame the page spends on the overlay, as a share of the time between frames. */
const BUSY_SHARE = 0.5;

/** For the console and the performance test (window.phone.camera). */
export interface CameraStats {
  camera: string;
  size: string;
  /** grab + gray (ms) and blobs (ms) of recent frames. */
  grabMs: number[];
  blobMs: number[];
  /** Frames drawn from each kind of region (overlay/region.ts), and the last region drawn. */
  regions: Record<Region["kind"], number>;
  region: string;
  /** Where frames are grabbed (camera/grab.ts): "worker", "check" (trying it), or "main: why". */
  grab: string;
  /** The page's own time per frame (ms): all of it when the page grabs; taking
   * the VideoFrame and posting it when the worker does. */
  pageMs: number[];
  frames: number;
  skipped: number;
  analysed: number;
  analyseMs: number[];
  focal: string;
  /** Which work canvas was chosen, and the trial's medians. */
  work: string;
  imageCapture: boolean;
  takePhotoMs: number | null;
  takePhotoError: string | null;
  still: string | null;
  /** Why the view last closed by itself, if it did. */
  closedBy: string | null;
}

export class CameraView {
  readonly stats: CameraStats = blankStats();
  /** How many back cameras the last opening saw. */
  rearCount = 0;
  private open = false;
  private gen = 0;
  private stream: MediaStream | null = null;
  private track: MediaStreamTrack | null = null;
  private capture: ImageCaptureLike | null = null;
  private caps: Promise<PhotoCaps | null> = Promise.resolve(null);
  private session: OverlaySession | null = null;
  private shooting = false;
  private lastWork = -Infinity;
  private gap = 0;
  private latest: OverlayView | null = null;
  private work: Partial<Record<WorkMode, WorkCanvas>> = {};
  /** Which work canvas is cheaper here, once tried (see workCanvas). */
  private workMode: WorkMode | null = null;
  private trial: Record<WorkMode, number[]> = { gpu: [], cpu: [] };
  /** The gray work frame, reused from frame to frame. */
  private gray = new Uint8Array(0);
  /** The camera is being asked for (getUserMedia): a permission prompt may be up. */
  private requesting = false;
  /** Another camera is being opened in place of the open one (switchTo). */
  private switching = false;
  /** The frame grab in a worker, when this browser can (camera/grab.ts). */
  private grab: FrameGrab | null = null;
  private shown = { title: "", detail: "", pins: "", hold: -1, phase: "" };
  private infoAt = 0;
  private readonly el: Record<string, HTMLElement>;
  private readonly video: VideoWithCallback;
  private readonly canvas: HTMLCanvasElement;

  constructor(private readonly deps: CameraDeps) {
    const $ = (id: string) => document.getElementById(id) as HTMLElement;
    this.el = Object.fromEntries(["cam", "cam-say-title", "cam-say-detail", "cam-count", "cam-ring", "cam-info", "cam-fail",
      "cam-fail-title", "cam-fail-text", "cam-switch", "cam-shutter", "cam-teach", "cam-live"].map((id) => [id, $(id)]));
    this.video = $("cam-video") as VideoWithCallback;
    this.canvas = $("cam-overlay") as HTMLCanvasElement;
    $("cam-close").onclick = () => this.close();
    $("cam-fail-close").onclick = () => this.close();
    this.el["cam-shutter"]!.onclick = () => void this.shoot();
    this.el["cam-teach"]!.onclick = () => deps.teach();
    $("cam-fail-system").onclick = () => {
      this.close();
      deps.systemCamera();
    };
    for (const id of ["cam-upload", "cam-fail-upload"]) {
      $(id).onclick = () => {
        this.close();
        deps.upload();
      };
    }
    (this.el["cam-switch"] as HTMLSelectElement).onchange = (e) => {
      const id = (e.target as HTMLSelectElement).value;
      const label = (e.target as HTMLSelectElement).selectedOptions[0]?.textContent ?? "";
      void this.switchTo({ deviceId: id, label }, true);
    };
    window.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Escape" && this.open) this.close();
    });
    // Hidden (another app, the lock button): give the camera back now.
    // Not while the camera is being asked for: iOS's permission prompt (asked
    // once a visit) can report the page hidden, and closing then sent the
    // person back to the page on the first tap. start() looks again once the
    // camera has answered.
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden" && this.open && !this.requesting) this.closeBy("the page was hidden");
    });
    // The phone's back gesture closes the camera, not the page.
    addEventListener("popstate", () => {
      if (this.open) this.close(false);
    });
  }

  get isOpen(): boolean {
    return this.open;
  }

  /** Tracks this view holds that are still live: 0 once closed (the e2e test checks it). */
  liveTracks(): number {
    return this.stream?.getTracks().filter((t) => t.readyState === "live").length ?? 0;
  }

  /** Open the camera view. Failing to get the camera shows why, and the system camera instead. */
  async start(): Promise<void> {
    if (this.open) return;
    this.open = true;
    const gen = ++this.gen;
    Object.assign(this.stats, blankStats());
    this.shooting = false;
    this.latest = null;
    this.shown = { title: "", detail: "", pins: "", hold: -1, phase: "" };
    this.el["cam"]!.hidden = false;
    this.el["cam-fail"]!.hidden = true;
    this.el["cam-live"]!.hidden = false;
    document.documentElement.classList.add("cam-on");
    history.pushState({ camera: true }, "");
    this.say("Starting the camera…", "");
    this.setShutter(false);

    const md = navigator.mediaDevices;
    if (!window.isSecureContext || !md?.getUserMedia) {
      this.fail("insecure", null);
      return;
    }
    const s = this.deps.settings();
    const avoid = ultraWides(s);
    let devices = await cameras();
    // Labels are there only once the camera has been allowed (Chrome remembers
    // that; Safari asks each visit): then the right camera opens first time.
    const first = labelled(devices) ? chooseCamera(devices, s.camera, avoid) : null;
    let stream: MediaStream;
    this.requesting = true;
    try {
      stream = await request(first);
    } catch (e) {
      if (!first || (e as Error).name === "NotAllowedError") return this.fail("error", e as Error);
      try {
        stream = await request(null);             // the remembered camera is gone
      } catch (e2) {
        return this.fail("error", e2 as Error);
      }
    } finally {
      this.requesting = false;
    }
    // Hidden while it was asked for: a permission prompt, which gives the page
    // back in a moment; else the person left, and the camera is given back.
    if (gen === this.gen && !(await visibleWithin(VISIBLE_AGAIN_MS))) {
      stop(stream);
      this.closeBy("the page stayed hidden after the camera was allowed");
      return;
    }
    if (gen !== this.gen) {
      stop(stream);
      return;
    }
    await this.attach(stream);
    devices = await cameras();
    const best = chooseCamera(devices, s.camera, avoid);
    if (best?.deviceId && best.deviceId !== this.current()?.deviceId) {
      // facingMode opened another camera: the virtual triple camera on iPhone,
      // the ultra-wide on a Galaxy S23 Ultra (S3).
      await this.switchTo(best, false);
    }
    if (gen !== this.gen) return;
    this.fillSwitcher(devices);
    const now = this.current();
    if (now) this.deps.saveCamera(now);
    this.session = new OverlaySession({
      chain: await this.deps.chain(),
      focalPx: () => this.focal()?.px ?? null,
      analyse: (p) => this.deps.analyse(p),
      show: (v) => this.draw(v),
      capture: () => this.shoot(),
      // Estimating the focal length costs 400-640 ms an analysis (step 1's log): once a second is enough.
      minIntervalMs: () => (this.focal() ? 0 : 1000),
    });
    this.grab = new FrameGrab(() => new Worker(new URL("../camera/grab-worker.ts", import.meta.url), { type: "module" }) as unknown as GrabWorkerLike,
      whyNotWorker(location.search));
    this.stats.grab = this.grabText();
    this.setShutter(true);
    this.loop(gen);
  }

  /** Release the camera and hide the view. */
  /** Close by itself, and say why (deps.closedBy). */
  private closeBy(why: string): void {
    if (!this.open) return;
    this.close();
    this.stats.closedBy = why;
    this.deps.closedBy?.(why);
  }

  close(fromButton = true): void {
    if (!this.open) return;
    this.open = false;
    this.gen++;
    if (this.stream) stop(this.stream);
    this.stream = this.track = null;
    this.capture = null;
    this.session = null;
    this.grab?.close();
    this.grab = null;
    this.video.srcObject = null;
    for (const w of Object.values(this.work)) w.canvas.width = w.canvas.height = 0;
    this.work = {};
    this.workMode = null;
    this.trial = { gpu: [], cpu: [] };
    this.el["cam"]!.hidden = true;
    document.documentElement.classList.remove("cam-on");
    if (fromButton && (history.state as { camera?: boolean } | null)?.camera) history.back();
  }

  /** The focal length the frames are analysed with, now. */
  focal(): FrameFocal | null {
    const v = this.video;
    if (!v.videoWidth) return null;
    return frameFocal(this.deps.settings(), this.current()?.label ?? null, { width: v.videoWidth, height: v.videoHeight });
  }

  // ---- the stream ----

  private current(): CameraInfo | null {
    if (!this.track) return null;
    return { deviceId: this.track.getSettings().deviceId ?? "", label: this.track.label };
  }

  private async attach(stream: MediaStream): Promise<void> {
    const old = this.stream;
    this.stream = stream;
    this.track = stream.getVideoTracks()[0] ?? null;
    if (old && old !== stream) stop(old);
    const track = this.track;
    if (!track) return;
    // The camera went away (another app took it, or it failed): close. Not
    // while switching cameras: iOS runs one camera at a time and ends the open
    // one as the next opens, and on the first tap of a visit the page switches
    // from the virtual triple camera it was given to the main lens (start()).
    track.addEventListener("ended", () => {
      if (this.open && this.track === track && !this.shooting && !this.switching) this.closeBy("the camera stopped");
    });
    // Firefox refused a second request at another size (S3): ask the live
    // track instead, only if it came well short of what it can do.
    try {
      const c = track.getCapabilities?.();
      const set = track.getSettings();
      const longMax = Math.max(c?.width?.max ?? 0, c?.height?.max ?? 0);
      if (longMax && Math.max(set.width ?? 0, set.height ?? 0) < 0.9 * longMax) {
        await track.applyConstraints({ width: { ideal: c!.width!.max }, height: { ideal: c!.height!.max } });
      }
    } catch {
      /* keep the size it gave */
    }
    const IC = (globalThis as { ImageCapture?: new (t: MediaStreamTrack) => ImageCaptureLike }).ImageCapture;
    this.capture = IC ? new IC(track) : null;
    this.caps = this.capture ? this.capture.getPhotoCapabilities().catch(() => null) : Promise.resolve(null);
    this.stats.imageCapture = !!this.capture;
    this.video.srcObject = stream;
    await this.video.play().catch(() => {});
    this.stats.camera = track.label;
  }

  private async switchTo(camera: CameraInfo, chosen: boolean): Promise<void> {
    if (!camera.deviceId || camera.deviceId === this.current()?.deviceId) return;
    const before = this.current();
    this.switching = true;
    try {
      const stream = await request(camera);
      if (!this.open) {
        stop(stream);
        return;
      }
      await this.attach(stream);
      if (chosen) this.deps.saveCamera(this.current() ?? camera);
    } catch {
      // Keep the camera that is open (iOS may have stopped it: then open it again).
      if (this.track?.readyState === "ended" && before) {
        try {
          await this.attach(await request(before));
        } catch (e) {
          this.fail("error", e as Error);
        }
      }
    } finally {
      this.switching = false;
    }
    this.fillSwitcher(await cameras());
  }

  private fillSwitcher(devices: CameraInfo[]): void {
    const rear = rearCameras(devices).filter((d) => d.deviceId);
    this.rearCount = rear.length;
    const select = this.el["cam-switch"] as HTMLSelectElement;
    select.hidden = rear.length < 2;
    const now = this.current();
    select.replaceChildren(...rear.map((d, i) => new Option(d.label || `Camera ${i + 1}`, d.deviceId, false, d.deviceId === now?.deviceId)));
  }

  // ---- frames ----

  private loop(gen: number): void {
    const v = this.video;
    const next = () => {
      if (gen !== this.gen) return;
      if (v.requestVideoFrameCallback) v.requestVideoFrameCallback((now) => tick(now));
      else requestAnimationFrame(tick);
    };
    const tick = (now: number) => {
      if (gen !== this.gen) return;
      next();
      this.frame(now);
    };
    next();
  }

  private frame(now: number): void {
    const v = this.video;
    if (this.shooting || !this.session || v.readyState < 2 || !v.videoWidth) return;
    // Leave the page at least half of its time: skip frames after a slow one;
    // and while the worker has the last frame, wait for it.
    if (now - this.lastWork < this.gap || this.grab?.busy) {
      this.stats.skipped++;
      return;
    }
    const t0 = performance.now();
    // Only the part of the frame round the chain while it is tracked, drawn
    // finely; else the whole frame, or a quarter of it (overlay/region.ts).
    const r = this.session.region(v.videoWidth, v.videoHeight);
    // In a worker when the browser can (camera/grab.ts); a check frame is
    // grabbed here as well, and the two copies compared.
    let check: VideoFrame | null = null;
    if (this.grab?.wants()) {
      const frame = this.videoFrame();
      if (frame && this.grab.mode === "worker") return this.toWorker(frame, now, r, t0);
      check = frame;
    }
    // The trial takes turns, so that both canvases see the same mix of regions.
    const mode = this.workMode ?? (this.trial.gpu.length <= this.trial.cpu.length ? "gpu" : "cpu");
    const work = this.workCanvas(mode, r.workWidth, r.workHeight);
    work.g.drawImage(v, r.x, r.y, r.width, r.height, 0, 0, r.workWidth, r.workHeight);
    const n = r.workWidth * r.workHeight;
    if (this.gray.length < n) this.gray = new Uint8Array(n);
    const gray = toGray(work.g.getImageData(0, 0, r.workWidth, r.workHeight).data, r.workWidth, r.workHeight, this.gray);
    const t1 = performance.now();
    if (check) this.check(check, r, gray.data.slice());
    if (!this.workMode) {
      this.trial[mode].push(t1 - t0);
      if (this.trial.cpu.length >= TRIAL_FRAMES && this.trial.gpu.length >= TRIAL_FRAMES) {
        this.workMode = median(this.trial.gpu) <= median(this.trial.cpu) ? "gpu" : "cpu";
        const other = this.work[this.workMode === "gpu" ? "cpu" : "gpu"];
        if (other) other.canvas.width = other.canvas.height = 0;
        this.stats.work = `${this.workMode} canvas (gpu ${median(this.trial.gpu).toFixed(1)} ms, cpu ${median(this.trial.cpu).toFixed(1)} ms)`;
      }
    }
    const blobsBefore = this.session.stats.blobMs;
    this.session.frame(gray, v.videoWidth, v.videoHeight, now, r);
    const cost = performance.now() - t0;
    this.lastWork = now;
    this.gap = cost / BUSY_SHARE - cost;
    this.counted(r, t1 - t0, this.session.stats.blobMs - blobsBefore, cost);
  }

  private counted(r: Region, grabMs: number, blobMs: number, pageMs: number): void {
    const st = this.stats;
    st.frames++;
    st.regions[r.kind]++;
    st.region = `${r.kind} ${r.workWidth}×${r.workHeight}`;
    push(st.grabMs, grabMs);
    push(st.blobMs, blobMs);
    push(st.pageMs, pageMs);
    if (this.session && this.session.stats.analysed !== st.analysed) {
      st.analysed = this.session.stats.analysed;
      push(st.analyseMs, this.session.stats.lastAnalyseMs);
    }
  }

  /** The video's current frame as a VideoFrame of the video's own size, or null
   * (and the page grabs from now on) if the browser cannot make one. */
  private videoFrame(): VideoFrame | null {
    const v = this.video, grab = this.grab!;
    let frame: VideoFrame;
    try {
      frame = new VideoFrame(v, { timestamp: Math.round(performance.now() * 1000) });
    } catch (e) {
      this.grabFailed(grab, `VideoFrame failed (${(e as Error).name})`);
      return null;
    }
    if (frame.displayWidth !== v.videoWidth || frame.displayHeight !== v.videoHeight) {
      this.grabFailed(grab, `VideoFrame ${frame.displayWidth}×${frame.displayHeight}, video ${v.videoWidth}×${v.videoHeight}`);
      frame.close();
      return null;
    }
    return frame;
  }

  /** The worker draws `frame` from `r` too: is its copy the page's (`page`)? */
  private check(frame: VideoFrame, r: Region, page: Uint8Array): void {
    const grab = this.grab!, gen = this.gen;
    grab.run(frame, r, this.session!.shape, true).then((reply) => {
      if (gen !== this.gen || this.grab !== grab) return;
      grab.checked(compareGrays(page, reply.gray ?? new Uint8Array(0)));
      this.stats.grab = this.grabText();
    }, (e: Error) => {
      if (gen === this.gen) this.grabFailed(grab, `worker: ${e.message}`);
    });
  }

  /** Hand this frame to the grab worker, which draws `r` and finds its blobs. */
  private toWorker(frame: VideoFrame, now: number, r: Region, t0: number): void {
    const grab = this.grab!, session = this.session!, v = this.video, gen = this.gen;
    const vw = v.videoWidth, vh = v.videoHeight;
    let pageMs = 0;
    grab.run(frame, r, session.shape).then((reply) => {
      if (gen !== this.gen || this.session !== session || this.shooting) return;
      session.found(reply.blobs ?? [], vw, vh, now, reply.blobMs ?? 0);
      this.counted(r, reply.grabMs ?? 0, reply.blobMs ?? 0, pageMs);
    }, (e: Error) => {
      if (gen === this.gen) this.grabFailed(grab, `worker: ${e.message}`);
    });
    pageMs = performance.now() - t0;
    this.lastWork = now;
    this.gap = 0;                                // the page's share is the VideoFrame and the post: next frame at once
  }

  private grabFailed(grab: FrameGrab, why: string): void {
    if (this.grab !== grab) return;
    grab.fallBack(why);
    this.stats.grab = this.grabText();
  }

  private grabText(): string {
    const g = this.grab;
    return !g ? "" : g.mode === "main" ? `main: ${g.why}` : g.mode;
  }

  /** The small canvas a frame is drawn into and read back from. Which kind is
   * cheaper depends on the browser: a GPU canvas scales the frame on the GPU
   * and reads back only the small copy; a CPU one (willReadFrequently) avoids
   * the read-back but may copy the whole frame out of the GPU first. In
   * headless Chromium (no GPU) a 3024×4032 frame took 34 ms into a GPU canvas
   * and 21 ms into a CPU one; a phone has a GPU. So the first frames try each,
   * and the rest use the faster. */
  private workCanvas(mode: WorkMode, width: number, height: number): WorkCanvas {
    let w = this.work[mode];
    if (!w) {
      const canvas = typeof OffscreenCanvas === "function" ? new OffscreenCanvas(width, height) : document.createElement("canvas");
      const g = canvas.getContext("2d", { willReadFrequently: mode === "cpu" }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
      w = this.work[mode] = { canvas, g };
    }
    // The regions change size from frame to frame: the canvas only grows, and
    // each frame is drawn into its top-left corner (resizing reallocates it).
    if (w.canvas.width < width || w.canvas.height < height) {
      w.canvas.width = Math.max(w.canvas.width, width);
      w.canvas.height = Math.max(w.canvas.height, height);
    }
    w.g.imageSmoothingEnabled = true;
    w.g.imageSmoothingQuality = "high";          // area-like averaging, as the coach was calibrated with
    return w;
  }

  // ---- drawing ----

  private draw(view: OverlayView): void {
    this.latest = view;
    const v = this.video, c = this.canvas;
    const vw = v.videoWidth, vh = v.videoHeight;
    const box = c.getBoundingClientRect();
    const dpr = Math.min(2, devicePixelRatio || 1);
    const cw = Math.round(box.width * dpr), ch = Math.round(box.height * dpr);
    if (c.width !== cw || c.height !== ch) {
      c.width = cw;
      c.height = ch;
    }
    const g = c.getContext("2d")!;
    g.clearRect(0, 0, cw, ch);
    if (vw && vh) {
      // The video is shown whole (object-fit: contain): what the still will hold.
      const k = Math.min(cw / vw, ch / vh), ox = (cw - vw * k) / 2, oy = (ch - vh * k) / 2;
      const X = (x: number) => ox + (x + 0.5) * k, Y = (y: number) => oy + (y + 0.5) * k;
      const phase = view.state.phase;
      const color = phase === "steady" || phase === "capture" ? "#4cc27a" : phase === "adjusting" ? "#f2b705" : "rgba(255,255,255,.8)";
      const chain = followChain(view.chain, view.blobs);
      const onChain = new Set(chain.map(([x, y]) => `${x},${y}`));
      g.lineWidth = 1.5 * dpr;
      g.strokeStyle = "rgba(255,255,255,.45)";
      for (const b of view.blobs) {
        if (onChain.has(`${b.x},${b.y}`)) continue;
        g.beginPath();
        g.arc(X(b.x), Y(b.y), Math.max(3 * dpr, (b.d / 2) * k), 0, 2 * Math.PI);
        g.stroke();
      }
      if (chain.length) {
        g.lineWidth = 3 * dpr;
        g.strokeStyle = color;
        g.lineJoin = "round";
        g.beginPath();
        chain.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))));
        g.stroke();
        const r = Math.max(5 * dpr, (median(view.blobs.map((b) => b.d)) / 2) * k + 2 * dpr);
        g.lineWidth = 2.5 * dpr;
        for (const [x, y] of chain) {
          g.beginPath();
          g.arc(X(x), Y(y), r, 0, 2 * Math.PI);
          g.stroke();
        }
      }
    }
    this.render(view);
  }

  private render(view: OverlayView): void {
    const st = view.state;
    const say = this.shooting ? { title: "Taking the photo…", detail: "Keep still for a moment." } : sayFor(st, { engineReady: this.deps.engineReady() });
    this.say(say.title, say.detail);
    const pins = pinsText(st);
    if (pins !== this.shown.pins) this.el["cam-count"]!.textContent = this.shown.pins = pins;
    const hold = st.phase === "capture" || st.phase === "captured" ? 1 : st.hold;
    if (Math.abs(hold - this.shown.hold) > 0.01 || st.phase !== this.shown.phase) {
      const ring = this.el["cam-ring"]!;
      ring.style.strokeDashoffset = String(RING * (1 - hold));
      this.el["cam-shutter"]!.dataset.phase = st.phase;
      this.shown.hold = hold;
      this.shown.phase = st.phase;
    }
    const now = performance.now();
    if (now - this.infoAt > 500) {
      this.infoAt = now;
      this.info();
    }
  }

  private info(): void {
    const v = this.video, f = this.focal();
    const st = this.stats;
    const focal = f ? `${f.focal35.toFixed(1)} mm ${f.from === "lens" ? "(this camera's last photo)" : f.from === "override" ? "(override)" : "(Settings)"}` : "focal estimated";
    st.focal = focal;
    st.size = `${v.videoWidth}×${v.videoHeight}`;
    const sharp = this.latest?.blobs.length ? median(this.latest.blobs.map((b) => b.sharpness)) : null;
    this.el["cam-info"]!.textContent = [st.camera || "camera", st.size, focal,
      st.grabMs.length ? `frame ${median(st.grabMs.slice(-30)).toFixed(0)}+${median(st.blobMs.slice(-30)).toFixed(0)} ms` : "",
      this.grab?.describe() ?? "",
      st.region,
      st.analyseMs.length ? `analysis ${Math.round(median(st.analyseMs))} ms` : "", sharp !== null ? `sharpness ${sharp.toFixed(2)}` : ""]
      .filter(Boolean).join(" · ");
    this.el["cam-teach"]!.hidden = !!f;
  }

  private say(title: string, detail: string): void {
    if (title !== this.shown.title) this.el["cam-say-title"]!.textContent = this.shown.title = title;
    if (detail !== this.shown.detail) this.el["cam-say-detail"]!.textContent = this.shown.detail = detail;
  }

  private setShutter(on: boolean): void {
    (this.el["cam-shutter"] as HTMLButtonElement).disabled = !on;
  }

  private fail(why: "insecure" | "error", e: Error | null): void {
    if (!this.open) return;
    if (this.stream) stop(this.stream);
    this.stream = this.track = null;
    this.el["cam-live"]!.hidden = true;
    this.el["cam-fail"]!.hidden = false;
    const [title, text] = why === "insecure"
      ? ["This page can't use the camera here.", "The live camera needs the site's https:// address."]
      : e?.name === "NotAllowedError" || e?.name === "SecurityError"
        ? ["The camera wasn't allowed.", "To allow it: on iPhone, the aA button in the address bar, Website Settings, Camera; on Android, the icon left of the address, Permissions, Camera. Then open the camera again."]
        : ["The camera couldn't be opened.", `${e?.name ?? "Error"}: ${e?.message ?? ""}. Another app may be using it.`];
    this.el["cam-fail-title"]!.textContent = title;
    this.el["cam-fail-text"]!.textContent = text;
  }

  // ---- the still ----

  /** Take the still now: from the shutter, or from the coach after a steady hold. */
  async shoot(): Promise<void> {
    if (this.shooting || !this.track || !this.open) return;
    this.shooting = true;
    this.setShutter(false);
    this.say("Taking the photo…", "Keep still for a moment.");
    this.el["cam"]!.classList.add("flash");
    const v = this.video;
    const frameSize = { width: v.videoWidth, height: v.videoHeight };
    const frameChain = this.latest?.chain.map((p) => [p[0], p[1]] as Point) ?? [];
    const camera = this.track.label;
    const facing = this.track.getSettings().facingMode ?? null;
    const t0 = performance.now();
    try {
      const photo = await this.takePhoto();
      const stamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, "").replace("T", "-");
      let still: Still;
      if (photo) {
        const ext = photo.type === "image/png" ? "png" : "jpg";
        still = { file: new File([photo], `photo-${stamp}.${ext}`, { type: photo.type || "image/jpeg" }), kind: "photo",
          camera, facing, frameFocal: null, frameChain, frameSize, ms: performance.now() - t0 };
      } else {
        const blob = await this.grabFrame();
        still = { file: new File([blob], `frame-${stamp}.jpg`, { type: "image/jpeg" }), kind: "frame",
          camera, facing, frameFocal: this.focal(), frameChain, frameSize, ms: performance.now() - t0 };
      }
      this.stats.still = `${still.kind} ${still.file.size} B in ${Math.round(still.ms)} ms`;
      this.close();
      this.deps.still(still);
    } catch (e) {
      this.shooting = false;
      this.setShutter(true);
      this.say("Couldn't take the photo.", (e as Error).message);
    } finally {
      this.el["cam"]?.classList.remove("flash");
    }
  }

  private async takePhoto(): Promise<Blob | null> {
    if (!this.capture) return null;
    const t0 = performance.now();
    try {
      const caps = await within(this.caps, 3000).catch(() => null);
      const w = caps?.imageWidth?.max, h = caps?.imageHeight?.max;
      // The largest photo size: iPhone's default is 1920×1080 (S2).
      const blob = await within(this.capture.takePhoto(w && h ? { imageWidth: w, imageHeight: h } : undefined), 20_000);
      this.stats.takePhotoMs = Math.round(performance.now() - t0);
      return blob.size ? blob : null;
    } catch (e) {
      this.stats.takePhotoError = `${(e as Error).name}: ${(e as Error).message}`;
      return null;                               // the frame instead
    }
  }

  private grabFrame(): Promise<Blob> {
    const v = this.video;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    return new Promise((resolve, reject) => c.toBlob((b) => {
      c.width = c.height = 0;                    // a 12-megapixel canvas: give it back at once (iOS counts it)
      if (b) resolve(b);
      else reject(new Error("the video frame could not be saved"));
    }, "image/jpeg", 0.95));
  }
}

/** The shutter ring's circumference (r = 36 in its 80×80 box). */
export const RING = 2 * Math.PI * 36;

async function cameras(): Promise<CameraInfo[]> {
  try {
    return (await navigator.mediaDevices.enumerateDevices())
      .filter((d) => d.kind === "videoinput")
      .map((d) => ({ deviceId: d.deviceId, label: d.label }));
  } catch {
    return [];
  }
}

function request(camera: CameraInfo | null): Promise<MediaStream> {
  const which = camera?.deviceId ? { deviceId: { exact: camera.deviceId } } : { facingMode: { ideal: "environment" } };
  return navigator.mediaDevices.getUserMedia({ video: { ...which, ...STREAM }, audio: false });
}

/** True once the page is visible, or false if it is still hidden after `ms`. */
function visibleWithin(ms: number): Promise<boolean> {
  if (document.visibilityState !== "hidden") return Promise.resolve(true);
  return new Promise((resolve) => {
    const done = (v: boolean) => {
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", seen);
      resolve(v);
    };
    const seen = () => {
      if (document.visibilityState !== "hidden") done(true);
    };
    const timer = setTimeout(() => done(false), ms);
    document.addEventListener("visibilitychange", seen);
  });
}

function stop(stream: MediaStream): void {
  for (const t of stream.getTracks()) t.stop();
}

function within<T>(p: Promise<T>, ms: number): Promise<T> {
  return Promise.race([p, new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`no answer in ${ms / 1000} s`)), ms))]);
}

function push(list: number[], v: number): void {
  list.push(Math.round(v * 100) / 100);
  if (list.length > 300) list.shift();
}

function median(v: number[]): number {
  if (!v.length) return 0;
  const s = [...v].sort((a, b) => a - b);
  return s[s.length >> 1]!;
}

function blankStats(): CameraStats {
  return { camera: "", size: "", grabMs: [], blobMs: [], regions: { full: 0, tile: 0, track: 0 }, region: "", grab: "", pageMs: [], frames: 0, skipped: 0, analysed: 0, analyseMs: [], focal: "", work: "",
    imageCapture: false, takePhotoMs: null, takePhotoError: null, still: null, closedBy: null };
}
