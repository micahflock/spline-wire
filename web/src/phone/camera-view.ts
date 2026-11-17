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
import type { Size } from "../camera/focal";
import type { FrameAnalysis, FramePoints } from "../core/protocol";
import type { ChainSpec, Point } from "../core/types";
import { toGray, workSize } from "../overlay/blobs";
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

/** Time per frame the page spends on the overlay, as a share of the time between frames. */
const BUSY_SHARE = 0.5;

/** For the console and the performance test (window.phone.camera). */
export interface CameraStats {
  camera: string;
  size: string;
  /** grab + gray (ms) and blobs (ms) of recent frames. */
  grabMs: number[];
  blobMs: number[];
  frames: number;
  skipped: number;
  analysed: number;
  analyseMs: number[];
  focal: string;
  imageCapture: boolean;
  takePhotoMs: number | null;
  takePhotoError: string | null;
  still: string | null;
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
  private work: { canvas: HTMLCanvasElement | OffscreenCanvas; g: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D } | null = null;
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
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden" && this.open) this.close();
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
    try {
      stream = await request(first);
    } catch (e) {
      if (!first || (e as Error).name === "NotAllowedError") return this.fail("error", e as Error);
      try {
        stream = await request(null);             // the remembered camera is gone
      } catch (e2) {
        return this.fail("error", e2 as Error);
      }
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
    this.setShutter(true);
    this.loop(gen);
  }

  /** Release the camera and hide the view. */
  close(fromButton = true): void {
    if (!this.open) return;
    this.open = false;
    this.gen++;
    if (this.stream) stop(this.stream);
    this.stream = this.track = null;
    this.capture = null;
    this.session = null;
    this.video.srcObject = null;
    if (this.work) this.work.canvas.width = this.work.canvas.height = 0;
    this.work = null;
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
    track.addEventListener("ended", () => {
      if (this.open && this.track === track && !this.shooting) this.close();
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
    // Leave the page at least half of its time: skip frames after a slow one.
    if (now - this.lastWork < this.gap) {
      this.stats.skipped++;
      return;
    }
    const t0 = performance.now();
    const ws = workSize(v.videoWidth, v.videoHeight);
    const work = this.workCanvas(ws);
    work.g.drawImage(v, 0, 0, ws.width, ws.height);
    const gray = toGray(work.g.getImageData(0, 0, ws.width, ws.height).data, ws.width, ws.height);
    const t1 = performance.now();
    const blobsBefore = this.session.stats.blobMs;
    this.session.frame(gray, v.videoWidth, v.videoHeight, now);
    const cost = performance.now() - t0;
    this.lastWork = now;
    this.gap = cost / BUSY_SHARE - cost;
    const st = this.stats;
    st.frames++;
    push(st.grabMs, t1 - t0);
    push(st.blobMs, this.session.stats.blobMs - blobsBefore);
    if (this.session.stats.analysed !== st.analysed) {
      st.analysed = this.session.stats.analysed;
      push(st.analyseMs, this.session.stats.lastAnalyseMs);
    }
  }

  private workCanvas(size: { width: number; height: number }) {
    if (!this.work) {
      // A GPU canvas: the frame is scaled down there and only the small copy is read back.
      const canvas = typeof OffscreenCanvas === "function" ? new OffscreenCanvas(size.width, size.height) : document.createElement("canvas");
      const g = canvas.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
      this.work = { canvas, g };
    }
    const { canvas, g } = this.work;
    if (canvas.width !== size.width || canvas.height !== size.height) {
      canvas.width = size.width;
      canvas.height = size.height;
    }
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";          // area-like averaging, as the coach was calibrated with
    return this.work;
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
  return { camera: "", size: "", grabMs: [], blobMs: [], frames: 0, skipped: 0, analysed: 0, analyseMs: [], focal: "",
    imageCapture: false, takePhotoMs: null, takePhotoError: null, still: null };
}
