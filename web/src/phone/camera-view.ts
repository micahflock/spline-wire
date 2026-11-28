// The phone page's camera (web-build.md D16, step 5): a live view (a card over the page)
// of the rear camera with the chain traced over it, a coach that says what to
// fix, and the still taken when the chain has been steady for a moment. There
// is no shutter: the coach takes the still, the lock-on rings on the pins showing
// the hold (the library picker on the page is the way round it).
// The diagnostics sit behind the Debug key (debug mode only). The card carries the coach's phase (data-phase) and whether the view is
// ready to take a still (data-ready), for the tests.
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
import { QrScan } from "../camera/qr-scan";
import { FULL_STREAM, smallStill, stillSize, streamFor, type StreamSize } from "../camera/stream";
import { canvasAsked, compareGrays, FrameGrab, grayCorrelation, grayDifference, turnsFor, whyNotWorker, type GrabWorkerLike } from "../camera/grab";
import type { Size } from "../camera/focal";
import type { FrameAnalysis, FramePoints } from "../core/protocol";
import type { ChainSpec, Point } from "../core/types";
import { toGray } from "../overlay/blobs";
import { ArrowAnimator, HoldAnimator, lockAlpha, lockRadius } from "../overlay/animate";
import { COACH } from "../overlay/coach";
import type { Region } from "../overlay/region";
import { OverlaySession, type OverlayView } from "../overlay/session";
import { followChain } from "../overlay/trace";
import { pinsText, sayFor } from "../overlay/words";
import { browserInfo, uaData, type UAData } from "./browser";
import { frameFocal, ultraWides, type FrameFocal } from "./lenses";
import type { LiveView, Settings } from "./settings";

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
  /** From the coach's call to the still's bytes. */
  ms: number;
  /** For a frame: why takePhoto gave no photo ("NotSupportedError: …"), if it was tried. */
  why?: string;
}

export interface CameraDeps {
  chain(): Promise<ChainSpec>;
  settings(): Settings;
  saveCamera(camera: CameraInfo): void;
  /** The phone's cameras as the view last found them (Settings, Camera, lists the back ones). */
  camerasListed?(devices: CameraInfo[]): void;
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
  /** A QR code seen in the frames (once per code while the view is open): the desktop page's pairing code. */
  code?(text: string): void;
  /** This camera's still came back small with the fast stream (camera/stream.ts): open it at full size from now on. */
  fullStream?(cameraLabel: string): void;
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

/** The frame rate asked for; the size is camera/stream.ts's (the full 4:3 sensor,
 * iPhone and Samsung 4032×3024, Pixel 4080×3072, delivered portrait; or the fast
 * 1440×1920). Ideal, not exact: a camera that cannot still opens. */
const FRAME_RATE = { frameRate: { ideal: 30 } };

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

/** The longest the view stays hidden waiting for the camera to settle at its final size (settled()):
 * past this it is shown anyway, so a camera that never reports its size does not leave it waiting. */
const SETTLE_MS = 2500;

/** A video frame callback this late, with the video playing, means the browser
 * stopped giving them (as it may for a video covered by Settings' chain-only
 * view): frames are then taken on animation frames, as without the callback. */
const VFC_STALL_MS = 1000;

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
  /** The worker's canvas and its trial (camera/grab.ts, CanvasTrial.describe). */
  canvas: string;
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
  /** The stream's size and why (camera/stream.ts, streamFor). */
  stream: string;
  /** The QR scan beside the overlay: frames decoded × the page's draw + the worker's decode of the last (ms). */
  qr: string;
  imageCapture: boolean;
  takePhotoMs: number | null;
  takePhotoError: string | null;
  still: string | null;
  /** The last takePhoto still's size (JPEG or PNG), read with the fast stream (camera/stream.ts). */
  stillSize: string | null;
  /** Why the view last closed by itself, if it did. */
  closedBy: string | null;
  /** The browser, OS, phone and build, as the info's first line shows them. */
  browser: string;
  /** The way the arrow over the chain last pointed (frame axes, y down), or null with none drawn. */
  arrow: [number, number] | null;
  /** Holds broken, by the problem that broke them (overlay/coach.ts): "moving 3, pins-missing 1". */
  breaks: string;
  /** What paces the frames: "video frames" (requestVideoFrameCallback), or "animation frames" and why. */
  loop: string;
  /** Settings, Live camera shows: the video in colour or grey under the chain, or only the chain. */
  shows: LiveView;
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
  /** What Chromium tells of the phone when asked (browser.ts): the real Android version and model. */
  private ua: UAData | null = null;
  /** The camera is being asked for (getUserMedia): a permission prompt may be up. */
  private requesting = false;
  /** Another camera is being opened in place of the open one (switchTo). */
  private switching = false;
  /** The open stream was asked for at the fast size (camera/stream.ts). */
  private fast = false;
  /** The frame grab in a worker, when this browser can (camera/grab.ts). */
  private grab: FrameGrab | null = null;
  /** Looks for the desktop page's pairing code in the frames (camera/qr-scan.ts). */
  private qr: QrScan | null = null;
  /** The arrow and the lock-on rings as drawn: eased (overlay/animate.ts). */
  private readonly arrowShown = new ArrowAnimator();
  private readonly holdShown = new HoldAnimator();
  /** The video frame callbacks stopped while the video played (VFC_STALL_MS): animation frames from then on. */
  private vfcStalled = false;
  /** The chain's centre on the canvas when last drawn: where a fading arrow stays. */
  private centre: Point | null = null;
  private shown = { title: "", detail: "", pins: "", phase: "", info: "" };
  private infoAt = 0;
  private readonly el: Record<string, HTMLElement>;
  private readonly video: VideoWithCallback;
  private readonly canvas: HTMLCanvasElement;

  constructor(private readonly deps: CameraDeps) {
    void uaData().then((d) => (this.ua = d));
    const $ = (id: string) => document.getElementById(id) as HTMLElement;
    this.el = Object.fromEntries(["cam", "cam-say-title", "cam-say-detail", "cam-count", "cam-info", "cam-fail",
      "cam-fail-title", "cam-fail-text", "cam-teach", "cam-live"].map((id) => [id, $(id)]));
    this.video = $("cam-video") as VideoWithCallback;
    this.canvas = $("cam-overlay") as HTMLCanvasElement;
    this.video.addEventListener("loadedmetadata", () => this.fit());
    this.video.addEventListener("resize", () => this.fit());   // the stream's size changed (another camera, turned)
    // A tap on the page round the card closes the view (there is no close key). Not while the camera starts:
    // nothing is shown then, and a second tap on Scan would only cancel it.
    const cam = this.el["cam"]!;
    cam.addEventListener("click", (e) => {
      if (e.target === cam && !document.documentElement.classList.contains("cam-starting")) this.close();
    });
    // The page behind does not scroll: overflow: hidden (phone.css, html.cam-on) is not enough for touch on iOS.
    // Only the failure card and the diagnostics, which may be longer than the screen, scroll themselves.
    cam.addEventListener("touchmove", (e) => {
      if (!(e.target as Element).closest(".cam-fail, .cam-info")) e.preventDefault();
    }, { passive: false });
    $("cam-fail-close").onclick = () => this.close();
    this.el["cam-teach"]!.onclick = () => deps.teach();
    $("cam-fail-system").onclick = () => {
      this.close();
      deps.systemCamera();
    };
    $("cam-fail-upload").onclick = () => {
      this.close();
      deps.upload();
    };
    // The diagnostics, behind the Debug key (closed until asked for).
    const debugKey = $("cam-debug");
    debugKey.onclick = () => {
      const on = this.el["cam-info"]!.hidden;
      this.el["cam-info"]!.hidden = !on;
      debugKey.setAttribute("aria-expanded", String(on));
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
    this.arrowShown.reset();
    this.holdShown.reset();
    this.centre = null;
    this.shown = { title: "", detail: "", pins: "", phase: "", info: "" };
    this.el["cam"]!.hidden = false;
    this.stats.shows = this.deps.settings().liveView;
    this.el["cam"]!.classList.toggle("chain-only", this.stats.shows === "chain");
    this.el["cam"]!.classList.toggle("grey", this.stats.shows === "grey");
    this.el["cam-fail"]!.hidden = true;
    this.el["cam-live"]!.hidden = false;
    // The Debug key and its diagnostics only in debug mode (settings.ts).
    const debugKey = document.getElementById("cam-debug")!;
    debugKey.hidden = !this.deps.settings().debug;
    if (debugKey.hidden) {
      this.el["cam-info"]!.hidden = true;
      debugKey.setAttribute("aria-expanded", "false");
    }
    document.documentElement.classList.add("cam-on");
    // Not shown until the camera has settled (reveal): opening it switches lenses and sizes on the
    // way, which drew one box after another. Meanwhile the Scan key shows it is starting (phone.css).
    document.documentElement.classList.add("cam-starting");
    history.pushState({ camera: true }, "");
    this.say("Starting the camera…", "");
    this.setReady(false);

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
    this.fast = !!first && this.streamChoice(first.label).fast;
    try {
      stream = await request(first, this.fast ? this.streamChoice(first!.label).size : FULL_STREAM);
    } catch (e) {
      if (!first || (e as Error).name === "NotAllowedError") return this.fail("error", e as Error);
      this.fast = false;
      try {
        stream = await request(null, FULL_STREAM);             // the remembered camera is gone
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
      await this.switchTo(best);
    }
    if (gen !== this.gen) return;
    this.listed(devices);
    const now = this.current();
    if (now) this.deps.saveCamera(now);
    await this.resize();
    if (gen !== this.gen) return;
    await this.settled();
    if (gen !== this.gen) return;
    this.reveal();
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
      whyNotWorker(location.search), undefined, canvasAsked(location.search));
    this.stats.grab = this.grabText();
    this.qr = new QrScan((text) => this.deps.code?.(text));
    this.setReady(true);
    this.loop(gen);
  }

  /** The card takes the video's shape (phone.css, .cam-live.fit), so the picture fills it with no
   * black bars. Kept from one opening to the next: the same camera opens at the same shape. */
  private fit(): void {
    const v = this.video;
    if (!v.videoWidth || !v.videoHeight) return;
    this.el["cam-live"]!.style.setProperty("--cam-ar", String(v.videoWidth / v.videoHeight));
    this.el["cam-live"]!.classList.add("fit");
  }

  /** Until the video shows frames at the size its track has settled on (after the lens switch and
   * resize(); the video may be turned from the track's width and height), or SETTLE_MS at most. */
  private async settled(): Promise<void> {
    const v = this.video, until = performance.now() + SETTLE_MS;
    const sides = (a: number, b: number) => `${Math.min(a, b)}x${Math.max(a, b)}`;
    while (performance.now() < until) {
      const set = this.track?.getSettings();
      const want = set?.width && set?.height ? sides(set.width, set.height) : null;
      if (v.readyState >= 2 && v.videoWidth && (!want || want === sides(v.videoWidth, v.videoHeight))) return;
      await new Promise((r) => requestAnimationFrame(r));
    }
  }

  /** Show the view: the card (or why it failed) pops in, at its final shape. */
  private reveal(): void {
    this.fit();
    document.documentElement.classList.remove("cam-starting");
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
    this.qr?.close();
    this.qr = null;
    this.video.srcObject = null;
    for (const w of Object.values(this.work)) w.canvas.width = w.canvas.height = 0;
    this.work = {};
    this.workMode = null;
    this.trial = { gpu: [], cpu: [] };
    this.el["cam"]!.hidden = true;
    document.documentElement.classList.remove("cam-on", "cam-starting");
    if (fromButton && (history.state as { camera?: boolean } | null)?.camera) history.back();
  }

  /** The focal length the frames are analysed with, now. */
  focal(): FrameFocal | null {
    const v = this.video;
    if (!v.videoWidth) return null;
    return frameFocal(this.deps.settings(), this.current()?.label ?? null, { width: v.videoWidth, height: v.videoHeight });
  }

  // ---- the stream ----

  /** The size to open a camera at (camera/stream.ts). */
  private streamChoice(label: string) {
    const s = this.deps.settings();
    return streamFor({
      liveSize: s.liveSize,
      focalKnown: frameFocal(s, label || null, { width: 3024, height: 4032 }) !== null,
      imageCapture: typeof (globalThis as { ImageCapture?: unknown }).ImageCapture === "function",
      refused: s.fullStreamCameras.includes(label),
    });
  }

  /** The open stream at the size its camera should have (camera/stream.ts): a
   * camera opened before its name was known (the first tap of a visit) opens at
   * full size, and is made smaller here once it is known. Left as it is if the
   * camera will not change. */
  private async resize(): Promise<void> {
    const track = this.track;
    if (!track) return;
    const want = this.streamChoice(track.label);
    this.stats.stream = want.why;
    if (want.fast === this.fast) return;
    try {
      await track.applyConstraints({ ...want.size, ...FRAME_RATE });
      this.fast = want.fast;
    } catch (e) {
      this.stats.stream = `${want.why}, not applied: ${(e as Error).name}`;
    }
  }

  private current(): CameraInfo | null {
    if (!this.track) return null;
    return { deviceId: this.track.getSettings().deviceId ?? "", label: this.track.label };
  }

  private async attach(stream: MediaStream): Promise<void> {
    const fast = this.fast;
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
      if (!fast && longMax && Math.max(set.width ?? 0, set.height ?? 0) < 0.9 * longMax) {
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

  private async switchTo(camera: CameraInfo): Promise<void> {
    if (!camera.deviceId || camera.deviceId === this.current()?.deviceId) return;
    const before = this.current();
    this.switching = true;
    try {
      const choice = this.streamChoice(camera.label);
      this.fast = choice.fast;
      const stream = await request(camera, choice.size);
      if (!this.open) {
        stop(stream);
        return;
      }
      await this.attach(stream);
    } catch {
      // Keep the camera that is open (iOS may have stopped it: then open it again).
      if (this.track?.readyState === "ended" && before) {
        try {
          this.fast = false;
          await this.attach(await request(before, FULL_STREAM));
        } catch (e) {
          this.fail("error", e as Error);
        }
      }
    } finally {
      this.switching = false;
    }
    this.listed(await cameras());
  }

  /** The cameras found, for Settings (the camera is chosen there, not in the view). */
  private listed(devices: CameraInfo[]): void {
    this.rearCount = rearCameras(devices).filter((d) => d.deviceId).length;
    this.deps.camerasListed?.(devices);
  }

  // ---- frames ----

  private loop(gen: number): void {
    const v = this.video;
    let waiting = 0;
    const next = () => {
      if (gen !== this.gen) return;
      const id = ++waiting;
      // Whichever comes first for this wait; the other finds `waiting` moved on.
      const go = (now: number) => {
        if (id === waiting) tick(now);
      };
      if (v.requestVideoFrameCallback && !this.vfcStalled) {
        v.requestVideoFrameCallback((now) => go(now));
        setTimeout(() => {
          if (id !== waiting || gen !== this.gen || v.paused || v.readyState < 2 || this.shooting) return;
          this.vfcStalled = true;
          this.stats.loop = `animation frames (no video frame for ${VFC_STALL_MS} ms)`;
          requestAnimationFrame(go);
        }, VFC_STALL_MS);
      } else {
        requestAnimationFrame(go);
      }
    };
    const tick = (now: number) => {
      if (gen !== this.gen) return;
      next();
      this.frame(now);
    };
    this.stats.loop = "requestVideoFrameCallback" in v && !this.vfcStalled ? "video frames" : this.stats.loop || "animation frames";
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
    // A pairing code (camera/qr-scan.ts), looked for while no chain is tracked.
    // A frame it takes is the overlay's frame skipped, under the same budget;
    // asked before the region, so that it does not use up a step of the scan.
    if (this.qr && !this.session.tracking && this.qr.frame(v, now)) {
      const cost = this.qr.grabMs;
      this.stats.qr = `${this.qr.frames + 1} × ${cost.toFixed(0)}+${this.qr.lastMs.toFixed(0)} ms`;
      this.lastWork = now;
      this.gap = cost / BUSY_SHARE - cost;
      return;
    }
    // Only the part of the frame round the chain while it is tracked, drawn
    // finely; else the whole frame, or a quarter of it (overlay/region.ts).
    const r = this.session.region(v.videoWidth, v.videoHeight);
    // In a worker when the browser can (camera/grab.ts); a check frame is
    // grabbed here as well, and the two copies compared.
    let check: VideoFrame | null = null;
    // The check skips whole frames: a whole frame came out right on an iPhone
    // while regions came out shifted, so only a region tests the worker's drawing.
    if (this.grab && (this.grab.mode === "worker" || r.kind !== "full") && this.grab.wants()) {
      const frame = this.videoFrame();
      if (frame && this.grab.mode === "worker") return this.toWorker(frame, now, r, t0);
      check = frame;
    }
    // The trial takes turns, so that both canvases see the same mix of regions.
    const mode = this.workMode ?? (this.trial.gpu.length <= this.trial.cpu.length ? "gpu" : "cpu");
    const work = this.workCanvas(mode, r.workWidth, r.workHeight);
    // The whole frame, scaled and shifted so that the region lands on the
    // canvas: not drawImage's source rectangle, which Safari on iPhone took in
    // the camera sensor's landscape, not the portrait video's (owner's
    // screenshots, 2026-11-19: overlays off the chain, a chain in plain view
    // not found). A whole-frame draw is what the page did before regions.
    work.g.drawImage(v, -r.x / r.scaleX, -r.y / r.scaleY, v.videoWidth / r.scaleX, v.videoHeight / r.scaleY);
    const n = r.workWidth * r.workHeight;
    if (this.gray.length < n) this.gray = new Uint8Array(n);
    const gray = toGray(work.g.getImageData(0, 0, r.workWidth, r.workHeight).data, r.workWidth, r.workHeight, this.gray,
      this.session.shape.channel);
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
    // The frame may come the sensor's way round (Safari on iPhone: 4032×3024 for
    // a 3024×4032 video): the worker turns it upright, and the check finds which way.
    const turns = turnsFor(frame.displayWidth, frame.displayHeight, v.videoWidth, v.videoHeight,
      (frame as unknown as { rotation?: unknown }).rotation);
    if (!turns || (grab.turnsKnown() && !turns.includes(grab.turn))) {
      this.grabFailed(grab, `VideoFrame ${frame.displayWidth}×${frame.displayHeight}, video ${v.videoWidth}×${v.videoHeight}`);
      frame.close();
      return null;
    }
    grab.setTurns(turns);
    return frame;
  }

  /** The worker draws `frame` from `r` too: is its copy the page's (`page`)? */
  private check(frame: VideoFrame, r: Region, page: Uint8Array): void {
    const grab = this.grab!, gen = this.gen;
    grab.run(frame, r, this.session!.shape, true).then((reply) => {
      if (gen !== this.gen || this.grab !== grab) return;
      const worker = reply.gray ?? new Uint8Array(0);
      grab.checked(compareGrays(page, worker), grayDifference(page, worker), grayCorrelation(page, worker));
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
      this.stats.canvas = grab.canvas;
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
    const now = performance.now();
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
    this.stats.arrow = null;
    if (vw && vh) {
      // The video is shown whole (object-fit: contain): what the still will hold.
      const k = Math.min(cw / vw, ch / vh), ox = (cw - vw * k) / 2, oy = (ch - vh * k) / 2;
      const X = (x: number) => ox + (x + 0.5) * k, Y = (y: number) => oy + (y + 0.5) * k;
      if (this.stats.shows === "chain") {
        // No video: the picture's edge (what the still will hold), and every dot or rim
        // found, faint, so that the view is seen to be looking before it finds the chain.
        g.strokeStyle = "rgba(255,255,255,.3)";
        g.lineWidth = 1 * dpr;
        g.strokeRect(ox + 0.5 * dpr, oy + 0.5 * dpr, vw * k - dpr, vh * k - dpr);
        g.fillStyle = "rgba(255,255,255,.35)";
        for (const b of view.blobs) {
          g.beginPath();
          g.arc(X(b.x), Y(b.y), Math.max(1.5 * dpr, (b.d / 2) * k), 0, 2 * Math.PI);
          g.fill();
        }
      }
      const phase = view.state.phase;
      // The pins' rings say how it goes: green once steady, amber while something needs fixing, white while looking.
      const color = phase === "steady" || phase === "capture" ? "#2fc46e" : phase === "adjusting" ? "#ffa025" : "rgba(255,255,255,.85)";
      // Only the chain is drawn. Blobs off it (glints, holes, clutter, each
      // found in one frame and lost in the next) were drawn as faint rings
      // until 2026-11-19 and only flickered: the pin count and the coach's
      // message already say when pins are missing.
      const chain = followChain(view.chain, view.blobs);
      if (chain.length) {
        const r = Math.max(5 * dpr, (median(view.blobs.map((b) => b.d)) / 2) * k + 2 * dpr);
        // The links: a broad white line, faint, under the rings.
        g.save();
        g.globalAlpha = 0.35;
        g.lineWidth = Math.max(4 * dpr, 1.1 * r);
        g.strokeStyle = "#fff";
        g.lineJoin = g.lineCap = "round";
        g.beginPath();
        chain.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))));
        g.stroke();
        g.restore();
        // Lock-on: a ring round each pin closing in on it as the hold fills,
        // meeting the pin's own ring as the still is taken (overlay/animate.ts).
        const hold = this.holdShown.update(phase === "steady" ? view.state.hold : phase === "capture" || phase === "captured" ? 1 : 0, now);
        if (hold > 0) {
          const R = lockRadius(r, hold), tick = Math.min(R - r, 0.35 * r);
          g.save();
          g.globalAlpha = lockAlpha(hold);
          g.strokeStyle = "#2fc46e";
          g.lineWidth = (1.5 + hold) * dpr;                // thin while wide, firmer as it closes
          g.lineCap = "round";
          for (const [x, y] of chain) {
            const px = X(x), py = Y(y);
            g.beginPath();
            g.arc(px, py, R, 0, 2 * Math.PI);
            // Four ticks pointing in, a sight's: shown while the ring is still wide.
            if (tick > 1) {
              for (let q = 0; q < 4; q++) {
                const a = Math.PI / 4 + (q * Math.PI) / 2, ca = Math.cos(a), sa = Math.sin(a);
                g.moveTo(px + ca * R, py + sa * R);
                g.lineTo(px + ca * (R - tick), py + sa * (R - tick));
              }
            }
            g.stroke();
          }
          g.restore();
        }
        g.lineWidth = 2.5 * dpr;
        g.strokeStyle = color;
        for (const [x, y] of chain) {
          g.beginPath();
          g.arc(X(x), Y(y), r, 0, 2 * Math.PI);
          g.stroke();
        }
        let cx = 0, cy = 0;
        for (const [x, y] of chain) {
          cx += X(x) / chain.length;
          cy += Y(y) / chain.length;
        }
        this.centre = [cx, cy];
        const arrow = view.state.arrow;
        let target: { dir: Point; along: number; length: number } | null = null;
        if (arrow && phase !== "capture" && phase !== "captured") {
          this.stats.arrow = arrow.dir;
          // The way to move the phone, just past the chain on that side (clear
          // of its pins); longer the more it is tilted.
          const [ux, uy] = arrow.dir;
          let reach = 0;
          for (const [x, y] of chain) reach = Math.max(reach, (X(x) - cx) * ux + (Y(y) - cy) * uy);
          const short = Math.min(vw, vh) * k;
          const grow = Math.min(1, Math.max(0, (arrow.tiltDeg - COACH.arrowDeg.enter) / 18));
          const length = short * (0.12 + 0.08 * grow);
          target = { dir: arrow.dir, along: reach + r + 10 * dpr + length / 2, length };
        }
        this.drawArrowEased(g, target, now, color, dpr, { ox, oy, w: vw * k, h: vh * k });
      } else {
        this.holdShown.update(0, now);
        this.drawArrowEased(g, null, now, color, dpr, { ox, oy, w: vw * k, h: vh * k });
      }
    }
    this.render(view);
  }

  /** The arrow eased toward `target` (overlay/animate.ts), round the chain's
   * centre and kept inside the picture; fading out where it was when there is none. */
  private drawArrowEased(g: CanvasRenderingContext2D, target: { dir: Point; along: number; length: number } | null, now: number,
    color: string, dpr: number, pic: { ox: number; oy: number; w: number; h: number }): void {
    const pose = this.arrowShown.update(target, now);
    if (!pose || !this.centre) return;
    const [ux, uy] = pose.dir, [cx, cy] = this.centre;
    const pad = 14 * dpr + pose.length / 2;
    const ax = Math.min(Math.max(cx + ux * pose.along, pic.ox + pad), pic.ox + pic.w - pad);
    const ay = Math.min(Math.max(cy + uy * pose.along, pic.oy + pad), pic.oy + pic.h - pad);
    g.save();
    g.globalAlpha = pose.alpha;
    drawArrow(g, ax, ay, pose.dir, pose.length, color, dpr);
    g.restore();
  }

  private render(view: OverlayView): void {
    const st = view.state;
    const say = this.shooting ? { title: "Taking the photo…", detail: "Keep still for a moment." } : sayFor(st, { engineReady: this.deps.engineReady() });
    this.say(say.title, say.detail);
    const pins = pinsText(st);
    if (pins !== this.shown.pins) {
      // "13 / 13 pins", the number found in green.
      this.shown.pins = pins;
      const m = /^(\d+)(.*)$/.exec(pins);
      this.el["cam-count"]!.replaceChildren(...(m ? [Object.assign(document.createElement("i"), { textContent: m[1] }), m[2]!] : [pins]));
    }
    if (st.phase !== this.shown.phase) this.el["cam-live"]!.dataset.phase = this.shown.phase = st.phase;
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
    st.size = `${v.videoWidth}×${v.videoHeight}${this.fast ? " (fast)" : ""}`;
    // What broke a steady hold (the coach's problems that lasted), most first: what made the trace turn yellow.
    const breaks = Object.entries(this.session?.coach.breaks ?? {}).sort((a, b) => b[1] - a[1]);
    st.breaks = breaks.map(([p, n]) => `${p} ${n}`).join(", ");
    const sharp = this.latest?.blobs.length ? median(this.latest.blobs.map((b) => b.sharpness)) : null;
    // One item a line, always the same lines (a dash until there is a value).
    // Only the last, where frames are grabbed, may wrap (its reason can be
    // long): last, so its wrapping moves no other line.
    // First, what a screenshot sent by someone else must say: the browser, the OS, the phone and the build.
    st.browser = `${browserInfo(navigator.userAgent, this.ua)} · build ${import.meta.env.BUILD_ID}`;
    const lines = [st.browser, st.camera || "camera",
      st.size + (this.fast || !st.stream ? "" : `, ${st.stream}`) + (st.shows === "chain" ? ", video hidden" : st.shows === "grey" ? ", video grey" : ""), focal,
      `frame ${st.grabMs.length ? `${median(st.grabMs.slice(-30)).toFixed(0)}+${median(st.blobMs.slice(-30)).toFixed(0)} ms` : "–"}`
        + (st.loop.startsWith("animation") ? `, ${st.loop}` : ""),
      `region ${st.region || "–"}`,
      `analysis ${st.analyseMs.length ? `${Math.round(median(st.analyseMs))} ms` : "–"}`,
      `sharpness ${sharp !== null ? sharp.toFixed(2) : "–"}`,
      `holds broken ${st.breaks || "–"}`,
      `qr ${st.qr || "–"}`,
      this.grab?.describe() ?? "grab –"];
    const text = lines.join("\n");
    if (text !== this.shown.info) {
      this.shown.info = text;
      this.el["cam-info"]!.replaceChildren(...lines.map((l) => Object.assign(document.createElement("div"), { textContent: l })));
    }
    this.el["cam-teach"]!.hidden = !!f;
  }

  private say(title: string, detail: string): void {
    if (title !== this.shown.title) this.el["cam-say-title"]!.textContent = this.shown.title = title;
    if (detail !== this.shown.detail) this.el["cam-say-detail"]!.textContent = this.shown.detail = detail;
  }

  /** Whether the view can take a still now (not while one is being taken). */
  private setReady(on: boolean): void {
    this.el["cam-live"]!.dataset.ready = String(on);
  }

  private fail(why: "insecure" | "error", e: Error | null): void {
    if (!this.open) return;
    if (this.stream) stop(this.stream);
    this.stream = this.track = null;
    this.el["cam-live"]!.hidden = true;
    this.el["cam-fail"]!.hidden = false;
    this.reveal();
    const [title, text] = why === "insecure"
      ? ["This page can't use the camera here.", "The live camera needs the site's https:// address."]
      : e?.name === "NotAllowedError" || e?.name === "SecurityError"
        ? ["The camera wasn't allowed.", "To allow it: on iPhone, the aA button in the address bar, Website Settings, Camera; on Android, the icon left of the address, Permissions, Camera. Then open the camera again."]
        : ["The camera couldn't be opened.", `${e?.name ?? "Error"}: ${e?.message ?? ""}. Another app may be using it.`];
    this.el["cam-fail-title"]!.textContent = title;
    this.el["cam-fail-text"]!.textContent = text;
  }

  // ---- the still ----

  /** Take the still now: from the coach after a steady hold. */
  async shoot(): Promise<void> {
    if (this.shooting || !this.track || !this.open) return;
    this.shooting = true;
    this.setReady(false);
    this.say("Taking the photo…", "Keep still for a moment.");
    this.el["cam"]!.classList.add("flash");
    const v = this.video;
    const frameSize = { width: v.videoWidth, height: v.videoHeight };
    const frameChain = this.latest?.chain.map((p) => [p[0], p[1]] as Point) ?? [];
    const camera = this.track.label;
    const facing = this.track.getSettings().facingMode ?? null;
    const t0 = performance.now();
    try {
      // The fast stream (camera/stream.ts) goes back to the full sensor for the
      // still: on the owner's iPhone (build 25) takePhoto gave no photo with it,
      // and the still was the 1440×1920 video frame.
      const wasFast = this.fast;
      if (wasFast) await this.fullForStill();
      this.stats.takePhotoError = null;
      const photo = await this.takePhoto();
      if (!photo && wasFast) this.deps.fullStream?.(camera);
      const stamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, "").replace("T", "-");
      let still: Still;
      if (photo) {
        if (wasFast) await this.checkStill(photo, camera);
        const ext = photo.type === "image/png" ? "png" : "jpg";
        still = { file: new File([photo], `photo-${stamp}.${ext}`, { type: photo.type || "image/jpeg" }), kind: "photo",
          camera, facing, frameFocal: null, frameChain, frameSize, ms: performance.now() - t0 };
      } else {
        const blob = await this.grabFrame();
        // The chain as the live view saw it, in this (possibly larger) frame's pixels.
        const k = Math.max(v.videoWidth, v.videoHeight) / Math.max(frameSize.width, frameSize.height);
        still = { file: new File([blob], `frame-${stamp}.jpg`, { type: "image/jpeg" }), kind: "frame",
          camera, facing, frameFocal: this.focal(), frameChain: k === 1 ? frameChain : frameChain.map(([x, y]) => [x * k, y * k] as Point),
          frameSize: { width: v.videoWidth, height: v.videoHeight }, ms: performance.now() - t0,
          ...(this.stats.takePhotoError ? { why: this.stats.takePhotoError } : {}) };
      }
      this.stats.still = `${still.kind} ${still.file.size} B in ${Math.round(still.ms)} ms`;
      this.close();
      this.deps.still(still);
    } catch (e) {
      this.shooting = false;
      this.setReady(true);
      this.say("Couldn't take the photo.", (e as Error).message);
    } finally {
      this.el["cam"]?.classList.remove("flash");
    }
  }

  /** The fast stream back at the full sensor for the still: the photo sizes
   * read again, and the video given up to 2 s to come at the new size. Left
   * as it is if the camera will not change. */
  private async fullForStill(): Promise<void> {
    const track = this.track, v = this.video;
    if (!track) return;
    try {
      await track.applyConstraints({ ...FULL_STREAM, ...FRAME_RATE });
      this.fast = false;
      const full = 0.9 * Math.max(FULL_STREAM.width.ideal, FULL_STREAM.height.ideal);
      for (let t = 0; t < 2000 && Math.max(v.videoWidth, v.videoHeight) < full; t += 50) await new Promise((r) => setTimeout(r, 50));
      if (this.capture) this.caps = this.capture.getPhotoCapabilities().catch(() => null);
    } catch (e) {
      this.stats.stream = `full size for the still refused: ${(e as Error).name}`;
    }
  }

  /** Taken while the stream was fast: did the still keep its full size? If not,
   * this camera opens at full size from now on (the still is used all the same). */
  private async checkStill(photo: Blob, camera: string): Promise<void> {
    try {
      const size = stillSize(new Uint8Array(await photo.slice(0, 256 * 1024).arrayBuffer()));
      this.stats.stillSize = size ? `${size.width}×${size.height}` : null;
      if (size && smallStill(size, FULL_STREAM)) this.deps.fullStream?.(camera);
    } catch {
      /* not read: nothing learned */
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

/** The video inputs, with labels once the camera has been allowed (Chrome remembers that; Safari asks each visit). */
export async function cameras(): Promise<CameraInfo[]> {
  try {
    return (await navigator.mediaDevices.enumerateDevices())
      .filter((d) => d.kind === "videoinput")
      .map((d) => ({ deviceId: d.deviceId, label: d.label }));
  } catch {
    return [];
  }
}

function request(camera: CameraInfo | null, size: StreamSize): Promise<MediaStream> {
  const which = camera?.deviceId ? { deviceId: { exact: camera.deviceId } } : { facingMode: { ideal: "environment" } };
  return navigator.mediaDevices.getUserMedia({ video: { ...which, ...size, ...FRAME_RATE }, audio: false });
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

/** An arrow `length` long centred on (cx, cy), pointing along the unit vector
 * `dir`, in `color` over a dark edge that keeps it visible on a light table. */
function drawArrow(g: CanvasRenderingContext2D, cx: number, cy: number, dir: Point, length: number, color: string, dpr: number): void {
  const [ux, uy] = dir;
  const head = Math.min(length * 0.45, 22 * dpr), wing = head * 0.6;
  const tx = cx + (ux * length) / 2, ty = cy + (uy * length) / 2;        // the tip
  const bx = tx - ux * head, by = ty - uy * head;                          // where the head meets the shaft
  const path = () => {
    g.beginPath();
    g.moveTo(cx - (ux * length) / 2, cy - (uy * length) / 2);
    g.lineTo(bx, by);
    g.moveTo(bx - uy * wing, by + ux * wing);
    g.lineTo(tx, ty);
    g.lineTo(bx + uy * wing, by - ux * wing);
    g.closePath();
  };
  g.save();
  g.lineJoin = "round";
  g.lineCap = "round";
  path();
  g.strokeStyle = "rgba(0,0,0,.55)";
  g.lineWidth = 9 * dpr;
  g.stroke();
  path();
  g.strokeStyle = color;
  g.fillStyle = color;
  g.lineWidth = 5 * dpr;
  g.stroke();
  g.fill();
  g.restore();
}

function median(v: number[]): number {
  if (!v.length) return 0;
  const s = [...v].sort((a, b) => a - b);
  return s[s.length >> 1]!;
}

function blankStats(): CameraStats {
  return { camera: "", size: "", grabMs: [], blobMs: [], regions: { full: 0, tile: 0, track: 0 }, region: "", grab: "", canvas: "", pageMs: [], frames: 0, skipped: 0, analysed: 0, analyseMs: [], focal: "", work: "", qr: "",
    imageCapture: false, takePhotoMs: null, takePhotoError: null, still: null, stillSize: null, stream: "", closedBy: null, browser: "", arrow: null, breaks: "", loop: "", shows: "grey" };
}
