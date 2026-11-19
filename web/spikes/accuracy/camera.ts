// The test page's camera: the phone page's choice of rear camera (O10,
// camera/choose.ts) and its still, ImageCapture.takePhoto() at the largest
// size, which keeps the camera's EXIF (D16). No overlay: the test page is
// about the measurement, not the coaching. Where takePhoto is missing
// (Firefox) the still is the full video frame, without EXIF.

import { chooseCamera, labelled, type CameraInfo } from "../../src/camera/choose";

interface ImageCaptureLike {
  takePhoto(options?: { imageWidth?: number; imageHeight?: number }): Promise<Blob>;
  getPhotoCapabilities(): Promise<{ imageWidth?: { max?: number }; imageHeight?: { max?: number } }>;
}

const STREAM = { width: { ideal: 4032 }, height: { ideal: 3024 } };
/** How long the camera may take to answer before the page says so. */
const OPEN_MS = 10_000;
/** How long to wait for the preview to play: the still comes from the track, not the video. */
const PLAY_MS = 1_500;

export interface Shot {
  blob: Blob;
  kind: "photo" | "frame";
  camera: string;
  ms: number;
}

export class Camera {
  private stream: MediaStream | null = null;
  private capture: ImageCaptureLike | null = null;

  constructor(private readonly video: HTMLVideoElement) {}

  get open(): boolean {
    return this.stream !== null;
  }

  /** Open the rear camera the phone page would; returns its label. */
  async start(remembered: CameraInfo | null, avoid: ReadonlySet<string>): Promise<string> {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("the camera needs the site over HTTPS");
    }
    let devices = await cameras();
    const first = labelled(devices) ? chooseCamera(devices, remembered, avoid) : null;
    let stream = await answer(request(first).catch(() => request(null)));
    devices = await cameras();
    const best = chooseCamera(devices, remembered, avoid);
    const current = stream.getVideoTracks()[0]?.getSettings().deviceId;
    if (best?.deviceId && best.deviceId !== current) {
      stop(stream);
      stream = await answer(request(best));
    }
    this.stream = stream;
    const track = stream.getVideoTracks()[0]!;
    const IC = (globalThis as { ImageCapture?: new (t: MediaStreamTrack) => ImageCaptureLike }).ImageCapture;
    this.capture = IC ? new IC(track) : null;
    this.video.srcObject = stream;
    // iOS Safari does not play a video it is not showing, and play() can then
    // wait until the page is hidden and shown again (seen 2026-11-19: the page
    // hung on "Starting the camera…" until the person switched apps). The
    // page shows the video first; this only makes sure nothing waits on it.
    await Promise.race([this.video.play().catch(() => {}), sleep(PLAY_MS)]);
    return track.label;
  }

  async shoot(): Promise<Shot> {
    const track = this.stream?.getVideoTracks()[0];
    if (!track) throw new Error("the camera is not open");
    const t0 = performance.now();
    let blob: Blob | null = null;
    if (this.capture) {
      try {
        const caps = await this.capture.getPhotoCapabilities().catch(() => null);
        const w = caps?.imageWidth?.max, h = caps?.imageHeight?.max;
        blob = await this.capture.takePhoto(w && h ? { imageWidth: w, imageHeight: h } : undefined);
        if (!blob.size) blob = null;
      } catch {
        blob = null;
      }
    }
    const kind = blob ? "photo" : "frame";
    blob ??= await this.frame();
    const shot = { blob, kind, camera: track.label, ms: Math.round(performance.now() - t0) } as Shot;
    this.stop();
    return shot;
  }

  stop(): void {
    if (this.stream) stop(this.stream);
    this.stream = null;
    this.capture = null;
    this.video.srcObject = null;
  }

  private frame(): Promise<Blob> {
    const v = this.video;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    return new Promise((resolve, reject) => c.toBlob((b) => {
      c.width = c.height = 0;
      if (b) resolve(b);
      else reject(new Error("the video frame could not be saved"));
    }, "image/jpeg", 0.95));
  }
}

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

/** The stream, or an error once the camera has not answered in OPEN_MS; a
 * stream that arrives after that is stopped at once, so the camera light goes off. */
function answer(pending: Promise<MediaStream>): Promise<MediaStream> {
  return new Promise((resolve, reject) => {
    let late = false;
    const timer = setTimeout(() => {
      late = true;
      reject(new Error(`the camera did not answer in ${OPEN_MS / 1000} s`));
    }, OPEN_MS);
    pending.then((s) => {
      clearTimeout(timer);
      if (late) stop(s);
      else resolve(s);
    }, (e) => {
      clearTimeout(timer);
      if (!late) reject(e);
    });
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function stop(stream: MediaStream): void {
  for (const t of stream.getTracks()) t.stop();
}
