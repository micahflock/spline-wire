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
    let stream = await request(first).catch(() => request(null));
    devices = await cameras();
    const best = chooseCamera(devices, remembered, avoid);
    const current = stream.getVideoTracks()[0]?.getSettings().deviceId;
    if (best?.deviceId && best.deviceId !== current) {
      stop(stream);
      stream = await request(best);
    }
    this.stream = stream;
    const track = stream.getVideoTracks()[0]!;
    const IC = (globalThis as { ImageCapture?: new (t: MediaStreamTrack) => ImageCaptureLike }).ImageCapture;
    this.capture = IC ? new IC(track) : null;
    this.video.srcObject = stream;
    await this.video.play().catch(() => {});
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

function stop(stream: MediaStream): void {
  for (const t of stream.getTracks()) t.stop();
}
