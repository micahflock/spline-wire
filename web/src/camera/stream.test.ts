import { describe, expect, it } from "vitest";
import { FAST_STREAM, FULL_STREAM, jpegSize, pngSize, smallStill, stillSize, streamFor } from "./stream";

describe("the live camera's stream size", () => {
  const ok = { liveSize: "fast" as const, focalKnown: true, imageCapture: true, refused: false };

  it("is fast once the focal length is known, with takePhoto stills", () => {
    expect(streamFor(ok)).toMatchObject({ size: FAST_STREAM, fast: true });
  });

  it("is the full sensor otherwise, and says why", () => {
    expect(streamFor({ ...ok, liveSize: "full" })).toMatchObject({ size: FULL_STREAM, fast: false, why: "full size (Settings)" });
    expect(streamFor({ ...ok, imageCapture: false }).why).toMatch(/video frame/);
    expect(streamFor({ ...ok, focalKnown: false }).why).toMatch(/focal length/);
    expect(streamFor({ ...ok, refused: true }).why).toMatch(/came back small/);
  });

  it("tells a small still from a full one by the long side", () => {
    expect(smallStill({ width: 4032, height: 3024 }, FULL_STREAM)).toBe(false);
    expect(smallStill({ width: 3024, height: 4032 }, FULL_STREAM)).toBe(false);
    expect(smallStill({ width: 1920, height: 1440 }, FULL_STREAM)).toBe(true);
  });
});

describe("jpegSize", () => {
  /** A JPEG's start: SOI, an APP1 of `app` bytes, a DQT, then SOF (baseline or progressive). */
  const jpeg = (w: number, h: number, app = 100, sof = 0xc0) => {
    const seg = (marker: number, body: number[]) => [0xff, marker, (body.length + 2) >> 8, (body.length + 2) & 255, ...body];
    return Uint8Array.from([0xff, 0xd8, ...seg(0xe1, Array(app).fill(7)), ...seg(0xdb, Array(65).fill(1)), 0xff, 0xff,
      ...seg(sof, [8, h >> 8, h & 255, w >> 8, w & 255, 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]), ...seg(0xda, [1, 2, 3])]);
  };

  it("reads the frame header past the other segments", () => {
    expect(jpegSize(jpeg(4032, 3024))).toEqual({ width: 4032, height: 3024 });
    expect(jpegSize(jpeg(1920, 1440, 40000, 0xc2))).toEqual({ width: 1920, height: 1440 });
  });

  it("gives null for what is not a JPEG, or is cut before its frame header", () => {
    expect(jpegSize(Uint8Array.from([0x89, 0x50, 0x4e, 0x47]))).toBeNull();
    expect(jpegSize(jpeg(4032, 3024).slice(0, 60))).toBeNull();
  });
});

describe("pngSize and stillSize", () => {
  it("reads a PNG's header, and either kind of still", () => {
    const png = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52,
      0, 0, 0x0f, 0xc0, 0, 0, 0x0b, 0xd0, 8, 2, 0, 0, 0]);
    expect(pngSize(png)).toEqual({ width: 4032, height: 3024 });
    expect(stillSize(png)).toEqual({ width: 4032, height: 3024 });
    expect(pngSize(Uint8Array.from([0xff, 0xd8]))).toBeNull();
  });
});
