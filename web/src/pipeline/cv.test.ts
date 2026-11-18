// cv.ts against cv2 itself (golden/cv.json.gz, web/scripts/golden_cv.py):
// bit for bit where OpenCV's arithmetic is integer or float32, to rounding
// in the ellipse fits, which solve linear systems.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  adaptiveMeanThreshold, brightMask, connectedComponents, fitEllipse, fitEllipseDirect, gaussianBlur8,
  halveArea8, outerContour, sampleLinear, type Gray,
} from "./cv";

interface ImageRec {
  width: number; height: number; image: string; blur08: string; blur10: string; half: string;
  adaptive: string; bright: string; labels: string; stats: number[][];
  contours: { label: number; points: number[]; ellipse?: number[] }[];
}

interface SmallRec {
  width: number; height: number; image: string; blur08: string; blur10: string; adaptive: string; bright: string;
  masks: { mask: string; labels: string; stats: number[][] }[];
}

const golden = JSON.parse(
  gunzipSync(readFileSync(new URL("./golden/cv.json.gz", import.meta.url))).toString("utf-8"),
) as {
  opencv: string; images: ImageRec[]; direct: { points: string; ellipse: number[] }[]; small: SmallRec[];
  remap: { x: string; y: string; value: string };
};

function bytes(s: string): Uint8Array {
  return Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
}
const u8 = (s: string) => bytes(s);
const i32 = (s: string) => new Int32Array(bytes(s).buffer);
const f32 = (s: string) => new Float32Array(bytes(s).buffer);

/** How many of two arrays' entries differ. */
function differing(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length);
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++;
  return n;
}

/** Ellipses equal to float32 rounding, the angle modulo 180 degrees. */
function expectSameEllipse(got: { cx: number; cy: number; width: number; height: number; angle: number }, want: number[]) {
  const [cx, cy, w, h, ang] = want as [number, number, number, number, number];
  const tol = (v: number) => 4e-6 * Math.max(1, Math.abs(v));
  expect(Math.abs(got.cx - cx)).toBeLessThanOrEqual(tol(cx));
  expect(Math.abs(got.cy - cy)).toBeLessThanOrEqual(tol(cy));
  expect(Math.abs(got.width - w)).toBeLessThanOrEqual(tol(w));
  expect(Math.abs(got.height - h)).toBeLessThanOrEqual(tol(h));
  if (Math.abs(h - w) > 1e-3 * h) {        // a circle's angle means nothing
    const d = (((got.angle - ang) % 180) + 180) % 180;
    expect(Math.min(d, 180 - d)).toBeLessThan(1e-3);
  }
}

describe(`cv.ts against cv2 ${golden.opencv}`, () => {
  golden.images.forEach((rec, n) => {
    const img: Gray = { width: rec.width, height: rec.height, data: u8(rec.image) };
    describe(`image ${n} (${rec.width}x${rec.height})`, () => {
      it("GaussianBlur, both sigmas, exactly", () => {
        expect(differing(gaussianBlur8(img, 0.8).data, u8(rec.blur08))).toBe(0);
        expect(differing(gaussianBlur8(img, 1.0).data, u8(rec.blur10))).toBe(0);
      });
      it("half-size INTER_AREA, exactly", () => {
        expect(differing(halveArea8(img).data, u8(rec.half))).toBe(0);
      });
      it("adaptive and bright masks, exactly", () => {
        expect(differing(adaptiveMeanThreshold(img, 25, -4), u8(rec.adaptive))).toBe(0);
        expect(differing(brightMask(img, 25), u8(rec.bright))).toBe(0);
      });
      it("connected components: labels and stats, exactly", () => {
        const c = connectedComponents(u8(rec.adaptive), rec.width, rec.height);
        expect(c.count).toBe(rec.stats.length);
        expect(differing(c.labels, i32(rec.labels))).toBe(0);
        rec.stats.forEach(([x, y, w, h, area], k) => {
          expect([c.x[k], c.y[k], c.w[k], c.h[k], c.area[k]]).toEqual([x, y, w, h, area]);
        });
      });
      it("outer contours exactly, and fitEllipse on them", () => {
        const c = connectedComponents(u8(rec.adaptive), rec.width, rec.height);
        expect(rec.contours.length).toBeGreaterThan(0);
        let degenerate = 0;
        for (const want of rec.contours) {
          const pts = outerContour(c, rec.width, rec.height, want.label);
          expect(Array.from(pts)).toEqual(want.points);
          if (!want.ellipse) continue;
          const e = fitEllipse(pts);
          if (want.ellipse[2]! < 1) {
            // A line one pixel wide: OpenCV finds the system singular and jitters
            // the points at random. Either way the "ellipse" is a sliver, which
            // detection rejects (8 px at least across).
            expect(e.width).toBeLessThan(2);
            degenerate++;
          } else {
            expectSameEllipse(e, want.ellipse);
          }
        }
        expect(degenerate).toBeLessThan(rec.contours.length / 4);
      });
    });
  });

  describe("small and odd sizes (narrower than the kernel and the box)", () => {
    for (const rec of golden.small) {
      it(`${rec.width}x${rec.height}`, () => {
        const img: Gray = { width: rec.width, height: rec.height, data: u8(rec.image) };
        expect(differing(gaussianBlur8(img, 0.8).data, u8(rec.blur08))).toBe(0);
        expect(differing(gaussianBlur8(img, 1.0).data, u8(rec.blur10))).toBe(0);
        expect(differing(adaptiveMeanThreshold(img, 25, -4), u8(rec.adaptive))).toBe(0);
        expect(differing(brightMask(img, 25), u8(rec.bright))).toBe(0);
        for (const m of rec.masks) {
          const c = connectedComponents(u8(m.mask), rec.width, rec.height);
          expect(c.count).toBe(m.stats.length);
          expect(differing(c.labels, i32(m.labels))).toBe(0);
          // Label 0's stats when no pixel is background are OpenCV's placeholders; detection never reads them.
          const first = c.area[0] === 0 ? 1 : 0;
          for (let k = first; k < c.count; k++) {
            expect([c.x[k], c.y[k], c.w[k], c.h[k], c.area[k]]).toEqual(m.stats[k]);
          }
        }
      });
    }
  });

  it("fitEllipseDirect on noisy ellipses", () => {
    for (const d of golden.direct) expectSameEllipse(fitEllipseDirect(f32(d.points)), d.ellipse);
  });

  it("remap (INTER_LINEAR, BORDER_REPLICATE), exactly", () => {
    const rec = golden.images[0]!;
    const img = { width: rec.width, height: rec.height, data: Float32Array.from(u8(rec.image)) };
    const xs = f32(golden.remap.x), ys = f32(golden.remap.y), want = f32(golden.remap.value);
    const got = Array.from(xs, (x, i) => sampleLinear(img, x, ys[i]!));
    expect(differing(got, want)).toBe(0);
  });
});
