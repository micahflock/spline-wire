import { describe, expect, it } from "vitest";
import { detectBlobs, shapeOf, toGray, workSize } from "./blobs";
import type { GrayFrame } from "./types";

/** A small scene drawn with 3×3 supersampling: discs and rounded bars on a flat table. */
function scene(width: number, height: number, table: number, draw: (x: number, y: number) => number | null, blur = 0): GrayFrame {
  const ss = 3;
  let data = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let j = 0; j < ss; j++) {
        for (let i = 0; i < ss; i++) sum += draw(x + (i + 0.5) / ss - 0.5, y + (j + 0.5) / ss - 0.5) ?? table;
      }
      data[y * width + x] = sum / (ss * ss);
    }
  }
  for (let pass = 0; pass < blur; pass++) {                      // repeated 3×3 box blur
    const next = new Float32Array(data.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0, n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx, yy = y + dy;
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) { sum += data[yy * width + xx]!; n++; }
          }
        }
        next[y * width + x] = sum / n;
      }
    }
    data = next;
  }
  return { width, height, data: Uint8Array.from(data, (v) => Math.round(v)) };
}

/** A chain: dark links (bars with round ends) between pins, a light dot on each pin. */
function chain(pins: [number, number][], dotR: number, linkR: number, dot = 230, link = 30) {
  const xs = pins.map((p) => p[0]), ys = pins.map((p) => p[1]);
  const x0 = Math.min(...xs) - linkR - 1, x1 = Math.max(...xs) + linkR + 1;
  const y0 = Math.min(...ys) - linkR - 1, y1 = Math.max(...ys) + linkR + 1;
  return (x: number, y: number): number | null => {
    if (x < x0 || x > x1 || y < y0 || y > y1) return null;
    for (const [px, py] of pins) if (Math.hypot(x - px, y - py) <= dotR) return dot;
    for (let k = 0; k + 1 < pins.length; k++) {
      const [ax, ay] = pins[k]!, [bx, by] = pins[k + 1]!;
      const t = Math.min(1, Math.max(0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)));
      if (Math.hypot(x - ax - t * (bx - ax), y - ay - t * (by - ay)) <= linkR) return link;
    }
    return null;
  };
}

/** The snap-fit chain's look (data/chain-snap.yaml): a thin light rim around a dark
 * hole, on the top edge of a head no wider than the rim: from above, the next
 * link's sleeve is hidden under it (radius linkR, here at most the rim's), and
 * the dark link shows outside the rim only where the arms (half-width armR,
 * 0.57 ring radii) leave it. */
function ringChain(pins: [number, number][], outerR: number, inner: number, linkR: number, armR = linkR,
  rim = 225, link = 40) {
  const x0 = Math.min(...pins.map((p) => p[0])) - linkR - 1, x1 = Math.max(...pins.map((p) => p[0])) + linkR + 1;
  const y0 = Math.min(...pins.map((p) => p[1])) - linkR - 1, y1 = Math.max(...pins.map((p) => p[1])) + linkR + 1;
  return (x: number, y: number): number | null => {
    if (x < x0 || x > x1 || y < y0 || y > y1) return null;
    for (const [px, py] of pins) {
      const r = Math.hypot(x - px, y - py);
      if (r <= inner * outerR) return 20;                 // the hole
      if (r <= outerR) return rim;
    }
    for (let k = 0; k + 1 < pins.length; k++) {
      const [ax, ay] = pins[k]!, [bx, by] = pins[k + 1]!;
      const t = Math.min(1, Math.max(0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)));
      if (Math.hypot(x - ax - t * (bx - ax), y - ay - t * (by - ay)) <= armR) return link;
    }
    for (const [px, py] of pins) if (Math.hypot(x - px, y - py) <= linkR) return link;
    return null;
  };
}

const SNAP = { fiducial: "ring", fiducial_mm: 8.79, ring_inner_mm: 7.07 };

const PINS: [number, number][] = Array.from({ length: 9 }, (_, k) => [70.3 + 60 * k, 180.6 + 50 * Math.sin(k * 0.8)]);

function matched(found: { x: number; y: number }[], pins: [number, number][]): number {
  return Math.max(...pins.map(([px, py]) => Math.min(...found.map((b) => Math.hypot(b.x - px, b.y - py)))));
}

describe("detectBlobs", { timeout: 60_000 }, () => {
  for (const [name, table] of [["a dark table", 60], ["a mid-gray table", 128], ["a white table", 235]] as const) {
    it(`finds every dot of a chain on ${name}, and nothing else`, () => {
      const blobs = detectBlobs(scene(640, 360, table, chain(PINS, 11, 18)));
      expect(blobs.length).toBe(PINS.length);
      expect(matched(blobs, PINS)).toBeLessThan(0.25);            // px
      for (const b of blobs) {
        expect(b.d).toBeGreaterThan(20);
        expect(b.d).toBeLessThan(24);
        expect(b.sharpness).toBeGreaterThan(0.9);
      }
    });
  }

  it("finds dots wider than the threshold window (they come out hollow) and small ones", () => {
    const big: [number, number][] = [[120, 180], [320, 170], [520, 190]];
    const found = detectBlobs(scene(640, 360, 128, chain(big, 40, 64)));
    expect(found.length).toBe(3);
    expect(matched(found, big)).toBeLessThan(0.3);
    const small: [number, number][] = Array.from({ length: 13 }, (_, k) => [100.5 + 14 * k, 180.2]);
    expect(detectBlobs(scene(640, 360, 128, chain(small, 3.5, 5.6))).length).toBe(13);
  });

  it("ignores bright things with no dark margin around them", () => {
    const glints = (x: number, y: number) =>
      Math.hypot(x - 200, y - 100) <= 9 ? 250 : Math.abs(x - 400) <= 12 && Math.abs(y - 200) <= 5 ? 250 : null;
    expect(detectBlobs(scene(640, 360, 236, glints))).toEqual([]);          // on a light table: no dark margin
    const paper = (x: number, y: number) => (x > 300 && x < 500 && y > 100 && y < 260 ? 240 : null);
    expect(detectBlobs(scene(640, 360, 90, paper))).toEqual([]);            // a big bright sheet
  });

  it("reports blur as lower sharpness", () => {
    const sharp = detectBlobs(scene(640, 360, 128, chain(PINS, 11, 18)));
    const soft = detectBlobs(scene(640, 360, 128, chain(PINS, 11, 18), 4));
    const softer = detectBlobs(scene(640, 360, 128, chain(PINS, 11, 18), 12));
    expect(soft.length).toBe(PINS.length);
    const med = (b: { sharpness: number }[]) => b.map((x) => x.sharpness).sort()[b.length >> 1]!;
    expect(med(sharp)).toBeGreaterThan(med(soft) + 0.2);
    expect(med(soft)).toBeGreaterThan(med(softer));
    expect(med(softer)).toBeLessThan(0.35);
    console.log(`sharpness: sharp ${med(sharp).toFixed(2)}, 4 blur passes ${med(soft).toFixed(2)}, 12 passes ${med(softer).toFixed(2)}`);
  });

  it("maps positions back to the frame the work frame was drawn from", () => {
    const { width, height, scale } = workSize(1920, 1080);
    expect([width, height, scale]).toEqual([640, 360, 3]);
    const blobs = detectBlobs(scene(640, 360, 128, chain(PINS, 11, 18)), scale);
    // work pixel centre c covers frame pixels 3c..3c+2, whose centre is 3c+1
    expect(matched(blobs, PINS.map(([x, y]) => [3 * x + 1, 3 * y + 1]))).toBeLessThan(0.75);
    expect(blobs[0]!.d).toBeGreaterThan(60);
  });

  it("is fast enough for every video frame", () => {
    const frame = scene(640, 360, 128, chain(PINS, 11, 18));
    detectBlobs(frame);
    const start = performance.now();
    for (let i = 0; i < 20; i++) detectBlobs(frame);
    const ms = (performance.now() - start) / 20;
    console.log(`detectBlobs 640x360: ${ms.toFixed(1)} ms`);
    expect(ms).toBeLessThan(30);
  });

  const SNAP_PINS: [number, number][] = Array.from({ length: 13 }, (_, k) => [70.4 + 42 * k, 180.3 + 40 * Math.sin(k * 0.6)]);
  for (const [name, table] of [["a dark table", 60], ["a light gray table", 200], ["white paper, brighter than the rims", 235]] as const) {
    it(`finds the snap-fit chain's rims as rings on ${name}, and does not take them for dots`, () => {
      const shape = shapeOf(SNAP);
      expect(shape.inner).toBeCloseTo(0.804, 3);
      // Nothing dark outside the rims but the arms.
      const frame = scene(640, 360, table, ringChain(SNAP_PINS, 8, shape.inner, 8, 4.55));
      const rings = detectBlobs(frame, 1, shape);
      expect(rings.length).toBe(13);
      expect(matched(rings, SNAP_PINS)).toBeLessThan(0.4);
      for (const r of rings) expect(r.d).toBeGreaterThan(14);
      expect(detectBlobs(frame, 1)).toEqual([]);                          // as dots: none
    });
  }

  it("does not take dots, or dark specks on a light table, for the snap-fit chain's rims", () => {
    const shape = shapeOf(SNAP);
    expect(detectBlobs(scene(640, 360, 128, chain(PINS, 11, 18)), 1, shape)).toEqual([]);   // dots, as rings: none
    const specks = (x: number, y: number) => {
      for (const [px, py] of SNAP_PINS) if (Math.hypot(x - px, y - py) <= 6.4) return 40;     // a rim's hole, no rim
      return null;
    };
    expect(detectBlobs(scene(640, 360, 128, specks), 1, shape)).toEqual([]);   // no bright band round them
  });

  it("gives rings a larger work frame, since their rims are thin", () => {
    expect(workSize(4032, 3024).width).toBe(640);
    expect(workSize(4032, 3024, shapeOf(SNAP)).width).toBe(800);
    expect(shapeOf({ fiducial: "dot", fiducial_mm: 5 })).toEqual({ inner: 0 });
  });

  it("turns RGBA into gray", () => {
    const g = toGray(new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255]), 3, 1);
    expect([...g.data]).toEqual([255, 0, 76]);
  });
});
