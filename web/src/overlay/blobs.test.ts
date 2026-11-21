import { describe, expect, it } from "vitest";
import { BLOBS, detectBlobs, largest, localThreshold, rescaleSharpness, shapeOf, thresholds, toGray, workSize } from "./blobs";
import { analyse } from "../pipeline/live";
import { specFromDict } from "../pipeline/chain";
import { fullRegion, RegionTracker } from "./region";
import { SNAP_SPEC, snapScene, VIDEO } from "./snap-frames";
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
  rim = 225, link = 40, hole = 20) {
  const x0 = Math.min(...pins.map((p) => p[0])) - linkR - 1, x1 = Math.max(...pins.map((p) => p[0])) + linkR + 1;
  const y0 = Math.min(...pins.map((p) => p[1])) - linkR - 1, y1 = Math.max(...pins.map((p) => p[1])) + linkR + 1;
  return (x: number, y: number): number | null => {
    if (x < x0 || x > x1 || y < y0 || y > y1) return null;
    for (const [px, py] of pins) {
      const r = Math.hypot(x - px, y - py);
      if (r <= inner * outerR) return hole;
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

  it("finds dark rims on light links when the chain says so (fiducial_shade dark), and only then", () => {
    const dark = shapeOf({ ...SNAP, fiducial_shade: "dark" });
    expect(dark).toEqual({ inner: shapeOf(SNAP).inner, dark: true });
    for (const table of [60, 200]) {
      const frame = scene(640, 360, table, ringChain(SNAP_PINS, 8, dark.inner, 8, 4.55, 30, 215, 235));
      const rings = detectBlobs(frame, 1, dark);
      expect(rings.length).toBe(13);
      expect(matched(rings, SNAP_PINS)).toBeLessThan(0.4);
      const asLight = detectBlobs(frame, 1, shapeOf(SNAP));                 // read as light rims: none on the pins
      expect(asLight.filter((b) => SNAP_PINS.some(([x, y]) => Math.hypot(b.x - x, b.y - y) < 2)).length).toBe(0);
    }
  });

  it("does not take dots, or dark specks on a light table, for the snap-fit chain's rims", () => {
    const shape = shapeOf(SNAP);
    expect(detectBlobs(scene(640, 360, 128, chain(PINS, 11, 18)), 1, shape)).toEqual([]);   // dots, as rings: none
    const specks = (x: number, y: number) => {
      for (const [px, py] of SNAP_PINS) if (Math.hypot(x - px, y - py) <= 6.4) return 40;     // a rim's hole, no rim
      return null;
    };
    expect(detectBlobs(scene(640, 360, 128, specks), 1, shape)).toEqual([]);   // no bright band round them
  });

  it("finds the rims below a grained table's hundreds of look-alikes, the largest candidates being tested first", () => {
    // The owner's wooden table (2026-11-21): its grain gave more than
    // BLOBS.maxCandidates components in each mask above the chain, and the first
    // that many in row order were all that was tested. Here: dark and light
    // specks, smaller than the holes, on a grid over the top of the frame.
    const shape = shapeOf(SNAP);
    const pins: [number, number][] = SNAP_PINS.map(([x, y]) => [x, y + 230]);
    const rings = ringChain(pins, 8, shape.inner, 8, 4.55);
    const grain = (x: number, y: number): number | null => {
      if (y > 330) return rings(x, y);
      const i = Math.round(x / 9), j = Math.round(y / 9);
      return Math.hypot(x - 9 * i, y - 9 * j) <= 2.6 ? ((i + j) % 2 ? 40 : 220) : null;
    };
    const frame = scene(640, 520, 128, grain);
    const boxes = (dark: boolean) => {
      const mask = localThreshold(frame, dark);
      let n = 0;
      for (let y = 0; y < 330; y += 9) for (let x = 0; x < 640; x += 9) if (mask[y * 640 + x]) n++;
      return n;
    };
    expect(boxes(true)).toBeGreaterThan(BLOBS.maxCandidates);           // more specks than are tested, in each mask
    expect(boxes(false)).toBeGreaterThan(BLOBS.maxCandidates);
    const found = detectBlobs(frame, 1, shape);
    expect(pins.filter(([x, y]) => found.some((b) => Math.hypot(b.x - x, b.y - y) < 0.4)).length).toBe(13);
  });

  it("tests the largest candidates, in row order", () => {
    const box = (cx: number, r: number) => ({ cx, cy: 0, rx: r, ry: r });
    const boxes = [box(0, 2), box(1, 5), box(2, 3), box(3, 5), box(4, 1)];
    expect(largest(boxes, 3).map((b) => b.cx)).toEqual([1, 2, 3]);
    expect(largest(boxes, 5)).toBe(boxes);
  });

  it("gives rings a larger work frame, since their rims are thin", () => {
    expect(workSize(4032, 3024).width).toBe(640);
    expect(workSize(4032, 3024, shapeOf(SNAP)).width).toBe(800);
    expect(shapeOf({ fiducial: "dot", fiducial_mm: 5 })).toEqual({ inner: 0 });
  });

  it("turns RGBA into gray, into a buffer it is given", () => {
    const rgba = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255]);
    expect([...toGray(rgba, 3, 1).data]).toEqual([255, 0, 76]);
    const into = new Uint8Array(8).fill(9);
    const g = toGray(rgba, 3, 1, into);
    expect([...g.data]).toEqual([255, 0, 76]);
    expect(g.data.buffer).toBe(into.buffer);
  });

  it("maps a region's blobs back to the video frame: its corner, and a scale across and down", () => {
    const frame = scene(640, 360, 128, chain(PINS, 11, 18));
    const plain = detectBlobs(frame, 1);
    const blobs = detectBlobs(frame, { x: 1000, y: 2000, scaleX: 2, scaleY: 1.5 });
    expect(blobs.length).toBe(plain.length);
    blobs.forEach((b, i) => {
      // work pixel centre c of a region from x0 at s video px per work px: x0 + (c + 0.5)·s − 0.5
      expect(b.x).toBeCloseTo(1000 + (plain[i]!.x + 0.5) * 2 - 0.5, 9);
      expect(b.y).toBeCloseTo(2000 + (plain[i]!.y + 0.5) * 1.5 - 0.5, 9);
      expect(b.d).toBeCloseTo(plain[i]!.d * 1.75, 9);
      expect(b.sharpness).toBe(plain[i]!.sharpness);
    });
  });

  it("reads sharpness in the pixels of the frame it is told to (a region drawn finer than the full frame)", () => {
    const frame = scene(640, 360, 128, chain(PINS, 11, 18), 4);
    const own = detectBlobs(frame, { x: 0, y: 0, scaleX: 2, scaleY: 2 });
    // Drawn at 2 video px per work px, read as if at 5: edges 2.5 times narrower.
    const asFull = detectBlobs(frame, { x: 0, y: 0, scaleX: 2, scaleY: 2, refScale: 5 });
    own.forEach((b, i) => expect(asFull[i]!.sharpness).toBeCloseTo(Math.min(1, b.sharpness * 2.5), 9));
    expect(rescaleSharpness(1, 0.4)).toBe(1);
    expect(rescaleSharpness(0.3, 0.4)).toBeCloseTo(0.75, 9);
    expect(rescaleSharpness(0.3, 2)).toBeCloseTo(0.15, 9);
  });
});

describe("thresholds (both masks in one pass, buffers reused)", { timeout: 60_000 }, () => {
  /** thresholds()'s masks unpadded, checking the zero border on the way. */
  function unpadded(frame: GrayFrame, withDark: boolean) {
    const { bright, dark, stride } = thresholds(frame, withDark);
    const { width: w, height: h } = frame;
    expect(stride).toBe(w + 2);
    expect(bright.length).toBe((w + 2) * (h + 2));
    const cut = (m: Uint8Array) => {
      const out = new Uint8Array(w * h);
      let border = 0;
      for (let y = 0; y < h + 2; y++) {
        for (let x = 0; x < w + 2; x++) {
          const v = m[y * stride + x]!;
          if (x === 0 || y === 0 || x === w + 1 || y === h + 1) border += v;
          else out[(y - 1) * w + x - 1] = v;
        }
      }
      expect(border).toBe(0);                     // the zero border, all round
      return out;
    };
    return { bright: cut(bright), dark: dark ? cut(dark) : null };
  }

  /** Noise in a few gray levels (many ties with the window's mean), on a ramp. */
  function noise(w: number, h: number, seed: number, levels = 12): GrayFrame {
    let s = seed;
    const rand = () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
    const data = new Uint8Array(w * h);
    for (let i = 0; i < data.length; i++) data[i] = Math.min(255, Math.floor(rand() * levels) * Math.floor(256 / levels) + (i % w) / 8);
    return { width: w, height: h, data };
  }

  it("gives localThreshold's masks bit for bit, at every size, with the buffers reused", () => {
    const frames: GrayFrame[] = [
      scene(640, 360, 128, chain(PINS, 11, 18)), scene(640, 360, 235, ringChain(PINS, 8, 0.8, 8, 4.55)),
      noise(300, 200, 1), noise(25, 25, 2), noise(26, 31, 3), noise(24, 100, 4), noise(100, 7, 5), noise(13, 13, 6),
      noise(1, 40, 7), noise(40, 1, 8), noise(1, 1, 9), noise(700, 30, 10, 3), noise(37, 512, 11, 255),
      // smaller after larger: the reused buffers hold the larger frame's masks
      noise(200, 150, 12), noise(50, 40, 13),
    ];
    for (const frame of frames) {
      for (const withDark of [true, false]) {
        const got = unpadded(frame, withDark);
        expect(got.bright).toEqual(localThreshold(frame));
        if (withDark) expect(got.dark).toEqual(localThreshold(frame, true));
        else expect(got.dark).toBeNull();
      }
    }
  });

  it("finds the same blobs as before on a frame that follows a larger one", () => {
    const pins: [number, number][] = Array.from({ length: 13 }, (_, k) => [70.4 + 42 * k, 180.3 + 40 * Math.sin(k * 0.6)]);
    const big = scene(640, 360, 60, ringChain(pins, 8, shapeOf(SNAP).inner, 8, 4.55));
    const small = scene(330, 260, 60, ringChain(pins.slice(0, 6), 8, shapeOf(SNAP).inner, 8, 4.55));
    const first = detectBlobs(small, 1, shapeOf(SNAP));
    detectBlobs(big, 1, shapeOf(SNAP));
    expect(detectBlobs(small, 1, shapeOf(SNAP))).toEqual(first);
    expect(first.length).toBe(6);
  });
});

describe("the live view's reach with the snap-fit chain (13 pins, a 3024×4032 video)", { timeout: 120_000 }, () => {
  const shape = shapeOf(SNAP_SPEC);
  const old = fullRegion(VIDEO.width, VIDEO.height, 800);     // what the camera view drew until 2026-11-19
  for (const span of [0.3, 0.22]) {
    for (const [name, table] of [["a mid-gray", 128], ["a light", 200], ["a dark", 60]] as const) {
      it(`finds every pin at span ${span} on ${name} table, in the region round the chain`, () => {
        const sc = snapScene(span, table);
        const tracker = new RegionTracker(shape, SNAP_SPEC.n_pins);
        tracker.next(VIDEO.width, VIDEO.height);
        tracker.chainFound(sc.pins, sc.pins.map(() => sc.rim), 0);
        const r = tracker.next(VIDEO.width, VIDEO.height);
        expect(r.kind).toBe("track");
        const blobs = detectBlobs(sc.draw(r), r, shape);
        // Every pin, in video pixels, within 0.3 px of the truth.
        for (const [x, y] of sc.pins) expect(Math.min(...blobs.map((b) => Math.hypot(b.x - x, b.y - y)))).toBeLessThan(0.3);
        // A look-alike or two (this scene's dark gap between two rims, half a pitch
        // from each) is left to the ordering, which takes the 13 pins and nothing else.
        expect(blobs.length).toBeLessThanOrEqual(15);
        const a = analyse(blobs.map((b) => [b.x, b.y]), blobs.map((b) => b.d), [VIDEO.width, VIDEO.height], specFromDict(SNAP_SPEC), null);
        expect(a.order.length).toBe(13);
        expect(a.gaps).toEqual([]);
        for (const i of a.order) expect(Math.min(...sc.pins.map(([x, y]) => Math.hypot(blobs[i]!.x - x, blobs[i]!.y - y)))).toBeLessThan(0.3);
        // The rims drawn about REGIONS.target (18) work px across, in a region far smaller than the old full frame.
        expect(sc.rim / r.scaleX).toBeGreaterThan(17);
        expect(r.workWidth * r.workHeight).toBeLessThan((old.workWidth * old.workHeight) / 4);
      });
    }
  }

  it("where the 800 px full frame lost them (the limit this replaced)", () => {
    const at = (span: number, table: number) => {
      const sc = snapScene(span, table);
      const blobs = detectBlobs(sc.draw(old), old, shape);
      return sc.pins.filter(([x, y]) => blobs.some((b) => Math.hypot(b.x - x, b.y - y) < 0.3 * b.d)).length;
    };
    expect(at(0.3, 128)).toBeLessThan(13);
    expect(at(0.3, 200)).toBeLessThan(13);
    expect(at(0.22, 128)).toBeLessThan(13);
  });
});
