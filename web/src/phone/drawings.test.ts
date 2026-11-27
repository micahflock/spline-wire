import { describe, expect, it } from "vitest";
import { cardCrop, clampBox, curveDrawing, fitBox, smoothPath, stripLayout, stripSvg } from "./drawings";
import { TapFilter, type Timer } from "./taps";
import { ORING, SNAP } from "../drawing/test-looks";

describe("chain strip", () => {
  it("lays out one row while the pins fit", () => {
    const { pins, per } = stripLayout(13, 358);
    expect(per).toBe(16);
    expect(new Set(pins.map((p) => p[1])).size).toBe(1);
    expect(pins[12]![0] - pins[11]![0]).toBe(22);
  });

  it("wraps serpentine: the second row runs back right to left from under the first row's end", () => {
    const { pins, per } = stripLayout(20, 358);
    expect(pins[per]![0]).toBe(pins[per - 1]![0]);
    expect(pins[per]![1]).toBeGreaterThan(pins[per - 1]![1]);
    expect(pins[per + 1]![0]).toBeLessThan(pins[per]![0]);
  });

  it("draws the chosen chain: one rimmed head per pin of the snap-fit chain, ending in its empty sleeve", () => {
    const svg = stripSvg(SNAP, 5, 358, 4);
    expect(svg.match(/<circle/g)).toHaveLength(10);            // a head and its rim on each pin
    expect(svg.match(/class="sleeve"/g)).toHaveLength(1);
    expect(svg.match(/class="new"/g)?.length).toBeGreaterThan(0);
    expect(svg).toContain('aria-label="A chain of 5 pins"');
  });

  it("draws the O-ring chain as plates with a dot on each pin", () => {
    const svg = stripSvg(ORING, 5, 358);
    expect(svg.match(/<circle/g)).toHaveLength(5);
    expect(svg.match(/<line/g)).toHaveLength(4);
    expect(svg).not.toContain("sleeve");
  });

  it("is empty, at the default chain's height, before the presets are loaded", () => {
    expect(stripSvg(null, 13, 358)).not.toContain("<circle");
  });
});

describe("boxes", () => {
  it("fits a box to an aspect about its centre", () => {
    const [x, y, w, h] = fitBox([0, 0, 100, 10], 2, 0);
    expect([w, h]).toEqual([100, 50]);
    expect(y + h / 2).toBe(5);
    expect(x).toBe(0);
  });

  it("keeps a crop inside the photo at its aspect", () => {
    expect(clampBox([-50, 10, 200, 100], 1000, 800)).toEqual([0, 10, 200, 100]);
    const [x, y, w, h] = clampBox([0, 0, 2000, 1000], 1000, 800);
    expect([x, y, w, h]).toEqual([0, 0, 1000, 500]);
  });

  it("crops the card to the chain with a margin, at the card's aspect", () => {
    const [x, y, w, h] = cardCrop([[400, 300], [600, 300], [600, 400]], 10, { width: 4000, height: 3000 }, 1.8);
    expect(w / h).toBeCloseTo(1.8);
    expect(x).toBeLessThan(400 - 30);
    expect(x + w).toBeGreaterThan(600 + 30);
    expect(y).toBeLessThan(300);
  });
});

describe("drawings", () => {
  it("passes a smooth path through every point", () => {
    const d = smoothPath([[0, 0], [10, 0], [20, 10], [30, 10]]);
    expect(d.startsWith("M0 0")).toBe(true);
    expect(d.match(/C/g)).toHaveLength(3);
    expect(d.endsWith(" 30 10")).toBe(true);
  });

  it("draws a curve's chain and spline y up", () => {
    const svg = curveDrawing({ pins: [[0, 0], [13, 0], [26, 5]], spline: [[0, 2], [26, 7]] }, 1);
    expect(svg).toContain('cy="-5"');
    expect(svg.match(/<circle/g)).toHaveLength(3);
  });
});

describe("taps", () => {
  function fake() {
    const due: { fn: () => void; h: number }[] = [];
    let n = 0;
    const timer: Timer = { set: (fn) => (due.push({ fn, h: ++n }), n), clear: (h) => { const i = due.findIndex((d) => d.h === h); if (i >= 0) due.splice(i, 1); } };
    return { timer, fire: () => due.splice(0).forEach((d) => d.fn()) };
  }

  it("waits out the window before a single tap edits", () => {
    const t = fake(), singles: number[] = [];
    const taps = new TapFilter<string>((at) => singles.push(at.x), () => singles.push(-1), t.timer);
    taps.tap({ x: 5, y: 5 }, 10, "touch");
    expect(singles).toEqual([]);
    t.fire();
    expect(singles).toEqual([5]);
  });

  it("takes two taps close together as a double tap, with no edit", () => {
    const t = fake(), got: string[] = [];
    const taps = new TapFilter<string>(() => got.push("single"), () => got.push("double"), t.timer);
    taps.tap({ x: 5, y: 5 }, 10, "touch");
    taps.tap({ x: 12, y: 8 }, 10, "touch");
    t.fire();
    expect(got).toEqual(["double"]);
  });

  it("lets a tap far from the first go as two single taps", () => {
    const t = fake(), got: number[] = [];
    const taps = new TapFilter<string>((at) => got.push(at.x), () => got.push(-1), t.timer);
    taps.tap({ x: 5, y: 5 }, 10, "touch");
    taps.tap({ x: 500, y: 5 }, 10, "touch");
    expect(got).toEqual([5]);
    t.fire();
    expect(got).toEqual([5, 500]);
  });
});
