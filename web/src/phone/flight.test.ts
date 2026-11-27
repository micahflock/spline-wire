import { describe, expect, it } from "vitest";
import { EASE, cubicBezier, lerpBox, lerpRect } from "./flight";

describe("cubicBezier", () => {
  it("is linear for the linear curve and pinned at both ends", () => {
    const linear = cubicBezier(1 / 3, 1 / 3, 2 / 3, 2 / 3);
    for (const t of [0, 0.1, 0.5, 0.9, 1]) expect(linear(t)).toBeCloseTo(t, 5);
    expect(EASE(0)).toBe(0);
    expect(EASE(1)).toBe(1);
  });

  it("lies on the Bézier curve through (0,0), (x1,y1), (x2,y2), (1,1)", () => {
    const b = (s: number, p1: number, p2: number) => 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s ** 2 * p2 + s ** 3;
    for (const [x1, y1, x2, y2] of [[0.25, 0.1, 0.25, 1], [0.2, 0.8, 0.2, 1], [0.42, 0, 0.58, 1]] as const) {
      const ease = cubicBezier(x1, y1, x2, y2);
      for (let s = 0.05; s < 1; s += 0.05) expect(ease(b(s, x1, x2))).toBeCloseTo(b(s, y1, y2), 6);
    }
  });

  it("eases out: fast at first, never past the end, always forward", () => {
    expect(EASE(0.25)).toBeGreaterThan(0.5);
    let last = 0;
    for (let t = 0.01; t <= 1; t += 0.01) {
      const v = EASE(t);
      expect(v).toBeGreaterThanOrEqual(last - 1e-9);
      expect(v).toBeLessThanOrEqual(1 + 1e-9);
      last = v;
    }
  });
});

describe("lerpRect and lerpBox", () => {
  const a = { left: 16, top: 120, width: 360, height: 200 }, b = { left: 0, top: 0, width: 412, height: 600 };

  it("start at the first and end at the second", () => {
    expect(lerpRect(a, b, 0)).toEqual(a);
    expect(lerpRect(a, b, 1)).toEqual(b);
    const p = { x: 100, y: 50, w: 900, h: 500 }, q = { x: 0, y: 0, w: 4000, h: 3000 };
    const s = lerpBox(p, q, 0), e = lerpBox(p, q, 1);
    for (const k of ["x", "y", "w", "h"] as const) {
      expect(s[k]).toBeCloseTo(p[k], 9);
      expect(e[k]).toBeCloseTo(q[k], 9);
    }
  });

  it("zooms geometrically about a centre that moves straight", () => {
    const m = lerpBox({ x: 0, y: 0, w: 100, h: 100 }, { x: 300, y: 300, w: 400, h: 400 }, 0.5);
    expect(m.w).toBeCloseTo(200, 9);                       // halfway in zoom: 100 → 200 → 400
    expect(m.x + m.w / 2).toBeCloseTo(275, 9);             // centres 50 and 500
  });
});
