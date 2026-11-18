// The scorer against Python's (web/scripts/golden_scoring.py): the plaque
// poses' pins exactly, and every score to 1e-6 mm. The fits end where their
// solvers stop (SciPy's least_squares for the circle, iterative closest
// points for the placement), so the numbers agree to the solvers' tolerance,
// not bit for bit.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import type { Vec2 } from "./numeric";
import { fitCircle, headline, score, type Score } from "./scoring";
import { GAUGES, gaugeEdge, PLAQUES, plaquePins } from "./targets";
import type { ChainSpec } from "./types";

interface Case { target: string; pins: Vec2[]; sign: number | null; gaps: number[]; spec: ChainSpec; score: Score }
const golden = JSON.parse(gunzipSync(readFileSync(new URL("./golden/scoring.json.gz", import.meta.url))).toString("utf-8")) as {
  plaques: { pose: string; n: number; pins: Vec2[] }[];
  cases: Case[];
};

function numbers(x: unknown, prefix = ""): [string, number][] {
  if (typeof x === "number") return [[prefix, x]];
  if (x && typeof x === "object") return Object.entries(x).flatMap(([k, v]) => numbers(v, `${prefix}.${k}`));
  return [];
}

describe("targets", () => {
  it("lists every gauge and plaque pose", () => {
    expect(GAUGES.map((g) => g.name)).toContain("convex-r40");
    expect(PLAQUES.map((p) => p.name)).toEqual(["plaque-s", "plaque-tight"]);
  });
  it("places a plaque's pins as Python does", () => {
    for (const { pose, n, pins } of golden.plaques) {
      const ours = plaquePins(pose, 13.0, n);
      expect(ours.length).toBe(n);
      for (let i = 0; i < n; i++) {
        expect(ours[i]![0]).toBeCloseTo(pins[i]![0]!, 9);
        expect(ours[i]![1]).toBeCloseTo(pins[i]![1]!, 9);
      }
    }
  });
  it("samples a gauge's edge on its shape", () => {
    for (const g of GAUGES.filter((g) => g.radius_mm !== undefined)) {
      for (const [x, y] of gaugeEdge(g, 1.0)) expect(Math.hypot(x, y)).toBeCloseTo(g.radius_mm!, 9);
    }
  });
});

describe("scores against Python's", () => {
  it.each(golden.cases.map((c, i) => [`${i} ${c.target}`, c] as const))("%s", (_, c) => {
    const ours = score(c.target, c.pins, c.spec, c.sign, c.gaps);
    const want = numbers(c.score), got = new Map(numbers(ours));
    // first_pin is left out: the S plaque is point-symmetric, so a chain cut
    // short fits either end equally, and which wins is a rounding tie.
    for (const [key, value] of want.filter(([k]) => k !== ".first_pin")) {
      expect(got.has(key), key).toBe(true);
      expect(Math.abs(got.get(key)! - value), `${key}: ${got.get(key)} vs ${value}`).toBeLessThan(1e-6);
    }
    expect(headline(ours)).toBeCloseTo(c.score.kind === "plaque" ? c.score.max_error_mm : c.score.inner.max_mm, 6);
  });
});

describe("fitCircle", () => {
  it("is geometric", () => {
    const pts: Vec2[] = Array.from({ length: 9 }, (_, i) => [3 + 50 * Math.cos(0.15 * i), -2 + 50 * Math.sin(0.15 * i)]);
    const { center, radius } = fitCircle(pts);
    expect(radius).toBeCloseTo(50, 9);
    expect(center[0]).toBeCloseTo(3, 9);
  });
});
