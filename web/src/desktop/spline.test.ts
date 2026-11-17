import { describe, expect, it } from "vitest";
import fixture from "../test-data/measurement.json";
import type { Point } from "../relay/messages";
import { splineSamples } from "./spline";

function maxDiff(a: Point[], b: Point[]): number {
  expect(a.length).toBe(b.length);
  return Math.max(...a.map((p, i) => Math.max(Math.abs(p[0] - b[i]![0]), Math.abs(p[1] - b[i]![1]))));
}

describe("the spline through the points is Python's (contact.spline_samples)", () => {
  it("matches the samples in the measurement fixture", () => {
    const curve = fixture.document.curve as unknown as { points: Point[]; spline: Point[] };
    // Python fits the unrounded points and writes points and samples rounded to
    // 4 decimals, so they may differ by a few 0.00005 mm; the exact cases are below.
    expect(maxDiff(splineSamples(curve.points), curve.spline)).toBeLessThan(2e-4);
  });

  // Expected values from: uv run python -c "...spline_samples(np.array(pts), per_segment=4)", rounded to 6 decimals.
  it("matches Python for three points (SciPy's parabola case)", () => {
    const pts: Point[] = [[0, 0], [10, 5], [18, -2]];
    const python: Point[] = [[0, 0], [2.588387, 2.387715], [5.080093, 4.021797], [7.475116, 4.902246],
      [9.773457, 5.029063], [11.975116, 4.402246], [14.080093, 3.021797], [16.088387, 0.887715], [18, -2]];
    expect(maxDiff(splineSamples(pts, 4), python)).toBeLessThan(1e-6);
  });

  it("matches Python for four points", () => {
    const pts: Point[] = [[0, 0], [9.5, 3], [17, 10], [21, 19.5]];
    const python: Point[] = [[0, 0], [2.571716, 0.331505], [5.055223, 0.976017], [7.432164, 1.910189],
      [9.68418, 3.110672], [11.792913, 4.554118], [13.740003, 6.21718], [15.507094, 8.07651], [17.075826, 10.108759],
      [18.427841, 12.290581], [19.544781, 14.598627], [20.408286, 17.009549], [21, 19.5]];
    expect(maxDiff(splineSamples(pts, 4), python)).toBeLessThan(1e-6);
  });

  it("starts and ends on the end points and keeps short or degenerate input as it is", () => {
    const curve = fixture.document.curve as unknown as { points: Point[] };
    const out = splineSamples(curve.points, 12);
    expect(out).toHaveLength(12 * (curve.points.length - 1) + 1);
    expect(maxDiff([out[0]!, out.at(-1)!], [curve.points[0]!, curve.points.at(-1)!])).toBeLessThan(1e-9);
    expect(splineSamples([[1, 2], [3, 4]])).toEqual([[1, 2], [3, 4]]);
    expect(splineSamples([[1, 2], [1, 2], [3, 4]])).toEqual([[1, 2], [1, 2], [3, 4]]);
  });
});
