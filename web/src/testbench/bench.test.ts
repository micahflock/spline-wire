// The test page's logic: a photo measured and scored as Python scores it,
// the focal length chosen as the phone page chooses it, the verdicts, the
// names and the report `splinewire score --report` reads.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { readGrayPng } from "../pipeline/golden/png";
import type { PlaqueScore } from "../pipeline/scoring";
import type { ChainSpec } from "../pipeline/types";
import {
  chooseFocal, isTarget, specFor, measureAndScore, photoName, phoneSlug, report, REPORT_SCHEMA, reportText, targetSide,
  testId, verdict, type TestRecord,
} from "./bench";

const golden = JSON.parse(gunzipSync(readFileSync(new URL("../pipeline/golden/scoring.json.gz", import.meta.url))).toString("utf-8")) as {
  photo: { focal_35mm: number; runs: Record<string, { pins_found: number; tilt_deg: number; score: PlaqueScore }> };
  cases: { spec: ChainSpec }[];
};
const spec = golden.cases[0]!.spec;              // the snap-fit chain, as Python loads it

describe("measureAndScore", () => {
  const gray = readGrayPng(new URL("../pipeline/golden/photo-plaque-s.png", import.meta.url));

  it("scores the S plaque as Python does, with EXIF and estimated focal lengths", () => {
    const r = measureAndScore(gray, spec, "plaque-s", { source: "exif", f35: golden.photo.focal_35mm });
    expect(r.pins_found).toBe(13);
    expect(r.tilt_deg).toBeCloseTo(golden.photo.runs.exif!.tilt_deg, 1);
    const main = r.focal.main.score as PlaqueScore, est = r.focal.estimated!.score as PlaqueScore;
    expect(main.max_error_mm).toBeCloseTo(golden.photo.runs.exif!.score.max_error_mm, 4);
    expect(est.max_error_mm).toBeCloseTo(golden.photo.runs.estimated!.score.max_error_mm, 3);
    expect(r.focal.main.focal_35mm).toBeCloseTo(26, 1);
    expect(verdict(r.focal.main, r.pins_found, 13).ok).toBe(true);
  });

  it("measures an object with no target, and refuses a photo with no chain", () => {
    const r = measureAndScore(gray, spec, "object", { source: "estimated", f35: null });
    expect(r.focal.main.score).toBeUndefined();
    expect(r.focal.estimated).toBeUndefined();
    const blank = { width: 400, height: 300, data: new Uint8Array(400 * 300).fill(200) };
    expect(() => measureAndScore(blank, spec, "plaque-s", { source: "exif", f35: 26 })).toThrow(/fiducials/);
  });
});

describe("the rest", () => {
  it("chooses the focal length as the phone page does", () => {
    expect(chooseFocal(24, 25, 26)).toEqual({ source: "exif", f35: 24 });
    expect(chooseFocal(null, 25, 26)).toEqual({ source: "remembered", f35: 25 });
    expect(chooseFocal(null, null, 26)).toEqual({ source: "saved", f35: 26 });
    expect(chooseFocal(null, null, null)).toEqual({ source: "estimated", f35: null });
  });

  it("measures a plaque at its printed pitch and the chain at its own", () => {
    expect(specFor(spec, "plaque-s", 13, "light", 12.98).pitch_mm).toBe(13.0);
    expect(specFor(spec, "convex-r25", 11, "dark", 12.98)).toMatchObject({ pitch_mm: 12.98, n_pins: 11, fiducial_shade: "dark" });
    expect(specFor(spec, "object", 13, "light", null).pitch_mm).toBe(spec.pitch_mm);
    expect(specFor(spec, "concave-r60", 13, "light", 1.298).pitch_mm).toBe(spec.pitch_mm);   // a slip: ignored
  });

  it("knows each target's side", () => {
    expect(targetSide("convex-r40")).toBe("inside");
    expect(targetSide("concave-r60")).toBe("outside");
    expect(targetSide("s-r60")).toBeNull();
    expect(targetSide("plaque-s")).toBeNull();
    expect(isTarget("straight") && isTarget("object") && !isTarget("nope")).toBe(true);
  });

  it("names tests and photos", () => {
    expect(phoneSlug("iPhone 16 Pro")).toBe("iphone16pro");
    expect(testId("Pixel 7", 4)).toBe("pixel7-004");
    expect(photoName("pixel7-004", "convex-r40", "image/jpeg")).toBe("pixel7-004-convex-r40.jpg");
    expect(photoName("a", "b", "image/heic")).toBe("a-b.heic");
  });

  it("judges against the pass marks", () => {
    const gauge = (max: number, radius: number) => ({ source: "exif" as const, focal_35mm: 24, score: {
      target: "convex-r40", kind: "gauge" as const,
      all: { points: 13, max_mm: max, rms_mm: max / 2, curve_max_mm: max },
      inner: { points: 11, max_mm: max, rms_mm: max / 2, curve_max_mm: max, radius_mm: 40 + radius, radius_error_mm: radius },
    } });
    expect(verdict(gauge(0.2, 0.1), 13, 13).ok).toBe(true);
    expect(verdict(gauge(0.6, 0.1), 13, 13).ok).toBe(false);
    expect(verdict(gauge(0.2, -0.7), 13, 13).ok).toBe(false);
    expect(verdict(gauge(0.2, 0.1), 12, 13)).toMatchObject({ ok: false, text: expect.stringContaining("12 of 13 pins") });
    expect(verdict({ source: "estimated", focal_35mm: null, error: "found 4 pins" }, 4, 13).ok).toBe(false);
  });

  it("writes the report splinewire score reads", () => {
    const t: TestRecord = {
      id: "iphone-001", photo: "iphone-001-plaque-s.jpg", created: "2026-11-18T12:00:00Z", phone: "iPhone", chain: "snap",
      shade: "dark", pins: 13, pitch_mm: 13, target: "plaque-s", route: "camera",
      conditions: { tilt: "20", surface: "white paper", light: "daylight", frame: "centre" },
      exif: { make: "Apple", model: "iPhone", focal_mm: 6.8, focal_35mm: 24 }, saved_focal_35mm: null, size: [4032, 3024],
    };
    const r = report([t], "0.7.0", "UA", "2026-11-18T12:00:00Z");
    expect(r.schema).toBe(REPORT_SCHEMA);
    const text = reportText(r);
    expect(text).toContain("iphone-001 plaque-s: not measured");
    expect(JSON.parse(text.slice(text.indexOf("{")))).toEqual(r);
  });
});
