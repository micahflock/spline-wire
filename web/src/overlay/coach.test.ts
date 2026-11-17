import { describe, expect, it } from "vitest";
import { COACH, Coach, check, extent } from "./coach";
import type { FrameReading, Point } from "./types";

/** A 13-pin chain across the middle of a 1920×1080 frame, straight-on, sharp and still. */
function reading(t: number, change: Partial<FrameReading> = {}): FrameReading {
  const chain: Point[] = Array.from({ length: 13 }, (_, k) => [360 + 100 * k, 540 + 60 * Math.sin(k)]);
  return { t, width: 1920, height: 1080, chain, pinsExpected: 13, missing: 0, tiltDeg: 5,
    sharpness: 0.8, motion: 0.05, ...change };
}

function run(coach: Coach, from: number, to: number, change: Partial<FrameReading> = {}, step = 100) {
  let s = coach.current;
  for (let t = from; t <= to; t += step) s = coach.update(reading(t, change));
  return s;
}

describe("coach", () => {
  it("takes the still after a steady hold, once", () => {
    const coach = new Coach();
    expect(coach.update(reading(0)).phase).toBe("steady");
    const mid = coach.update(reading(350));
    expect(mid.phase).toBe("steady");
    expect(mid.hold).toBeCloseTo(0.5);
    expect(run(coach, 400, 700).phase).toBe("capture");
    expect(coach.update(reading(800)).phase).toBe("captured");
    coach.reset();
    expect(coach.update(reading(900)).phase).toBe("steady");
  });

  it("says what is wrong, most important first, and only what has lasted", () => {
    const coach = new Coach();
    const bad = { tiltDeg: 40, motion: 2 };
    expect(coach.update(reading(0, bad))).toMatchObject({ phase: "adjusting", problems: [] });
    expect(run(coach, 100, 300, bad).problems).toEqual(["tilted", "moving"]);
    expect(run(coach, 400, 700, { chain: [] })).toMatchObject({ phase: "searching", problems: ["no-chain"] });
    const part = reading(0).chain.slice(0, 9);
    expect(run(coach, 800, 1100, { chain: part })).toMatchObject({ problems: ["pins-missing"], pinsFound: 9 });
  });

  it("starts the hold again after a problem, and after a stall", () => {
    const coach = new Coach();
    run(coach, 0, 500);
    expect(coach.update(reading(600, { motion: 3 })).phase).toBe("adjusting");
    expect(run(coach, 700, 1300).phase).toBe("steady");          // 600 ms of good readings: not yet
    expect(coach.update(reading(1400)).phase).toBe("capture");
    coach.reset();
    run(coach, 2000, 2500);
    expect(coach.update(reading(2500 + COACH.STALE_MS + 1)).hold).toBe(0);
  });

  it("does not flicker on the borderline: keeping is easier than entering", () => {
    const edge = { tiltDeg: 27 };
    expect(check(reading(0, edge), false)).toEqual(["tilted"]);
    expect(check(reading(0, edge), true)).toEqual([]);
    const coach = new Coach();
    run(coach, 0, 300);
    expect(run(coach, 400, 700, edge).phase).toBe("capture");
  });

  it("names the right problem for a chain at the edge, far away, bridged or blurred", () => {
    const shifted = reading(0).chain.map(([x, y]) => [x - 350, y] as Point);
    expect(check(reading(0, { chain: shifted }), false)).toEqual(["at-edge"]);
    const small = reading(0).chain.map(([x, y]) => [960 + (x - 960) / 4, 540 + (y - 540) / 4] as Point);
    expect(check(reading(0, { chain: small }), false)).toEqual(["too-far"]);
    expect(check(reading(0, { chain: reading(0).chain.slice(1), missing: 1 }), false)).toEqual(["pins-missing"]);
    expect(check(reading(0, { sharpness: 0.1 }), false)).toEqual(["blurred"]);
    expect(check(reading(0, { tiltDeg: null, motion: null }), false)).toEqual([]);
    expect(extent(reading(0)).span).toBeCloseTo(1200 / 1080);
  });

  it("nudges toward straight-on without blocking the shot", () => {
    const coach = new Coach();
    const s = run(coach, 0, 700, { tiltDeg: 18 });
    expect(s).toMatchObject({ phase: "capture", straighter: true });
  });
});
