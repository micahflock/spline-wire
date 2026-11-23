import { describe, expect, it } from "vitest";
import { COACH, Coach, check, extent } from "./coach";
import type { FrameReading, Point } from "./types";

/** A 13-pin chain across the middle of a 1920×1080 frame, straight-on, sharp and still. */
function reading(t: number, change: Partial<FrameReading> = {}): FrameReading {
  const chain: Point[] = Array.from({ length: 13 }, (_, k) => [360 + 100 * k, 540 + 60 * Math.sin(k)]);
  return { t, width: 1920, height: 1080, chain, pinsExpected: 13, missing: 0, tiltDeg: 5, tiltDir: [1, 0],
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

  it("starts the hold again after a problem that lasts, and after a stall", () => {
    const coach = new Coach();
    run(coach, 0, 500);
    expect(run(coach, 600, 800, { motion: 3 }).phase).toBe("steady");          // not lasted PROBLEM_MS yet
    expect(coach.update(reading(900, { motion: 3 })).phase).toBe("adjusting");   // 300 ms: broken
    expect(coach.breaks).toEqual({ moving: 1 });
    expect(run(coach, 1000, 1600).phase).toBe("steady");          // 600 ms of good readings: not yet
    expect(coach.update(reading(1700)).phase).toBe("capture");
    coach.reset();
    run(coach, 2000, 2500);
    expect(coach.update(reading(2500 + COACH.STALE_MS + 1)).hold).toBe(0);
  });

  it("rides out a bad reading or two while holding (no flicker), but takes the still only on a good one", () => {
    const coach = new Coach();
    run(coach, 0, 300, {}, 25);
    // Every fourth reading at 40 a second has a problem, none lasting.
    const bad: Partial<FrameReading>[] = [{ motion: 3 }, { sharpness: 0.2 }, { tiltDeg: 40 }];
    let k = 0;
    for (let t = 325; t < 700; t += 25) {
      const s = coach.update(reading(t, t % 100 === 0 ? bad[k++ % 3] : {}));
      expect(s.phase).toBe("steady");
    }
    const flawed = coach.update(reading(700, { motion: 3 }));        // the hold is long enough, but not on this reading
    expect(flawed).toMatchObject({ phase: "steady" });
    expect(flawed.hold).toBeLessThan(1);
    expect(coach.update(reading(725)).phase).toBe("capture");
    expect(coach.breaks).toEqual({});
  });

  it("takes the still only after a whole hold with every pin: a pin short starts the countdown again, still green", () => {
    const coach = new Coach();
    run(coach, 0, 500, {}, 25);
    const short = coach.update(reading(525, { chain: reading(0).chain.slice(0, 12) }));   // one reading a pin short
    expect(short).toMatchObject({ phase: "steady", hold: 0 });
    expect(coach.update(reading(550, { chain: [] })).phase).toBe("steady");                 // none at all, briefly
    let s = coach.update(reading(575));
    expect(s.phase).toBe("steady");
    for (let t = 600; t < 575 + COACH.HOLD_MS; t += 25) {
      s = coach.update(reading(t));
      expect(s.phase).toBe("steady");
    }
    expect(s.hold).toBeLessThan(1);
    expect(coach.update(reading(575 + COACH.HOLD_MS)).phase).toBe("capture");
    expect(coach.breaks).toEqual({});
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
    // The chain spans 1200 / 1080 of the short side: shrunk to 0.24, too far to start
    // the hold (minSpan.enter 0.26) but near enough to keep it (keep 0.22); to 0.20, too far.
    const shrink = (span: number) => reading(0).chain.map(([x, y]) => [960 + ((x - 960) * span * 1080) / 1200, 540 + ((y - 540) * span * 1080) / 1200] as Point);
    expect(check(reading(0, { chain: shrink(0.24) }), false)).toEqual(["too-far"]);
    expect(check(reading(0, { chain: shrink(0.24) }), true)).toEqual([]);
    expect(check(reading(0, { chain: shrink(0.2) }), true)).toEqual(["too-far"]);
    expect(check(reading(0, { chain: shrink(0.27) }), false)).toEqual([]);
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

  it("shows an arrow the way to move the phone once the readings agree, only while it is tilted", () => {
    const coach = new Coach();
    expect(coach.update(reading(0, { tiltDeg: 5, tiltDir: [0, 1] })).arrow).toBeNull();          // straight enough
    // Readings agreeing builds up: the second tilted reading in the same way shows it.
    let s = coach.update(reading(100, { tiltDeg: 20, tiltDir: [0, 1] }));
    expect(s.arrow).not.toBeNull();
    expect(s.arrow!.dir[1]).toBeCloseTo(1);
    expect(s.arrow!.tiltDeg).toBe(20);
    expect(s.phase).toBe("steady");                        // a nudge: 20 degrees does not block the shot
    // Between the two limits it stays (no blinking on the borderline)...
    expect(coach.update(reading(200, { tiltDeg: 10, tiltDir: [0, 1] })).arrow).not.toBeNull();
    // ...and goes below the looser one, with the chain, or with the tilt unknown.
    expect(coach.update(reading(300, { tiltDeg: 8, tiltDir: [0, 1] })).arrow).toBeNull();
    s = coach.update(reading(400, { tiltDeg: 20, tiltDir: [0, 1] }));
    expect(s.arrow).not.toBeNull();
    expect(coach.update(reading(500, { chain: [], tiltDeg: 20 })).arrow).toBeNull();
    coach.update(reading(600, { tiltDeg: 20, tiltDir: [0, 1] }));
    expect(coach.update(reading(700, { tiltDeg: null, tiltDir: null })).arrow).toBeNull();
    // One reading is not enough after a fresh start.
    expect(new Coach().update(reading(0, { tiltDeg: 30, tiltDir: [0, 1] })).arrow).toBeNull();
  });

  it("turns the arrow as fast in time whatever the rate of readings", () => {
    const turned = (step: number) => {
      const coach = new Coach();
      for (let t = 0; t <= 600; t += step) coach.update(reading(t, { tiltDeg: 30, tiltDir: [1, 0] }));
      let a = coach.current.arrow!;
      for (let t = 600 + step; t <= 800; t += step) a = coach.update(reading(t, { tiltDeg: 30, tiltDir: [0, 1] })).arrow!;
      return (Math.atan2(a.dir[1], a.dir[0]) * 180) / Math.PI;
    };
    expect(turned(25)).toBeCloseTo(turned(100), 0);           // 200 ms of the new way, at 40 or 10 readings a second
  });

  it("shows no arrow while the readings disagree", () => {
    const coach = new Coach();
    const dirs: Point[] = [[1, 0], [-1, 0], [0, 1], [0, -1], [0.7, 0.7], [-0.7, -0.7], [1, 0], [-1, 0]];
    dirs.forEach((d, k) => expect(coach.update(reading(100 * k, { tiltDeg: 30, tiltDir: d })).arrow).toBeNull());
  });

  it("turns the arrow gradually, so one reading off does not swing it", () => {
    // 30 degrees blocks the shot, so the coach keeps coaching throughout.
    const coach = new Coach();
    for (let t = 0; t <= 500; t += 100) coach.update(reading(t, { tiltDeg: 30, tiltDir: [1, 0] }));
    const a = coach.update(reading(600, { tiltDeg: 30, tiltDir: [0, 1] })).arrow!;
    const angle = (Math.atan2(a.dir[1], a.dir[0]) * 180) / Math.PI;
    expect(angle).toBeGreaterThan(10);
    expect(angle).toBeLessThan(45);                         // less than halfway toward the odd reading
    expect(Math.hypot(...a.dir)).toBeCloseTo(1);
    let s = a;
    for (let t = 700; t <= 2000; t += 100) s = coach.update(reading(t, { tiltDeg: 30, tiltDir: [0, 1] })).arrow!;
    expect(s.dir[1]).toBeGreaterThan(0.999);               // and settles on a steady new way
  });

  it("drops the arrow while taking the still", () => {
    const coach = new Coach();
    expect(coach.update(reading(0, { tiltDeg: 18, tiltDir: [1, 0] })).arrow).toBeNull();
    expect(coach.update(reading(100, { tiltDeg: 18, tiltDir: [1, 0] })).arrow).not.toBeNull();
    const s = run(coach, 200, 700, { tiltDeg: 18, tiltDir: [1, 0] });
    expect(s.phase).toBe("capture");
    expect(s.arrow).toBeNull();
    expect(coach.update(reading(800, { tiltDeg: 18, tiltDir: [1, 0] })).arrow).toBeNull();     // captured
  });
});
