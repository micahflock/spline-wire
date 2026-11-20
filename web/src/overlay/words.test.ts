import { describe, expect, it } from "vitest";
import { followChain } from "./trace";
import type { CoachState } from "./types";
import { ARROW_DETAIL, PROBLEM_WORDS, pinsText, sayFor } from "./words";

const state = (s: Partial<CoachState>): CoachState =>
  ({ phase: "adjusting", problems: [], hold: 0, pinsFound: 11, pinsExpected: 13, straighter: false, arrow: null, ...s });
const ready = { engineReady: true };

describe("what the camera view says", () => {
  it("names the most important problem in a person's words", () => {
    expect(sayFor(state({ problems: ["too-far", "moving"] }), ready).title).toBe("Move closer");
    expect(sayFor(state({ problems: ["moving"] }), ready).title).toBe("Hold still");
    expect(sayFor(state({ problems: ["tilted"] }), ready).title).toBe("Tilt the phone flatter over the chain");
    expect(sayFor(state({ problems: ["pins-missing"] }), ready).title).toBe("Part of the chain is out of view");
    expect(sayFor(state({ problems: ["blurred"] }), ready).title).toBe("Out of focus");
    expect(sayFor(state({ phase: "searching", problems: ["no-chain"] }), ready).title).toBe("Point the camera at the chain");
    for (const w of Object.values(PROBLEM_WORDS)) expect(w.title.length).toBeLessThan(40);
  });

  it("says hold still while the ring fills, and what it is waiting for", () => {
    expect(sayFor(state({ phase: "steady", hold: 0.5 }), ready)).toEqual({ title: "Hold still…", detail: "" });
    expect(sayFor(state({ phase: "steady", straighter: true }), ready).detail).toMatch(/Straighter/);
    const arrow = { dir: [1, 0] as [number, number], tiltDeg: 15 };
    expect(sayFor(state({ phase: "steady", straighter: true, arrow }), ready).detail).toMatch(/arrow/);
    expect(sayFor(state({ phase: "capture" }), ready).title).toBe("Taking the photo…");
    expect(sayFor(state({ problems: [] }), ready).title).toBe("Hold still…");      // a problem not shown yet
    expect(sayFor(state({}), { engineReady: false }).title).toMatch(/engine/);
  });

  it("points to the arrow when the phone is tilted and there is one", () => {
    expect(sayFor(state({ problems: ["tilted"] }), ready)).toEqual(PROBLEM_WORDS.tilted);
    const arrow = { dir: [0, 1] as [number, number], tiltDeg: 30 };
    expect(sayFor(state({ problems: ["tilted"], arrow }), ready)).toEqual({ title: PROBLEM_WORDS.tilted.title, detail: ARROW_DETAIL });
    expect(sayFor(state({ problems: ["moving"], arrow }), ready)).toEqual(PROBLEM_WORDS.moving);
  });

  it("counts pins once a chain is seen", () => {
    expect(pinsText(state({}))).toBe("11 of 13 pins");
    expect(pinsText(state({ phase: "searching", pinsFound: 0 }))).toBe("");
  });
});

describe("the trace over the video", () => {
  it("follows the current frame's blobs, and keeps pins without one close by", () => {
    const chain: [number, number][] = [[0, 0], [100, 0], [200, 0], [300, 0]];
    const blob = (x: number, y: number) => ({ x, y, d: 10, contrast: 50, sharpness: 1 });
    const now = followChain(chain, [blob(8, 3), blob(108, 2), blob(250, 60), blob(500, 500)]);
    expect(now).toEqual([[8, 3], [108, 2], [200, 0], [300, 0]]);  // more than half a link from any pin
    expect(followChain([], [blob(0, 0)])).toEqual([]);
  });
});
