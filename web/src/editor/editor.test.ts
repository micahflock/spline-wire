import { describe, expect, it } from "vitest";
import type { PhotoView } from "../core/types";
import { EditHistory, describe as say, editsAfterTap, hitTest } from "./edits";
import { Gestures, wheelFactor, type PointerInput } from "./gestures";
import { chainBox, fit, fitWidth, keepPhotoInView, pinch, toPhoto, toScreen, zoomAt, zoomLimits, type ViewBox } from "./viewport";

const el = { width: 400, height: 300 };
const photo = { width: 4000, height: 3000 };
const limits = zoomLimits(photo, 20);

function view(): PhotoView {
  return {
    width: 4000, height: 3000, r: 20,
    pins: [{ x: 1000, y: 1000, r: 20, k: 0 }, { x: 1200, y: 1000, r: 20, k: 1 }, { x: 1400, y: 1000, r: 20, k: 2, edit: 0, snapped: true }],
    gaps: [], unused: [{ x: 3000, y: 2000, r: 18 }], curve: [], side_px: null,
    removed: [{ x: 1600, y: 1000, edit: 0 }],
    edits: { add: [[1401, 999]], remove: [[1600, 1000]] },
  };
}

describe("viewport", () => {
  it("converts between screen and photo pixels", () => {
    const box = { x: 100, y: 200, w: 800, h: 600 };
    expect(toPhoto(box, el, { x: 200, y: 150 })).toEqual({ x: 500, y: 500 });
    expect(toScreen(box, el, { x: 500, y: 500 })).toEqual({ x: 200, y: 150 });
  });

  it("zooms about the pointer and stays within limits", () => {
    const box = { x: 0, y: 0, w: 4000, h: 3000 };
    const z = zoomAt(box, el, { x: 100, y: 100 }, 0.5, limits);
    expect(toPhoto(z, el, { x: 100, y: 100 })).toEqual(toPhoto(box, el, { x: 100, y: 100 }));
    expect(z.w).toBe(2000);
    expect(zoomAt(box, el, { x: 0, y: 0 }, 1e-6, limits).w).toBe(120);      // 6 marker radii
    expect(zoomAt(box, el, { x: 0, y: 0 }, 100, limits).w).toBe(8000);      // twice the photo
  });

  it("pinches so the photo points under the fingers stay under them", () => {
    const start: ViewBox = { x: 0, y: 0, w: 4000, h: 3000 };
    const from: [{ x: number; y: number }, { x: number; y: number }] = [{ x: 100, y: 150 }, { x: 200, y: 150 }];
    const to: typeof from = [{ x: 50, y: 160 }, { x: 250, y: 160 }];        // spread to twice the distance, moved down
    const box = pinch(start, el, from, to, limits);
    expect(box.w).toBeCloseTo(2000);
    expect(box.h / box.w).toBeCloseTo(0.75);
    for (let i = 0; i < 2; i++) {
      const before = toPhoto(start, el, from[i]!), after = toPhoto(box, el, to[i]!);
      expect(after.x).toBeCloseTo(before.x);
      expect(after.y).toBeCloseTo(before.y);
    }
  });

  it("frames the chain in the element's shape and keeps the photo on screen", () => {
    const b = fit(chainBox(view()), el);
    expect(b.w / b.h).toBeCloseTo(4 / 3);
    expect(b.x).toBeLessThan(1000);
    expect(b.x + b.w).toBeGreaterThan(1600);
    const lost = keepPhotoInView({ x: 9000, y: -9000, w: 1000, h: 750 }, photo);
    expect(lost.x).toBe(3800);
    expect(lost.y).toBe(-600);
  });
});

describe("gestures", () => {
  function setup() {
    let box: ViewBox = { x: 0, y: 0, w: 4000, h: 3000 };
    const g = new Gestures(() => box, () => el, () => limits);
    const apply = (i: ReturnType<Gestures["pointerMove"]>) => {
      if (i.type === "view") box = i.box;
      return i;
    };
    const p = (id: number, x: number, y: number, t = 0, kind: PointerInput["kind"] = "touch"): PointerInput => ({ id, kind, x, y, t });
    return { g, apply, p, box: () => box };
  }

  it("taps where the press began, within the slop", () => {
    const { g, apply, p } = setup();
    g.pointerDown(p(1, 100, 100));
    apply(g.pointerMove(p(1, 106, 104)));
    expect(g.pointerUp(p(1, 106, 104, 200))).toEqual({ type: "tap", at: { x: 100, y: 100 }, kind: "touch" });
  });

  it("pans without a jump and does not tap after a drag or a long hold", () => {
    const { g, apply, p, box } = setup();
    g.pointerDown(p(1, 100, 100));
    apply(g.pointerMove(p(1, 120, 100)));                // past the slop: starts the pan from here
    expect(box().x).toBe(0);
    apply(g.pointerMove(p(1, 140, 100)));
    expect(box().x).toBeCloseTo(-200);                   // 20 screen px = 200 photo px
    expect(g.pointerUp(p(1, 140, 100)).type).toBe("none");
    g.pointerDown(p(2, 50, 50, 1000));
    expect(g.pointerUp(p(2, 50, 50, 1000 + 700)).type).toBe("none");
  });

  it("never taps when a second finger came down, and the last finger pans on smoothly", () => {
    const { g, apply, p, box } = setup();
    g.pointerDown(p(1, 100, 150));
    g.pointerDown(p(2, 200, 150));
    apply(g.pointerMove(p(2, 300, 150)));
    apply(g.pointerMove(p(2, 301, 150)));
    const zoomed = box();
    expect(zoomed.w).toBeLessThan(4000);
    expect(g.pointerUp(p(2, 301, 150)).type).toBe("none");
    apply(g.pointerMove(p(1, 100, 150)));               // no movement: no jump
    expect(box()).toEqual(zoomed);
    expect(g.pointerUp(p(1, 100, 150, 100)).type).toBe("none");
  });

  it("forgets everything on cancel", () => {
    const { g, p } = setup();
    g.pointerDown(p(1, 100, 100));
    g.cancel();
    expect(g.active).toBe(false);
    expect(g.pointerUp(p(1, 100, 100)).type).toBe("none");
  });

  it("scales wheel and trackpad zoom", () => {
    expect(wheelFactor(100, 0, false)).toBeCloseTo(Math.exp(0.15));
    expect(wheelFactor(3, 1, false)).toBeCloseTo(Math.exp(48 * 0.0015));
    expect(wheelFactor(-10, 0, true)).toBeLessThan(1);
  });
});

describe("edits", () => {
  it("follows the desktop editor's rules for each kind of marker", () => {
    const v = view();
    expect(editsAfterTap(v, { x: 2000, y: 500 }, 5).edits.add).toContainEqual([2000, 500]);     // empty: add
    expect(editsAfterTap(v, { x: 1005, y: 1000 }, 5).edits.remove).toContainEqual([1000, 1000]); // pin: remove
    expect(editsAfterTap(v, { x: 3000, y: 2010 }, 5).edits.add).toContainEqual([3000, 2000]);    // unused: use
    expect(editsAfterTap(v, { x: 1400, y: 1000 }, 5).edits.add).toEqual([]);                     // mine: delete
    expect(editsAfterTap(v, { x: 1600, y: 1000 }, 5).edits.remove).toEqual([]);                  // ghost: restore
    expect(v.edits).toEqual(view().edits);                                                        // not mutated
  });

  it("gives fingers a larger target than the marker", () => {
    const v = view();
    expect(hitTest(v, { x: 1000, y: 1040 }, 5)).toBeNull();
    expect(hitTest(v, { x: 1000, y: 1040 }, 45)?.kind).toBe("pin");
    expect(say(hitTest(v, { x: 1000, y: 1000 }, 5))).toBe("Pin 1. Tap to remove it.");
  });

  it("undoes in order and rolls back a failed send", () => {
    const h = new EditHistory();
    const a = { add: [], remove: [] }, b = { add: [[1, 2]] as [number, number][], remove: [] };
    h.commit(a);
    h.commit(b);
    h.rollback();
    expect(h.undo()).toEqual(a);
    expect(h.canUndo).toBe(false);
  });
});

describe("fitWidth", () => {
  it("puts the photo's width across the element and centres it top to bottom, past the element's ends", () => {
    // A portrait photo in a landscape element: the middle band, full width.
    expect(fitWidth({ width: 3000, height: 4000 }, { width: 400, height: 300 })).toEqual({ x: 0, y: 875, w: 3000, h: 2250 });
    // A landscape photo in a taller element: full width, with room above and below.
    expect(fitWidth({ width: 4000, height: 3000 }, { width: 400, height: 400 })).toEqual({ x: 0, y: -500, w: 4000, h: 4000 });
  });
});
