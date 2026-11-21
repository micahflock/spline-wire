import { describe, expect, it } from "vitest";
import { ANIM, ArrowAnimator, HoldAnimator, lockAlpha, lockRadius } from "./animate";

const angle = (d: [number, number]) => (Math.atan2(d[1], d[0]) * 180) / Math.PI;

describe("the arrow's animation", () => {
  it("fades in where the coach puts it, then follows it smoothly, the short way round", () => {
    const a = new ArrowAnimator();
    expect(a.update(null, 0)).toBeNull();
    const target = { dir: [1, 0] as [number, number], along: 100, length: 80 };
    expect(a.update(target, 0)).toMatchObject({ dir: [1, 0], along: 100, alpha: 0 });
    const half = a.update(target, ANIM.arrowFadeMs / 2)!;
    expect(half.alpha).toBeCloseTo(0.5);
    expect(a.update(target, 1000)!.alpha).toBe(1);
    // A turn from 170 to -170 degrees goes through 180, not through 0.
    const b = new ArrowAnimator();
    b.update({ ...target, dir: [Math.cos(Math.PI * 170 / 180), Math.sin(Math.PI * 170 / 180)] }, 0);
    const mid = b.update({ ...target, dir: [Math.cos(-Math.PI * 170 / 180), Math.sin(-Math.PI * 170 / 180)] }, ANIM.arrowTauMs)!;
    expect(Math.abs(angle(mid.dir))).toBeGreaterThan(170);
    // A sudden change of direction is followed over a few time constants, not at once.
    const c = new ArrowAnimator();
    c.update(target, 0);
    const step = c.update({ ...target, dir: [0, 1], along: 200 }, 16)!;
    expect(angle(step.dir)).toBeGreaterThan(0);
    expect(angle(step.dir)).toBeLessThan(10);
    expect(step.along).toBeLessThan(115);
    let s = step;
    for (let t = 32; t <= 1000; t += 16) s = c.update({ ...target, dir: [0, 1], along: 200 }, t)!;
    expect(angle(s.dir)).toBeCloseTo(90, 0);
    expect(s.along).toBeCloseTo(200, 0);
  });

  it("fades out where it was when the coach drops it, then is gone", () => {
    const a = new ArrowAnimator();
    const target = { dir: [0, -1] as [number, number], along: 50, length: 60 };
    a.update(target, 0);
    a.update(target, 1000);
    const going = a.update(null, 1000 + ANIM.arrowFadeMs / 2)!;
    expect(going.alpha).toBeCloseTo(0.5);
    expect(going.dir[1]).toBeCloseTo(-1);
    expect(a.update(null, 1000 + ANIM.arrowFadeMs + 1)).toBeNull();
  });
});

describe("the lock-on rings", () => {
  it("close from lockScale times the pin's ring to the ring itself as the hold fills", () => {
    expect(lockRadius(10, 0)).toBeCloseTo(10 * ANIM.lockScale);
    expect(lockRadius(10, 1)).toBe(10);
    expect(lockRadius(10, 0.5)).toBeLessThan(10 * (1 + ANIM.lockScale) / 2);     // fast at first, then settling
    expect(lockAlpha(0)).toBe(0);
    expect(lockAlpha(1)).toBeCloseTo(0.95);
    expect(lockAlpha(0.5)).toBeGreaterThan(lockAlpha(0.1));
  });

  it("follow the hold as it fills, and open out smoothly when it is broken", () => {
    const h = new HoldAnimator();
    expect(h.update(0.2, 0)).toBe(0.2);
    expect(h.update(0.6, 16)).toBe(0.6);
    const after = h.update(0, 32);
    expect(after).toBeGreaterThan(0.4);                       // not snapped open
    let v = after;
    for (let t = 48; t <= 1000; t += 16) v = h.update(0, t);
    expect(v).toBe(0);
  });
});
