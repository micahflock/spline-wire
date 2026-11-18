import { describe, expect, it } from "vitest";
import { fixed, fsum, median, roundHalfEven } from "./numeric";

describe("numeric helpers match Python", () => {
  it("fsum is exact and order-free", () => {
    expect(fsum([0.1, 0.2, 0.3])).toBe(0.6);
    expect(fsum([1e100, 1, -1e100])).toBe(1);
    const v = [0.1, 1e-17, 3.3, -2.2, 7e5, 1 / 3];
    expect(fsum(v)).toBe(fsum([...v].reverse()));
  });

  it("median and rounding", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
    expect(roundHalfEven(1.5)).toBe(2);
    expect(roundHalfEven(2.5)).toBe(2);
    expect(roundHalfEven(-0.5)).toBe(-0);
    expect(roundHalfEven(1.4999)).toBe(1);
  });

  it("formats as Python's .2f", () => {
    expect(fixed(0.125, 2)).toBe("0.12");
    expect(fixed(0.375, 2)).toBe("0.38");
    expect(fixed(1.005, 2)).toBe("1.00");      // 1.005 is just under in binary
    expect(fixed(0.5612, 2)).toBe("0.56");
    expect(fixed(2.675, 2)).toBe("2.67");
  });
});
