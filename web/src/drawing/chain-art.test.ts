import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { chainArt, chainMarkSvg, linkStyle, reachMm, SNAP_ART, tailPositions } from "./chain-art";
import { ORING, SNAP } from "./test-looks";

describe("the chain as drawn", () => {
  it("knows the snap-fit chain by its round contact and rims, and draws anything else as plates", () => {
    expect(linkStyle(SNAP)).toBe("snap");
    expect(linkStyle(ORING)).toBe("plates");
    expect(linkStyle({ ...SNAP, contact: "edge" })).toBe("plates");
    expect(tailPositions(SNAP)).toBe(1);
    expect(tailPositions(ORING)).toBe(0);
    expect(reachMm(SNAP)).toBeCloseTo(4.395, 3);
  });

  it("draws the snap-fit chain's heads to the rim's size, the rim as wide as the preset's", () => {
    const svg = chainArt(SNAP, [[0, 0], [13, 0], [26, 0]], 1);
    expect(svg).toContain(`r="4.39" style="fill:var(--ca-link)"`);
    expect(svg).toContain(`r="3.97" style="fill:none;stroke:var(--ca-mark)" stroke-width="0.86"`);
    expect(svg.match(/class="sleeve"/g)).toHaveLength(1);
    expect(svg).toContain(`stroke-width="${2 * SNAP_ART.armHalf}"`);
  });

  it("swaps the tones for a chain printed the other way round", () => {
    const svg = chainArt({ ...ORING, fiducial_shade: "dark" }, [[0, 0], [10, 0]], 1);
    expect(svg).toContain("fill:var(--ca-link)");          // the dots, dark
    expect(svg).toContain("stroke:var(--ca-pale)");        // the plate, pale
  });

  it("the mark: a straight piece, whose box holds the heads and the snap-fit chain's tail", () => {
    const svg = chainMarkSvg(SNAP, 3);
    const box = svg.match(/viewBox="([^"]+)"/)![1]!.split(" ").map(Number);
    expect(box[0]).toBeLessThan(-4.39);
    expect(box[0]! + box[2]!).toBeGreaterThan(3 * 13 + 4.39);
    expect(svg).toContain(`aria-hidden="true"`);
  });

  it("draws from the presets' own numbers (data/chain*.yaml)", () => {
    const read = (file: string) => {
      const text = new TextDecoder().decode(readFileSync(new URL(`../../../data/${file}`, import.meta.url)));
      const v = (k: string) => text.match(new RegExp(`^${k}:\\s*(\\S+)`, "m"))?.[1];
      return { pitch_mm: Number(v("pitch_mm")), half_width_mm: Number(v("half_width_mm")), fiducial: v("fiducial") ?? "dot",
        fiducial_mm: Number(v("fiducial_mm")), ring_inner_mm: Number(v("ring_inner_mm") ?? 0), contact: v("contact") ?? "edge" };
    };
    expect(read("chain-snap.yaml")).toEqual(SNAP);
    expect(read("chain.yaml")).toEqual(ORING);
  });
});
