import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { chainArt, chainMarkSvg, linkStyle, reachMm, SNAP_ART } from "./chain-art";
import { ORING, SNAP } from "./test-looks";

describe("the chain as drawn", () => {
  it("knows the snap-fit chain by its round contact and rims, and draws anything else as plates", () => {
    expect(linkStyle(SNAP)).toBe("snap");
    expect(linkStyle(ORING)).toBe("plates");
    expect(linkStyle({ ...SNAP, contact: "edge" })).toBe("plates");
    expect(reachMm(SNAP)).toBeCloseTo(4.395 + SNAP_ART.edge, 3);
    expect(reachMm(ORING)).toBe(4);
  });

  it("draws the snap-fit chain: a head on every pin, the last hook's included, one link fewer", () => {
    const svg = chainArt(SNAP, [[0, 0], [13, 0], [26, 0]], 1);
    expect(svg.match(/<circle/g)).toHaveLength(6);             // 3 heads, each with its rim
    expect(svg.match(/<line/g)).toHaveLength(2);
    expect(svg).toContain(`stroke-width="${2 * SNAP_ART.armHalf}"`);
  });

  it("draws the rim as wide as the preset's, in the mark's colour, with the link's colour outside it as well as inside", () => {
    const svg = chainArt(SNAP, [[0, 0], [13, 0]], 1);
    expect(svg).toContain(`r="${+(4.395 + SNAP_ART.edge).toFixed(2)}" style="fill:var(--ca-link)"`);
    expect(svg).toContain(`r="3.97" style="fill:none;stroke:var(--ca-mark)" stroke-width="0.86"`);
    const dark = chainArt({ ...SNAP, fiducial_shade: "dark" }, [[0, 0], [13, 0]], 1);
    expect(dark).toContain("fill:var(--ca-pale)");
    expect(dark).toContain("stroke:var(--ca-link)");          // the rim dark on a pale head
  });

  it("swaps the tones for a chain printed the other way round", () => {
    const svg = chainArt({ ...ORING, fiducial_shade: "dark" }, [[0, 0], [10, 0]], 1);
    expect(svg).toContain("fill:var(--ca-link)");          // the dots, dark
    expect(svg).toContain("stroke:var(--ca-pale)");        // the plate, pale
  });

  it("the mark: a straight piece of 3 pins, whose box holds the heads", () => {
    const svg = chainMarkSvg(SNAP, 3);
    const box = svg.match(/viewBox="([^"]+)"/)![1]!.split(" ").map(Number);
    expect(svg.match(/<circle/g)).toHaveLength(6);
    expect(box[0]).toBeLessThan(-reachMm(SNAP));
    expect(box[0]! + box[2]!).toBeGreaterThan(2 * 13 + reachMm(SNAP));
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
