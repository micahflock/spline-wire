import { describe, expect, it } from "vitest";
import type { PhotoView } from "../core/types";
import { overlaySvg } from "./draw";

describe("overlay", () => {
  it("draws each pin as its detected outline, with gaps, strays and ghosts and no numbers", () => {
    const v: PhotoView = {
      width: 100, height: 80, r: 2,
      pins: [{ x: 10, y: 10, r: 2, r2: 2, angle: 0, k: 0 }, { x: 20, y: 10, r: 2.5, r2: 2, angle: 30, k: 1 }, { x: 40, y: 10, r: 2, k: 2, edit: 0, snapped: false }],
      gaps: [1], unused: [{ x: 50, y: 50, r: 2 }], curve: [[10, 14], [40, 14]], side_px: null,
      removed: [{ x: 30, y: 10, edit: 0 }], edits: { add: [[40, 10]], remove: [[30, 10]] },
    };
    const svg = overlaySvg(v, { pending: { x: 5, y: 5 } });
    expect(svg.match(/class="ov-pin"/g)).toHaveLength(2);
    expect(svg).toContain('<circle class="ov-pin" cx="10" cy="10" r="2" data-k="0"/>');
    expect(svg).toContain('<ellipse class="ov-pin" cx="20" cy="10" rx="2.5" ry="2" transform="rotate(30 20 10)" data-k="1"/>');
    expect(svg).toContain('class="ov-hand" cx="40" cy="10" r="2" data-k="2" data-edit="0"');
    expect(svg).not.toContain("<text");
    expect(svg.match(/class="ov-gap"/g)).toHaveLength(1);
    expect(svg).toContain('class="ov-unused"');
    expect(svg).toContain('class="ov-removed"');
    expect(svg).toContain('class="ov-pending"');
    expect(svg).toContain('class="ov-edge" points="10,14 40,14"');
  });
});
