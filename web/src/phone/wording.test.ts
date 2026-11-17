import { describe, expect, it } from "vitest";
import type { MeasurementResult, PhotoView } from "../core/types";
import { overlaySvg } from "./draw";
import { errorAdvice, errorTitle, failureText, numbers, readNote, readWarning } from "./wording";

describe("wording", () => {
  it("puts the core's warnings in plain words", () => {
    expect(readWarning("1 pin(s) not detected; the curve bridges those gaps").title).toBe("1 pin not found");
    expect(readWarning("found 11 pins on the chain, which is set to have 13; is the whole...").title).toBe("Found 11 pins, expected 13");
    expect(readWarning("no focal length (none in the photo's EXIF, and none set); estimating it").title).toBe("Focal length estimated");
    expect(readWarning("link lengths deviate from the pitch by up to 0.62 mm; check").title).toBe("A link is 0.62 mm off its length");
    expect(readWarning("something new")).toEqual({ title: "Check this measurement", text: "something new" });
    expect(readNote("ignored 3 look-alike detection(s) not on the chain").title).toBe("3 stray dots ignored");
  });

  it("heads a person's error and keeps the message itself", () => {
    expect(errorTitle("found 5 fiducials; need the whole chain in view")).toBe("Only 5 dots found.");
    expect(errorTitle("found 2 fiducials; need the whole chain in view")).toBe("No chain found in this photo.");
    expect(errorTitle("found 4 pins in a row, joined by 3 links")).toBe("Only 4 pins in a row.");
    expect(errorAdvice("found 4 pins in a row, joined by 3 links", true)).toMatch(/tap each pin/);
    expect(errorAdvice("found 4 pins in a row, joined by 3 links", false)).not.toMatch(/tap each pin/);
    const person = failureText({ kind: "person", message: "notes.txt is not a photo this program can read" });
    expect(person).toEqual({ title: "Couldn't read this photo.", text: "notes.txt is not a photo this program can read", bug: false });
    expect(failureText({ kind: "bug", type: "KeyError", message: "'pins'" }).text).toBe("KeyError: 'pins'");
  });

  it("flags an estimated focal length and missing pins in the numbers", () => {
    const r = {
      pins: 12, missing_pins: 1, rejected: 0, curve_points: 25, tilt_deg: 12.34, focal_35mm: 27.1, focal_source: "estimated",
      link_residual_rms_mm: 0.02, link_residual_max_mm: 0.05, object_side: "inside", warnings: [], notes: [], truth: null, truth_status: null,
    } as MeasurementResult;
    const rows = Object.fromEntries(numbers(r, 13).map(([k, v, s, w]) => [k, { v, s, w }]));
    expect(rows.Pins).toEqual({ v: "12 of 13", s: "1 missing, bridged", w: true });
    expect(rows["Focal length"]).toEqual({ v: "27.1 mm", s: "estimated from the chain", w: true });
    expect(rows.Tilt!.v).toBe("12.3°");
    expect(numbers({ ...r, pins: 13, missing_pins: 0, focal_source: "exif" }, 13).filter(([, , , w]) => w)).toEqual([]);
  });
});

describe("overlay", () => {
  it("draws pins with their position along the chain, gaps, strays and ghosts", () => {
    const v: PhotoView = {
      width: 100, height: 80, r: 2,
      pins: [{ x: 10, y: 10, r: 2, k: 0 }, { x: 20, y: 10, r: 2, k: 1 }, { x: 40, y: 10, r: 2, k: 2, edit: 0, snapped: false }],
      gaps: [1], unused: [{ x: 50, y: 50, r: 2 }], curve: [[10, 14], [40, 14]], side_px: null,
      removed: [{ x: 30, y: 10, edit: 0 }], edits: { add: [[40, 10]], remove: [[30, 10]] },
    };
    const svg = overlaySvg(v, { pending: { x: 5, y: 5 } });
    expect(svg.match(/class="ov-pin"/g)).toHaveLength(2);
    expect(svg).toContain('class="ov-hand" cx="40" cy="10" r="2.6" data-k="2" data-edit="0"');
    expect(svg.match(/class="ov-gap"/g)).toHaveLength(1);
    expect(svg).toContain('class="ov-unused"');
    expect(svg).toContain('class="ov-removed"');
    expect(svg).toContain('class="ov-pending"');
    expect(svg).toContain('class="ov-edge" points="10,14 40,14"');
  });
});
