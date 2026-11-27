import { describe, expect, it } from "vitest";
import type { MeasurementResult } from "../core/types";
import { errorAdvice, errorTitle, failureText, focalStatus, measurementRows, numbers, photoFocalText, readNote, readWarning, spaceText } from "./wording";

describe("wording", () => {
  it("puts the core's warnings in plain words", () => {
    expect(readWarning("1 pin(s) not detected; the curve bridges those gaps").title).toBe("1 pin not found");
    expect(readWarning("found 11 pins on the chain, which is set to have 13; is the whole...").title).toBe("Found 11 pins, expected 13");
    expect(readWarning("no focal length (none in the photo's EXIF, and none set); estimating it").title).toBe("Focal length estimated");
    expect(readWarning("link lengths deviate from the pitch by up to 0.62 mm; check").title).toBe("A link is 0.62 mm off its length");
    expect(readWarning("something new")).toEqual({ title: "Check this measurement", text: "something new" });
    expect(readNote("ignored 3 look-alike detection(s) not on the chain").title).toBe("3 stray dots ignored");
  });

  it("speaks of clicks where the photo is edited with a mouse", () => {
    expect(readWarning("1 pin(s) not detected; the curve bridges those gaps").text).toMatch(/^The curve bridges the gap. Tap /);
    expect(readWarning("1 pin(s) not detected; the curve bridges those gaps", "mouse").text).toMatch(/ Click the missing dot/);
    expect(readWarning("2 pin(s) you added have no fiducial under them", "mouse").text).not.toMatch(/tap/i);
    expect(readWarning("2 pin(s) you added have no fiducial under them", "mouse").title).toBe("2 pins placed where you clicked");
    expect(readWarning("2 pin(s) you added have no fiducial under them").title).toBe("2 pins placed where you tapped");
    expect(readNote("ignored 3 look-alike detection(s) not on the chain", "mouse").text).toMatch(/Click one/);
    expect(errorAdvice("found 4 pins in a row, joined by 3 links", true, "mouse")).toMatch(/click each pin/);
    expect(errorAdvice("x.txt is not a photo this program can read", false, "mouse")).not.toMatch(/camera button/);
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
    // The phone's Measurement table: the same figures and flags, in the brief's rows.
    const spec = Object.fromEntries(measurementRows(r, 13).map(([k, v, s, w]) => [k, { v, s, w }]));
    expect(Object.keys(spec)).toEqual(["Pins found", "Camera tilt", "Link length error", "Focal length"]);
    expect(spec["Pins found"]).toEqual({ v: "12 / 13", s: "1 missing, bridged", w: true });
    expect(spec["Focal length"]).toEqual({ v: "27.1 mm", s: "estimated from the chain", w: true });
    expect(spec["Link length error"]!.s).toBe("rms · max 0.050");
    expect(measurementRows({ ...r, pins: 13, missing_pins: 0, focal_source: "exif" }, 13).filter(([, , , w]) => w)).toEqual([]);
  });
});

describe("a focal length the page supplied", () => {
  it("says where it came from", () => {
    expect(photoFocalText({ why: "lens", lens: "Pixel 7 Pro back camera 6.81mm f/1.85" })).toMatch(/^remembered for Pixel 7 Pro/);
    expect(photoFocalText({ why: "frame", lens: null })).toMatch(/focal length in Settings.*video frame/);
    expect(photoFocalText({ why: "frame", lens: "Back Camera" })).toMatch(/camera's last photo/);
  });
});

describe("what the phone keeps", () => {
  it("counts photos with their size and curves without their photo", () => {
    expect(spaceText(3, 12_400_000, 2)).toBe("3 photos (12 MB) and 2 curves without their photos");
    expect(spaceText(1, 3_210_000, 0)).toBe("1 photo (3.2 MB)");
    expect(spaceText(0, 0, 1)).toBe("1 curve without its photo");
    expect(spaceText(0, 0, 0)).toBe("Nothing yet");
  });

  it("says where the focal length for photos without camera data comes from", () => {
    const lens = { name: "Pixel 7 Pro back camera 6.81mm f/1.85", focal35: 24 };
    expect(focalStatus({ focal35: 24, lens }, lens)).toBe(
      "24 mm, learned from Pixel 7 Pro back camera 6.81mm f/1.85. Used for photos without camera data, which iPhones often strip.");
    expect(focalStatus({ focal35: 26, lens: null }, lens)).toMatch(/Clear it to use the 24 mm learned from Pixel 7 Pro/);
    expect(focalStatus({ focal35: 26, lens: null }, null)).toMatch(/Clear it to use one learned from a photo\.$/);
    expect(focalStatus(null, null)).toMatch(/^Not known yet: .* estimated from the chain/);
  });
});
