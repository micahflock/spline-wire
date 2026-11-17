import { describe, expect, it } from "vitest";
import fixture from "../test-data/measurement.json";
import type { MeasurementDocument } from "../core/types";
import { MAX_PACKED_CHARS, pack, parseFiles, unpack } from "../relay/files";
import { b64url } from "../relay/crypto";
import { parseCurve, pointsTsv } from "../relay/messages";
import { curveFrom } from "./curve";

const doc = fixture.document as unknown as MeasurementDocument;
const ctx = { chainName: "O-ring chain", pitch_mm: 10, half_width_mm: 4, n_pins: 13, now: new Date("2026-11-18T08:00:00Z") };

describe("curve from a Python measurement", () => {
  it("makes a curve message the desktop accepts", () => {
    const c = curveFrom(doc, ctx)!;
    expect(parseCurve(c)).toEqual(c);
    expect(c.points_mm).toEqual(doc.curve!.points);
    expect(c.measured_at).toBe("2026-11-18T08:00:00.000Z");
    // The clipboard table the desktop makes from the points is Python's.
    expect(pointsTsv(c.points_mm)).toBe(fixture.outputs.tsv);
  });

  it("has nothing to send for a failed measurement", () => {
    expect(curveFrom({ ...doc, ok: false, result: null, curve: null, error: "found 0 fiducials" }, ctx)).toBeNull();
  });
});

describe("files in the message (D14)", () => {
  it("travels in the curve message and survives the desktop's checks", async () => {
    const c = curveFrom(doc, ctx, await pack(fixture.outputs))!;
    const checked = parseCurve(JSON.parse(JSON.stringify(c)))!;
    expect(checked.files).toEqual(c.files);
    expect((await unpack(checked.files!)).dxf).toBe(fixture.outputs.dxf);
    expect(parseCurve({ ...c, files: { dxf: "!!" } })).toBeNull();
  });

  it("round-trips Python's output files, small enough for one message", async () => {
    const packed = await pack(fixture.outputs);
    const size = Object.values(packed).reduce((n, s) => n + s!.length, 0);
    console.log(`packed files: ${size} base64 chars for ${Object.values(fixture.outputs).join("").length} chars of text`);
    expect(size).toBeLessThan(MAX_PACKED_CHARS / 3);
    expect(parseFiles(packed)).toEqual(packed);
    const back = await unpack(packed);
    for (const kind of ["json", "csv", "svg", "dxf", "fusion_csv"] as const) {
      expect(back[kind]).toBe((fixture.outputs as Record<string, string>)[kind]);
    }
  });

  it("refuses what is not packed files and survives junk", async () => {
    expect(parseFiles({ dxf: "not base64!" })).toBeNull();
    expect(parseFiles({ dxf: "A".repeat(MAX_PACKED_CHARS + 1) })).toBeNull();
    expect(parseFiles({ future_kind: "AAAA" })).toEqual({});
    expect(await unpack({ dxf: b64url(new TextEncoder().encode("not gzip")) })).toEqual({});
  });

  it("leaves the files out rather than exceed the cap", async () => {
    let seed = 1;
    const noise = Array.from({ length: 60_000 }, () => ((seed = (seed * 48271) % 2147483647) % 36).toString(36)).join("");
    expect(await pack({ dxf: noise })).toEqual({});
  });
});
