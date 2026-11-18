// session.ts against splinewire/session.py (golden/session.json.gz,
// web/scripts/golden_session.py): the same documents and the same files, on
// the committed photos under a range of options, failures included.
//
// Strings must match exactly. Numbers must match to within one unit of the
// last digit they are rounded to: the TypeScript geometry agrees with
// Python's to ~1e-7 mm, so a value right on a rounding edge (0.00005 mm at 4
// decimals) can round the other way; unrounded numbers (the DXF's
// coordinates, the truth comparison) must agree to 1e-4 mm, as in golden.test.ts. Files are
// compared the same way, number by number, the text between the numbers
// exactly.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import type { MeasureOptions, MeasurementDocument } from "../core/types";
import { readGrayPng } from "./golden/png";
import { PhotoSession } from "./session";

interface Case { photo: string; options: MeasureOptions; document: MeasurementDocument; outputs: Record<string, string> | null }

const cases = JSON.parse(
  gunzipSync(readFileSync(new URL("./golden/session.json.gz", import.meta.url))).toString("utf-8"),
) as Case[];

/** Decimal places a number was written with. */
function places(x: number): number {
  const s = String(x);
  if (s.includes("e")) return 12;
  const dot = s.indexOf(".");
  return dot < 0 ? 0 : s.length - dot - 1;
}

let rounding = 0;

function same(got: unknown, want: unknown, path: string): void {
  if (typeof want === "number" && typeof got === "number") {
    // Rounded numbers: one unit of the last digit. Unrounded ones (the truth comparison): relative.
    const unit = places(want) > 8 ? 1e-4 * Math.max(1, Math.abs(want)) : 10 ** -Math.max(places(want), 1);
    if (got !== want) rounding++;
    expect(Math.abs(got - want), `${path}: ${got} against ${want}`).toBeLessThanOrEqual(unit * 1.0001);
    return;
  }
  if (Array.isArray(want)) {
    expect(Array.isArray(got), path).toBe(true);
    expect((got as unknown[]).length, `${path} length`).toBe(want.length);
    want.forEach((w, i) => same((got as unknown[])[i], w, `${path}[${i}]`));
    return;
  }
  if (want && typeof want === "object") {
    expect(got && typeof got === "object", path).toBeTruthy();
    expect(Object.keys(got as object).sort(), `${path} keys`).toEqual(Object.keys(want).sort());
    for (const [k, v] of Object.entries(want)) same((got as Record<string, unknown>)[k], v, `${path}.${k}`);
    return;
  }
  expect(got, path).toEqual(want);
}

const NUMBER = /-?\d+\.\d+(?:e[-+]\d+)?/g;

function sameText(got: string, want: string, kind: string, atLeast = 0): void {
  expect(got.replace(NUMBER, "#"), `${kind}: the text between numbers`).toBe(want.replace(NUMBER, "#"));
  const g = got.match(NUMBER) ?? [], w = want.match(NUMBER) ?? [];
  w.forEach((s, i) => {
    const digits = s.split(".")[1]!.replace(/e.*/, "").length;
    // Unrounded numbers (the DXF's coordinates, the truth comparison): as golden.test.ts, 1e-4 mm (the solver stops that far apart with an estimated focal length).
    const unit = Math.max(atLeast, digits > 8 ? 1e-4 * Math.max(1, Math.abs(Number(s))) : 10 ** -digits);
    if (g[i] !== s) rounding++;
    expect(Math.abs(Number(g[i]) - Number(s)), `${kind} number ${i}: ${g[i]} against ${s}`).toBeLessThanOrEqual(unit * 1.0001);
  });
}

describe("PhotoSession against Python", () => {
  const sessions = new Map<string, PhotoSession>();
  for (const [n, c] of cases.entries()) {
    it(`${c.photo} #${n}: ${c.document.error ?? "ok"}`, () => {
      let s = sessions.get(c.photo);
      if (!s) {
        s = new PhotoSession(c.photo, readGrayPng(new URL(`./golden/${c.photo}`, import.meta.url)), null, c.document.info);
        sessions.set(c.photo, s);
      }
      same(s.measure(c.options), c.document, "document");
      if (c.outputs) {
        const files = s.outputs();
        for (const kind of ["json", "csv", "svg", "fusion_csv", "tsv"] as const) sameText(files[kind]!, c.outputs[kind]!, kind);
        // The DXF's coordinates have 6 decimals, finer than the two measurements agree
        // (1e-4 mm, golden.test.ts); dxf.test.ts holds the writer itself to Python's byte for byte.
        sameText(files.dxf, c.outputs.dxf!, "dxf", 1e-4);
      }
    }, 60000);
  }
  it("reports", () => console.log(`[session] ${cases.length} documents; ${rounding} numbers differ in their last digits (no more than allowed)`));
});
