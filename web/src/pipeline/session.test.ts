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
import { readGrayPng, readRgbaPng } from "./golden/png";
import { grayFromRgba } from "./decode";
import { searchBox } from "./detect";
import { PhotoSession } from "./session";
import type { GrayChannel } from "./types";

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

/** A golden photo as the engine reads it for `channel`: a gray PNG as it is, a colour one
 * (photo-snap-orange.png) by decode.ts's conversion, which Python's gray_from_image matches. */
function photoGray(photo: string, channel: GrayChannel) {
  const url = new URL(`./golden/${photo}`, import.meta.url);
  if (photo !== "photo-snap-orange.png") return readGrayPng(url);
  const rgba = readRgbaPng(url);
  return grayFromRgba(rgba.data, rgba.width, rgba.height, channel);
}

describe("PhotoSession against Python", () => {
  const sessions = new Map<string, PhotoSession>();
  for (const [n, c] of cases.entries()) {
    it(`${c.photo} #${n}: ${c.document.error ?? "ok"}`, () => {
      const raw = c.options.chain.gray_channel ?? "luma";
      const channel: GrayChannel = ["luma", "r", "g", "b"].includes(raw) ? raw : "luma";
      let s = sessions.get(c.photo);
      if (!s) {
        s = new PhotoSession(c.photo, photoGray(c.photo, channel), null, c.document.info, channel);
        sessions.set(c.photo, s);
      }
      // As the engine does (core/engine.ts): another channel, the photo read again.
      if (channel !== s.channel) s.useImage(photoGray(c.photo, channel), channel);
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
  it("measures a photo only in the channel it was read with (the engine reads it again first)", () => {
    const c = cases.find((k) => k.photo === "photo-snap-orange.png" && k.options.chain.gray_channel === "b" && k.document.ok)!;
    const s = new PhotoSession(c.photo, photoGray(c.photo, "luma"), null, c.document.info);
    const doc = s.measure(c.options);
    expect(doc.ok).toBe(false);
    expect(doc.error).toBe("Error: the photo was read as its luma, the chain reads its b");      // a bug, shown with its type
    s.useImage(photoGray(c.photo, "b"), "b");
    same(s.measure(c.options), c.document, "document");
    expect(() => s.useImage({ width: 1, height: 1, data: new Uint8Array(1) }, "g")).toThrow(/not \d+×\d+/);
  });
  it("reports", () => console.log(`[session] ${cases.length} documents; ${rounding} numbers differ in their last digits (no more than allowed)`));
});

describe("search_px: the chain looked for in a box first (the live view knows where it is)", () => {
  const c = cases.find((k) => k.photo === "photo-snap-07.png" && k.document.ok)!;
  // The golden photo is 848×240, smaller than a search box's least side (512 px):
  // set in a 2400×1600 photo of its table's gray, so that a box is a part of it.
  const small = readGrayPng(new URL(`./golden/${c.photo}`, import.meta.url));
  const big = (() => {
    const W = 2400, H = 1600, ox = 800, oy = 700, data = new Uint8Array(W * H).fill(small.data[0]!);
    for (let y = 0; y < small.height; y++) data.set(small.data.subarray(y * small.width, (y + 1) * small.width), (oy + y) * W + ox);
    return { width: W, height: H, data };
  })();
  const fresh = () => new PhotoSession(c.photo, big, null, { ...c.document.info, width: big.width, height: big.height });
  const pinsOf = (d: MeasurementDocument) => d.view!.pins.map((p) => [p.x, p.y] as [number, number]);

  it("finds the same chain in its box, without searching the whole photo", () => {
    const whole = fresh().measure(c.options);
    const pins = pinsOf(whole);
    const xs = pins.map((p) => p[0]), ys = pins.map((p) => p[1]);
    const link = Math.hypot(pins[1]![0] - pins[0]![0], pins[1]![1] - pins[0]![1]);
    const box: [number, number, number, number] = [Math.min(...xs) - 1.5 * link, Math.min(...ys) - 1.5 * link, Math.max(...xs) + 1.5 * link, Math.max(...ys) + 1.5 * link];
    const s = fresh();
    const doc = s.measure({ ...c.options, search_px: box });
    expect(doc.ok).toBe(true);
    expect(doc.result!.pins).toBe(whole.result!.pins);
    expect((s as unknown as { detected: unknown }).detected).toBeNull();          // the whole photo was never searched
    const got = pinsOf(doc);
    for (const [k, p] of pins.entries()) expect(Math.hypot(got[k]![0] - p[0], got[k]![1] - p[1])).toBeLessThan(0.01);
  });

  it("falls back to the whole photo when the box misses the chain, and refuses a box that is not one", () => {
    const s = fresh();
    const doc = s.measure({ ...c.options, search_px: [0, 0, 300, 300] });
    expect(doc.ok).toBe(true);
    expect(doc.result!.pins).toBe(fresh().measure(c.options).result!.pins);
    expect((s as unknown as { detected: unknown }).detected).not.toBeNull();
    const bad = fresh().measure({ ...c.options, search_px: [1, 2, "x"] as unknown as [number, number, number, number] });
    expect(bad.ok).toBe(false);
    expect(bad.error).toMatch(/search_px/);
  });

  it("grows a box to aligned corners and 512 px, inside the photo, as detect.search_box does", () => {
    expect(searchBox(3024, 4032, [10, 10, 20, 20])).toEqual([0, 0, 512, 512]);
    expect(searchBox(3024, 4032, [3000, 100, 3010, 110])).toEqual([2512, 0, 3024, 512]);
    expect(searchBox(3024, 4032, [-50, -50, 5000, 5000])).toBeNull();
    expect(searchBox(3024, 4032, [101, 203, 1999, 1501])).toEqual([96, 200, 2000, 1504]);
  });
});
