// The chain card's link against the shared test vectors (chainlink-vectors.json,
// also read by tests/test_chainlink.py): the same results and messages as Python.

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  ALPHABET, applyLink, ChainLinkError, checkCharacter, checkOk, fmtMm, hasLink, linkUrl, linkValue, parseLink,
  type ChainLink, type PresetNumbers,
} from "./chainlink";
import { specFromDict } from "./chain";
import { PipelineError } from "./errors";

interface Vectors {
  presets: PresetNumbers;
  check_characters: { alphabet?: string; text: string; k: string }[];
  good: { link: string; parsed: ChainLink }[];
  bad: { link: string; error: string; message: string }[];
  build: { fields: ChainLink; value: string }[];
  build_bad: { fields: ChainLink; error: string; message: string }[];
}
const V = JSON.parse(new TextDecoder().decode(readFileSync(new URL("./chainlink-vectors.json", import.meta.url)))) as Vectors;
const P = V.presets;

function refusal(f: () => unknown): ChainLinkError {
  try {
    f();
  } catch (err) {
    if (err instanceof ChainLinkError) return err;
    throw err;
  }
  throw new Error("not refused");
}

describe("chain card link: the shared vectors", () => {
  it("has the vectors' check characters (Luhn mod N)", () => {
    for (const c of V.check_characters) {
      expect(checkCharacter(c.text, c.alphabet), c.text).toBe(c.k);
      expect(checkOk(c.text, c.k, c.alphabet), c.text).toBe(true);
    }
  });

  it("reads every good link as Python does", () => {
    for (const g of V.good) expect(parseLink(g.link, P), g.link).toEqual(g.parsed);
    expect(V.good.length).toBeGreaterThanOrEqual(10);
  });

  it("refuses every bad link with Python's code and message", () => {
    for (const b of V.bad) {
      const err = refusal(() => parseLink(b.link, P));
      expect([err.code, err.message], b.link).toEqual([b.error, b.message]);
    }
    for (const code of ["check", "pitch", "version", "cut_short", "unknown_field"]) {
      expect(V.bad.some((b) => b.error === code), code).toBe(true);
    }
  });

  it("builds the vectors' values, and refuses what Python refuses", () => {
    for (const b of V.build) {
      expect(linkValue(b.fields, P), JSON.stringify(b.fields)).toBe(b.value);
      expect(parseLink(b.value, P).base).toBe(b.fields.base);
    }
    for (const b of V.build_bad) {
      const err = refusal(() => linkValue(b.fields, P));
      expect([err.code, err.message]).toEqual([b.error, b.message]);
    }
  });
});

describe("chain card link", () => {
  it("is a pipeline error: its message is shown to the person as is", () => {
    const err = refusal(() => parseLink("1;base=snap;p=13;k=0", P));
    expect(err.name).toBe("ChainLinkError");
    expect(err).toBeInstanceOf(PipelineError);
  });

  it("catches every one changed character, and every swap of neighbours but 0 and z", () => {
    const body = "1;base=snap;p=12.987;h=4.39", k = checkCharacter(body);
    let subs = 0;
    for (let i = 0; i < body.length; i++) {
      const c = body[i]!;
      if (!ALPHABET.includes(c)) continue;
      for (const r of ALPHABET) if (r !== c) { subs++; expect(checkOk(body.slice(0, i) + r + body.slice(i + 1), k)).toBe(false); }
    }
    expect(subs).toBe(19 * 35);
    const n = 36, f = (x: number) => Math.floor((2 * x) / n) + ((2 * x) % n);
    const missed: string[] = [];
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) if ((f(a) + b - f(b) - a) % n === 0) missed.push(ALPHABET[a]! + ALPHABET[b]!);
    expect(missed).toEqual(["0z"]);
  });

  it("makes the card's address, and reads it back", () => {
    const url = linkUrl({ base: "snap", pitch_mm: 12.987, half_width_mm: 4.39 }, P);
    expect(url).toBe("https://micahflock.github.io/spline-wire/phone/?debug#cal=1;base=snap;p=12.987;h=4.39;k=k");
    expect(linkUrl({ base: "snap", pitch_mm: 13 }, P, "https://example.org/phone/#pair=x")).toBe("https://example.org/phone/#cal=1;base=snap;p=13;k=1");
    expect(parseLink(new URL(url).hash, P)).toEqual({ base: "snap", pitch_mm: 12.987, half_width_mm: 4.39 });
    expect(hasLink(new URL(url).hash)).toBe(true);
    expect(hasLink("#pair=abc&cal=1")).toBe(true);
    expect(hasLink("#pair=abc")).toBe(false);
    expect(hasLink("")).toBe(false);
  });

  it("formats millimetres as Python does", () => {
    expect([13, 12.987, 12.9875, 4.39, 0.0004, 3.951].map(fmtMm)).toEqual(["13", "12.987", "12.988", "4.39", "0", "3.951"]);
  });

  it("puts its numbers over the preset's, and keeps the rest", () => {
    const snap = specFromDict({ pitch_mm: 13, half_width_mm: 4.39, fiducial: "ring", fiducial_mm: 8.79, ring_inner_mm: 7.07, n_pins: 13, contact: "round", max_bend_deg: 110 });
    expect(applyLink(snap, { base: "snap", pitch_mm: 12.9 })).toEqual({ ...snap, pitch_mm: 12.9 });
    expect(applyLink(snap, { base: "snap", pitch_mm: 13.1, half_width_mm: 4.2, fiducial_shade: "dark", gray_channel: "b", n_pins: 20 }))
      .toEqual({ ...snap, pitch_mm: 13.1, half_width_mm: 4.2, fiducial_shade: "dark", n_pins: 20 });
  });
});
