import { describe, expect, it } from "vitest";
import type { ChainSpec, PhotoInfo } from "../core/types";
import { MemoryStore } from "../relay/store";
import {
  DEFAULT_SETTINGS, chainLabel, chainSpec, defaultPins, focalOffer, loadSettings, parseFocal, parsePins, presetKey,
  rememberExif, saveSettings,
} from "./settings";

const chains: Record<string, ChainSpec> = {
  default: { pitch_mm: 10, half_width_mm: 4, fiducial_mm: 5, n_pins: 13 },
  snap: { pitch_mm: 13, half_width_mm: 4.68, fiducial_mm: 7.2, n_pins: 11, fiducial: "ring" },
};

const info = (focal: number | null, model: string | null = "iPhone 13"): PhotoInfo => ({
  format: "JPEG", width: 4032, height: 3024, make: model ? "Apple" : null, model, focal_mm: null, focal_35mm: focal,
});

describe("settings", () => {
  it("start from the defaults and survive a round trip", () => {
    const store = new MemoryStore();
    expect(loadSettings(store)).toEqual(DEFAULT_SETTINGS);
    const s = { ...DEFAULT_SETTINGS, chain: "snap", pins: 9, savedFocal35: 26, side: "outside" as const };
    saveSettings(store, s);
    expect(loadSettings(store)).toEqual(s);
  });

  it("drop fields that are out of range or of the wrong type", () => {
    const store = new MemoryStore();
    store.set("phone-settings", JSON.stringify({
      chain: 3, pins: 2, savedFocal35: 4, focalOverride35: "26", side: "left", lastExif: { focal35: 900 }, declinedFocal35: 26,
    }));
    expect(loadSettings(store)).toEqual({ ...DEFAULT_SETTINGS, declinedFocal35: 26 });
    store.set("phone-settings", "{not json");
    expect(loadSettings(store)).toEqual(DEFAULT_SETTINGS);
  });

  it("parse focal lengths as the core accepts them (5 to 500 mm), comma or point", () => {
    expect(parseFocal("")).toBeNull();
    expect(parseFocal(" 26 ")).toBe(26);
    expect(parseFocal("24,5")).toBe(24.5);
    expect(parseFocal("4")).toBe("invalid");
    expect(parseFocal("abc")).toBe("invalid");
    expect(parsePins("12")).toBe(12);
    expect(parsePins("2")).toBe("invalid");
    expect(parsePins("12.5")).toBe("invalid");
  });

  it("measure with the preset and the photo's pin count", () => {
    expect(chainSpec(chains, "snap", 7)).toEqual({ ...chains.snap, n_pins: 7 });
    expect(chainSpec(chains, "gone", 7).pitch_mm).toBe(10);          // a preset this build lacks: the first
    expect(presetKey(chains, "gone")).toBe("default");
    expect(defaultPins(DEFAULT_SETTINGS, chains)).toBe(13);
    expect(defaultPins({ ...DEFAULT_SETTINGS, chain: "snap" }, chains)).toBe(11);
    expect(defaultPins({ ...DEFAULT_SETTINGS, pins: 20 }, chains)).toBe(20);
    expect(chainLabel("default")).toBe("O-ring chain");
    expect(chainLabel("mine")).toBe("mine");
  });
});

describe("the remembered focal length", () => {
  it("comes from any photo with EXIF and is offered until saved or declined", () => {
    let s = rememberExif(DEFAULT_SETTINGS, info(null));
    expect(s).toBe(DEFAULT_SETTINGS);                 // nothing to remember
    expect(focalOffer(s)).toBeNull();
    s = rememberExif(s, info(26));
    expect(s.lastExif).toEqual({ focal35: 26, camera: "Apple iPhone 13" });
    expect(focalOffer(s)).toEqual({ focal35: 26, camera: "Apple iPhone 13" });
    expect(focalOffer({ ...s, savedFocal35: 26 })).toBeNull();
    expect(focalOffer({ ...s, declinedFocal35: 26 })).toBeNull();
    // A different camera later: offered again even after declining the first.
    s = rememberExif({ ...s, declinedFocal35: 26 }, info(24, "iPhone 15"));
    expect(focalOffer(s)).toEqual({ focal35: 24, camera: "Apple iPhone 15" });
  });

  it("is not changed by the same photo twice", () => {
    const s = rememberExif(DEFAULT_SETTINGS, info(26));
    expect(rememberExif(s, info(26))).toBe(s);
    expect(rememberExif(s, info(26, null)).lastExif).toEqual({ focal35: 26, camera: null });
  });
});
