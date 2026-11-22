import { describe, expect, it } from "vitest";
import type { ChainSpec } from "../core/types";
import { MemoryStore } from "../relay/store";
import { learn } from "./lenses";
import {
  DEFAULT_SETTINGS, chainLabel, chainSpec, defaultPins, loadSettings, parseFocal, parsePins, presetKey, saveSettings,
} from "./settings";

const chains: Record<string, ChainSpec> = {
  snap: { pitch_mm: 13, half_width_mm: 4.68, fiducial_mm: 7.2, n_pins: 11, fiducial: "ring" },
  oring: { pitch_mm: 10, half_width_mm: 4, fiducial_mm: 5, n_pins: 13 },
};

describe("settings", () => {
  it("start from the defaults and survive a round trip", () => {
    const store = new MemoryStore();
    expect(loadSettings(store)).toEqual(DEFAULT_SETTINGS);
    const s = { ...DEFAULT_SETTINGS, chain: "snap", pins: 9, savedFocal35: 26, side: "outside" as const, photoMode: "upload" as const,
      shade: "dark" as const, keepPhotos: 3, liveShows: "chain" as const };
    saveSettings(store, s);
    expect(loadSettings(store)).toEqual(s);
  });

  it("drop fields that are out of range or of the wrong type", () => {
    const store = new MemoryStore();
    store.set("phone-settings", JSON.stringify({
      chain: 3, pins: 2, savedFocal35: 4, focalOverride35: "26", side: "left", lenses: [{ key: "x" }], cameras: { a: 1 }, camera: "Back Camera", photoMode: "file",
      keepPhotos: 7, liveShows: "none",
    }));
    expect(loadSettings(store)).toEqual(DEFAULT_SETTINGS);
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
    expect(chainSpec(chains, "snap", 7)).toEqual({ ...chains.snap, n_pins: 7, fiducial_shade: "light" });
    expect(chainSpec(chains, "snap", 7, "dark")).toEqual({ ...chains.snap, n_pins: 7, fiducial_shade: "dark" });
    expect(chainSpec(chains, "gone", 7).pitch_mm).toBe(13);          // a preset this build lacks: the first, the default
    expect(presetKey(chains, "gone")).toBe("snap");
    expect(presetKey(chains, "default")).toBe("snap");                // saved before the rename: the default now
    expect(DEFAULT_SETTINGS.chain).toBe("snap");
    expect(defaultPins(DEFAULT_SETTINGS, chains)).toBe(11);
    expect(defaultPins({ ...DEFAULT_SETTINGS, chain: "oring" }, chains)).toBe(13);
    expect(defaultPins({ ...DEFAULT_SETTINGS, pins: 20 }, chains)).toBe(20);
    expect(chainLabel("snap")).toBe("Snap-fit chain");
    expect(chainLabel("oring")).toBe("O-ring chain");
    expect(chainLabel("mine")).toBe("mine");
  });
});

describe("the remembered focal lengths", () => {
  it("keep each lens and the chosen camera across a reload", () => {
    const store = new MemoryStore();
    const s = learn({ ...DEFAULT_SETTINGS, camera: { deviceId: "d1", label: "Back Camera" } },
      { make: "Apple", model: "iPhone 16 Pro", lens: "iPhone 16 Pro back camera 6.765mm f/1.78", focalMm: 6.765, focal35: 24 },
      { width: 4032, height: 3024 }, { cameraLabel: "Back Camera", now: 5 });
    saveSettings(store, s);
    expect(loadSettings(store)).toEqual(s);
  });

  it("drop the single value kept before D17, whose lens is unknown, and keep the person's saved one", () => {
    const store = new MemoryStore();
    store.set("phone-settings", JSON.stringify({ savedFocal35: 26, lastExif: { focal35: 30, camera: "Apple iPhone 13" }, declinedFocal35: 26 }));
    const s = loadSettings(store);
    expect(s).toEqual({ ...DEFAULT_SETTINGS, savedFocal35: 26 });
    expect("lastExif" in s).toBe(false);
  });
});
