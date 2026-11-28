import { describe, expect, it } from "vitest";
import type { ChainSpec } from "../core/types";
import { ChainLinkError, parseLink } from "../pipeline/chainlink";
import { MemoryStore } from "../relay/store";
import {
  MAX_CHAINS, calKey, calLabel, findChain, forgetChain, linkText, photoCalibration, rememberChain, storedCalibration, typedChain,
  type Calibration,
} from "./calibration";
import {
  DEFAULT_SETTINGS, activeChain, chainSpec, debugLeftovers, loadSettings, newPhotoChain, saveSettings, settingsChainLabel, useChain,
  withPhoneChain,
} from "./settings";

const chains: Record<string, ChainSpec> = {
  snap: { pitch_mm: 13, half_width_mm: 4.39, fiducial_mm: 8.79, ring_inner_mm: 7.07, n_pins: 13, fiducial: "ring", contact: "round",
    max_bend_deg: 110, fiducial_shade: "light", gray_channel: "luma" },
  oring: { pitch_mm: 10, half_width_mm: 4, fiducial_mm: 5, n_pins: 13 },
};
const card = (text: string, at = 1): Calibration => ({ ...parseLink(text, chains), via: "card", at });
const A = card("1;base=snap;p=12.987;h=4.39;k=k");

describe("the phone's calibrated chains", () => {
  it("are their numbers: a link with the preset's half width written out is the same chain", () => {
    const short = card("1;base=snap;p=12.987;k=u");
    expect(calKey(short, chains)).toBe(calKey(A, chains));
    expect(rememberChain([A], { ...short, at: 2 }, chains)).toEqual([{ ...short, at: 2 }]);
    expect(calKey(card("1;base=snap;p=13.1;k=v"), chains)).not.toBe(calKey(A, chains));
  });

  it("keep the newest ten, the one used last first", () => {
    let list: Calibration[] = [];
    for (let i = 0; i < 12; i++) list = rememberChain(list, { base: "snap", pitch_mm: 12.9 + i / 100, via: "typed", at: i }, chains);
    expect(list).toHaveLength(MAX_CHAINS);
    expect(list[0]!.pitch_mm).toBeCloseTo(13.01);
    expect(findChain(list, calKey(list[3]!, chains), chains)).toBe(list[3]);
    expect(forgetChain(list, calKey(list[3]!, chains), chains)).toHaveLength(9);
    expect(findChain(list, null, chains)).toBeUndefined();
  });

  it("are called by their preset and numbers, the half width only when it is not the preset's", () => {
    expect(calLabel(A, chains)).toBe("Calibrated snap-fit chain · pitch 12.987 mm");
    expect(calLabel({ base: "snap", pitch_mm: 13.1, half_width_mm: 4.2 }, chains)).toBe("Calibrated snap-fit chain · pitch 13.1 mm · half width 4.2 mm");
  });

  it("can be typed: a pitch, and a half width if it was measured, checked as a card's", () => {
    expect(typedChain("snap", "12,98", "", chains, 5)).toEqual({ base: "snap", pitch_mm: 12.98, via: "typed", at: 5 });
    expect(typedChain("snap", " 12.9874 ", "4.4", chains, 5)).toEqual({ base: "snap", pitch_mm: 12.987, half_width_mm: 4.4, via: "typed", at: 5 });
    const refused = (p: string, h = "") => { try { typedChain("snap", p, h, chains, 0); } catch (e) { return (e as ChainLinkError).message; } return null; };
    expect(refused("")).toBe("Type the chain's pitch in mm, measured over several links.");
    expect(refused("14")).toMatch(/^The pitch, 14 mm, is more than 2% from the snap chain's 13 mm/);
    expect(refused("13", "5")).toMatch(/^The half width, 5 mm, is more than 10%/);
    expect(refused("13 mm")).toBe("The pitch (13 mm) is not a number of millimetres, such as 12.987.");
  });

  it("are read back from storage only if whole", () => {
    expect(storedCalibration(A, chains)).toEqual(A);
    expect(storedCalibration({ ...A, via: "qr" }, null)).toBeNull();
    expect(storedCalibration({ ...A, pitch_mm: 15 }, chains)).toBeNull();
    expect(storedCalibration({ ...A, pitch_mm: 15 }, null)).not.toBeNull();     // checked against the presets once they are loaded
    expect(storedCalibration("1;base=snap", null)).toBeNull();
  });

  it("give a photo their numbers to keep", () => {
    expect(photoCalibration(card("1;base=snap;p=13.1;k=v"), chains)).toEqual({ key: "snap|13.1|4.39|||", pitch_mm: 13.1, half_width_mm: 4.39 });
  });
});

describe("settings with a calibrated chain", () => {
  it("use it: preset, colours, channel and pins from the card; its numbers over the preset's", () => {
    const s = useChain({ ...DEFAULT_SETTINGS, pins: 9 }, card("1;base=snap;p=13.2;h=4.4;shade=dark;ch=b;n=40;k=g"), chains, 7);
    expect([s.chain, s.shade, s.channel, s.pins, s.chains.length, s.chains[0]!.at]).toEqual(["snap", "dark", "b", 40, 1, 7]);
    expect(activeChain(s, chains)?.pitch_mm).toBe(13.2);
    const photo = newPhotoChain(s, chains);
    expect(photo).toEqual({ chain: "snap", cal: { key: s.cal, pitch_mm: 13.2, half_width_mm: 4.4 } });
    expect(chainSpec(chains, photo.chain, 40, "dark", "b", photo.cal)).toEqual({ ...chains.snap, pitch_mm: 13.2, half_width_mm: 4.4,
      n_pins: 40, fiducial_shade: "dark", gray_channel: "b" });
    expect(settingsChainLabel(s, chains)).toBe("Calibrated snap-fit chain · pitch 13.2 mm · half width 4.4 mm");
  });

  it("leave the colours a card does not give, and bring the preset's pin count", () => {
    const s = useChain({ ...DEFAULT_SETTINGS, shade: "dark", channel: "r", pins: 9 }, A, chains);
    expect([s.shade, s.channel, s.pins]).toEqual(["dark", "r", null]);
  });

  it("measure with the preset's numbers when none is in use", () => {
    expect(newPhotoChain(DEFAULT_SETTINGS, chains)).toEqual({ chain: "snap" });
    const s = { ...useChain(DEFAULT_SETTINGS, A, chains), cal: null };
    expect(activeChain(s, chains)).toBeUndefined();
    expect(settingsChainLabel(s, chains)).toBe("Snap-fit chain");
  });

  it("survive a round trip, and a calibration in use outside debug mode is named", () => {
    const store = new MemoryStore();
    const s = useChain(DEFAULT_SETTINGS, A, chains, 3);
    saveSettings(store, s);
    expect(loadSettings(store)).toEqual(s);
    expect(debugLeftovers(s, chains)).toEqual(["Calibrated snap-fit chain · pitch 12.987 mm"]);
    store.set("phone-settings", JSON.stringify({ ...s, chains: [{ base: "snap" }, "x"] }));
    expect(loadSettings(store)).toEqual({ ...s, chains: [], cal: null });
  });
});

describe("the phone's chain on the desktop page", () => {
  it("goes as its card link, and is used there, marked as the phone's", () => {
    const sent = linkText(A, chains);
    expect(sent).toBe("1;base=snap;p=12.987;h=4.39;k=k");
    const desk = withPhoneChain(DEFAULT_SETTINGS, parseLink(sent, chains), chains, 9);
    expect(activeChain(desk, chains)).toMatchObject({ pitch_mm: 12.987, via: "phone", at: 9 });
    expect(settingsChainLabel(desk, chains)).toBe("Calibrated snap-fit chain · pitch 12.987 mm · from the phone");
    // The phone back on the preset's numbers: the phone's chain is no longer used here, but stays listed.
    const back = withPhoneChain(desk, null, chains);
    expect([back.cal, back.chains.length]).toEqual([null, 1]);
    // A chain chosen here is left alone.
    const own = useChain(DEFAULT_SETTINGS, { ...A, pitch_mm: 13.1, via: "typed" }, chains);
    expect(withPhoneChain(own, null, chains)).toBe(own);
  });
});
