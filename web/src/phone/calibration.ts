// The phone's calibrated chains (work queue item 14, D30): chains measured one
// by one, each set up from its card's link (pipeline/chainlink.ts) or typed in
// by hand. An entry is its numbers (no serial number): scanning the same card
// again only makes its entry the active one. The phone keeps the newest
// MAX_CHAINS; the presets (the nominal chains) are always there besides.
//
// A calibration names its preset (`base`) and puts its pitch and half width over
// the preset's; its colours and channel and pin count, when the card gives them,
// are copied into the page's own settings when it is chosen (settings.ts
// useChain), where the person can still change them.

import { checkLink, ChainLinkError, fmtMm, linkValue, type ChainLink, type PresetNumbers } from "../pipeline/chainlink";

export interface Calibration extends ChainLink {
  /** From a card's link, typed in Settings, or (on the desktop page) the paired phone's. */
  via: "card" | "typed" | "phone";
  /** Last chosen or scanned (ms since 1970): the list is newest first. */
  at: number;
}

/** What a photo remembers of the calibrated chain it is measured with: its numbers,
 * so that choosing another chain later does not quietly measure it again differently. */
export interface PhotoCalibration {
  key: string;
  pitch_mm: number;
  half_width_mm: number;
}

export const MAX_CHAINS = 10;

/** How the presets are called on the page. */
const CHAIN_NAMES: Record<string, string> = { snap: "Snap-fit chain", oring: "O-ring chain" };

export function chainLabel(key: string): string {
  return CHAIN_NAMES[key] ?? key;
}

/** The entry's identity: its preset and numbers (the half width resolved to the preset's when
 * not given), and the colours, channel and pins it sets. */
export function calKey(c: ChainLink, presets: PresetNumbers): string {
  const h = c.half_width_mm ?? presets[c.base]?.half_width_mm;
  return [c.base, fmtMm(c.pitch_mm), h === undefined ? "" : fmtMm(h), c.fiducial_shade ?? "", c.gray_channel ?? "", c.n_pins ?? ""].join("|");
}

/** The list with `c` first, made the newest; an entry with the same numbers is replaced. */
export function rememberChain(list: readonly Calibration[], c: Calibration, presets: PresetNumbers): Calibration[] {
  const key = calKey(c, presets);
  return [c, ...list.filter((x) => calKey(x, presets) !== key)].slice(0, MAX_CHAINS);
}

export function forgetChain(list: readonly Calibration[], key: string, presets: PresetNumbers): Calibration[] {
  return list.filter((x) => calKey(x, presets) !== key);
}

export function findChain(list: readonly Calibration[], key: string | null | undefined, presets: PresetNumbers): Calibration | undefined {
  return key ? list.find((x) => calKey(x, presets) === key) : undefined;
}

/** "Calibrated snap-fit chain · pitch 12.987 mm", and the half width when it differs from the preset's. */
export function calLabel(c: { base: string; pitch_mm: number; half_width_mm?: number; via?: Calibration["via"] }, presets: PresetNumbers): string {
  const name = chainLabel(c.base);
  const parts = [`Calibrated ${name.charAt(0).toLowerCase()}${name.slice(1)}`, `pitch ${fmtMm(c.pitch_mm)} mm`];
  const nominal = presets[c.base]?.half_width_mm;
  if (c.half_width_mm !== undefined && (nominal === undefined || fmtMm(c.half_width_mm) !== fmtMm(nominal))) {
    parts.push(`half width ${fmtMm(c.half_width_mm)} mm`);
  }
  if (c.via === "phone") parts.push("from the phone");
  return parts.join(" · ");
}

/** What the entry says, as a card's link says it. */
export function linkOf(c: Calibration): ChainLink {
  const { via: _via, at: _at, ...link } = c;
  return link;
}

/** The entry as a card link's cal value (the relay's chain message carries it). */
export function linkText(c: Calibration, presets: PresetNumbers): string {
  return linkValue(linkOf(c), presets);
}

/** The numbers a photo keeps (PhotoRecord.cal). */
export function photoCalibration(c: Calibration, presets: PresetNumbers): PhotoCalibration {
  return {
    key: calKey(c, presets),
    pitch_mm: c.pitch_mm,
    half_width_mm: c.half_width_mm ?? presets[c.base]?.half_width_mm ?? 0,
  };
}

/** A number typed by the person, a comma allowed for the decimal point; null when empty. */
function typedNumber(text: string, what: string): number | null {
  const t = text.trim().replace(",", ".");
  if (!t) return null;
  if (!/^[0-9]+(\.[0-9]*)?$|^\.[0-9]+$/.test(t)) {
    throw new ChainLinkError("number", `The ${what} (${text.trim()}) is not a number of millimetres, such as 12.987.`);
  }
  return Math.floor(Number(t) * 1000 + 0.5) / 1000;
}

/** An entry from Settings' Type a pitch: the pitch (required) and the half width (optional), checked
 * as a card's are. Throws ChainLinkError with a message for the person. */
export function typedChain(base: string, pitchText: string, halfWidthText: string, presets: PresetNumbers, now: number): Calibration {
  const pitch = typedNumber(pitchText, "pitch");
  if (pitch === null) throw new ChainLinkError("missing", "Type the chain's pitch in mm, measured over several links.");
  const half = typedNumber(halfWidthText, "half width");
  const link: ChainLink = { base, pitch_mm: pitch, ...(half !== null ? { half_width_mm: half } : {}) };
  checkLink(link, presets);
  return { ...link, via: "typed", at: now };
}

/** A calibration from storage, every field checked; null if it is not one. */
export function storedCalibration(v: unknown, presets: PresetNumbers | null): Calibration | null {
  const c = v as Calibration;
  if (!c || typeof c !== "object" || typeof c.base !== "string" || typeof c.pitch_mm !== "number") return null;
  if (c.via !== "card" && c.via !== "typed" && c.via !== "phone") return null;
  if (typeof c.at !== "number" || !Number.isFinite(c.at)) return null;
  const link: ChainLink = { base: c.base, pitch_mm: c.pitch_mm };
  if (c.half_width_mm !== undefined) link.half_width_mm = c.half_width_mm;
  if (c.fiducial_shade !== undefined) link.fiducial_shade = c.fiducial_shade;
  if (c.gray_channel !== undefined) link.gray_channel = c.gray_channel;
  if (c.n_pins !== undefined) link.n_pins = c.n_pins;
  if (presets) {
    try {
      checkLink(link, presets);
    } catch {
      return null;
    }
  }
  return { ...link, via: c.via, at: c.at };
}
