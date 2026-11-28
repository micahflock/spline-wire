// A chain card's link: one chain's calibrated numbers in the phone page's address,
//
//   https://micahflock.github.io/spline-wire/phone/#cal=1;base=snap;p=12.987;h=4.39;k=k
//
// The port of splinewire/chainlink.py, which is the reference: the grammar, the
// check character and worked examples are in docs/chain-link.md (D30), and both
// are checked against the same test vectors (chainlink-vectors.json), messages
// and all. A change here is made there too.
//
// A link names its preset (`base`) and carries what differs from chain to chain
// within a run: the pitch, and optionally the half width, the fiducials' shade,
// the colour channel and the pin count. Everything else comes from the preset.

import { GRAY_CHANNELS, makeSpec } from "./chain";
import { PipelineError } from "./errors";
import type { ChainSpec, GrayChannel } from "./types";

export const VERSION = 1;
/** The fragment parameter: #cal=… (beside #pair=…). */
export const PARAM = "cal";
/** A link's pitch is within 2% of its preset's, its half width within 10%. */
export const PITCH_BAND = 0.02;
export const HALF_WIDTH_BAND = 0.1;
export const MIN_PINS = 3;
export const MAX_PINS = 200;
const SHADES = ["light", "dark"] as const;
/** The colour channel the photo's gray image is made from: the spec's gray_channel (work queue item 13). */
export const CHANNELS = GRAY_CHANNELS;
export type { GrayChannel };
export const ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz";
/** The phone page a card opens: ?debug while chain cards are being tested (work queue item 14). */
export const PHONE_PAGE = "https://micahflock.github.io/spline-wire/phone/?debug";

/** Field names in the order the builder writes them, between the version and k. */
const FIELDS = ["base", "p", "h", "shade", "ch", "n"];
const NUMBER = /^(0|[1-9][0-9]*)(\.[0-9]{1,3})?$/;
const NAME = /^[a-z]+$/;
const DIGITS = /^[0-9]+$/;
const RETRY = "scan it again or type the pitch in Settings";

export type ChainLinkErrorCode =
  | "not_a_link" | "damaged" | "version" | "cut_short" | "check" | "unknown_field"
  | "missing" | "unknown_preset" | "number" | "pitch" | "half_width" | "value";

/** A link that cannot be used, with a message for a person (Python's ChainLinkError, a ValueError).
 * `code` says which check failed; the test vectors name it. */
export class ChainLinkError extends PipelineError {
  override name = "ChainLinkError";
  constructor(readonly code: ChainLinkErrorCode, message: string) {
    super(message);
  }
}

/** What a link says. A field left out is not given: the preset's own value is used. */
export interface ChainLink {
  base: string;
  pitch_mm: number;
  half_width_mm?: number;
  fiducial_shade?: "light" | "dark";
  gray_channel?: GrayChannel;
  n_pins?: number;
}

/** The numbers of a preset that a link is checked against (a ChainSpec has them). */
export type PresetNumbers = Record<string, { pitch_mm: number; half_width_mm: number }>;

// --- the check character ------------------------------------------------------

function luhnSum(codes: number[], n: number, doubleLast: boolean): number {
  let total = 0, double = doubleLast;
  for (let i = codes.length - 1; i >= 0; i--) {
    const a = double ? codes[i]! * 2 : codes[i]!;
    total += Math.floor(a / n) + (a % n);
    double = !double;
  }
  return total;
}

const codesOf = (text: string, alphabet: string): number[] =>
  [...text.toLowerCase()].filter((c) => alphabet.includes(c)).map((c) => alphabet.indexOf(c));

/** Luhn mod N (N = the alphabet's length, 36 by default) over the digits and letters of `text`;
 * anything else is skipped. */
export function checkCharacter(text: string, alphabet = ALPHABET): string {
  const n = alphabet.length;
  return alphabet[(n - (luhnSum(codesOf(text, alphabet), n, true) % n)) % n]!;
}

/** Whether `k` is the check character of `text`. */
export function checkOk(text: string, k: string, alphabet = ALPHABET): boolean {
  const n = alphabet.length;
  return k.length === 1 && alphabet.includes(k.toLowerCase()) && luhnSum(codesOf(text + k, alphabet), n, false) % n === 0;
}

// --- numbers ------------------------------------------------------------------

/** x in thousandths, half up (Python's math.floor(x * 1000 + 0.5)). */
const thousandths = (x: number): number => Math.floor(x * 1000 + 0.5);

/** Up to 3 decimals, trailing zeros dropped: 13 -> "13", 12.9870 -> "12.987". */
export function fmtMm(x: number): string {
  let m = thousandths(x);
  const sign = m < 0 ? "-" : "";
  m = Math.abs(m);
  return `${sign}${Math.floor(m / 1000)}.${String(m % 1000).padStart(3, "0")}`.replace(/0+$/, "").replace(/\.$/, "");
}

function number(field: string, what: string, text: string): number {
  if (!NUMBER.test(text)) {
    throw new ChainLinkError("number", `The ${what} (${field}=${text}) is not a number of millimetres with up to 3 decimals, such as 12.987.`);
  }
  return Number(text);
}

// --- checking what a link says ------------------------------------------------

function nominal(presets: PresetNumbers, base: string) {
  if (!Object.hasOwn(presets, base)) {
    throw new ChainLinkError("unknown_preset", `Spline Wire has no chain preset named "${base}" `
      + `(there are ${Object.keys(presets).join(", ")}): if the card is new, reload the page to update Spline Wire.`);
  }
  return presets[base]!;
}

function inBand(what: "pitch" | "half width", value: number, base: string, nom: number, band: number): void {
  if (!(Math.abs(value - nom) <= band * nom + 1e-9)) {
    throw new ChainLinkError(what === "pitch" ? "pitch" : "half_width",
      `The ${what}, ${fmtMm(value)} mm, is more than ${Math.round(band * 100)}% from the ${base} chain's ${fmtMm(nom)} mm: `
      + `it must be between ${fmtMm(nom * (1 - band))} and ${fmtMm(nom * (1 + band))} mm. Check the number, or measure the chain again.`);
  }
}

/** The link if every value is one a card may carry, else a ChainLinkError. */
export function checkLink(link: ChainLink, presets: PresetNumbers): ChainLink {
  const nom = nominal(presets, link.base);
  inBand("pitch", link.pitch_mm, link.base, nom.pitch_mm, PITCH_BAND);
  if (link.half_width_mm !== undefined) inBand("half width", link.half_width_mm, link.base, nom.half_width_mm, HALF_WIDTH_BAND);
  if (link.fiducial_shade !== undefined && !(SHADES as readonly string[]).includes(link.fiducial_shade)) {
    throw new ChainLinkError("value", `The colours (shade=${link.fiducial_shade}) must be light or dark.`);
  }
  if (link.gray_channel !== undefined && !(CHANNELS as readonly string[]).includes(link.gray_channel)) {
    throw new ChainLinkError("value", `The colour channel (ch=${link.gray_channel}) must be luma, r, g or b.`);
  }
  if (link.n_pins !== undefined && !(Number.isInteger(link.n_pins) && link.n_pins >= MIN_PINS && link.n_pins <= MAX_PINS)) {
    throw new ChainLinkError("value", `The number of pins (n=${link.n_pins}) must be a whole number from ${MIN_PINS} to ${MAX_PINS}.`);
  }
  return link;
}

// --- building -----------------------------------------------------------------

/** The `cal` parameter's value, "1;base=snap;p=12.987;h=4.39;k=k". Numbers are rounded to
 * 3 decimals (half up) first, then checked. */
export function linkValue(link: ChainLink, presets: PresetNumbers): string {
  for (const [what, x] of [["pitch", link.pitch_mm], ["half width", link.half_width_mm]] as const) {
    if (x !== undefined && !Number.isFinite(x)) {
      throw new ChainLinkError("number", `The ${what} must be a number of millimetres, such as 12.987.`);
    }
  }
  const l: ChainLink = { ...link, pitch_mm: thousandths(link.pitch_mm) / 1000 };
  if (l.half_width_mm !== undefined) l.half_width_mm = thousandths(l.half_width_mm) / 1000;
  checkLink(l, presets);
  const fields = [String(VERSION), `base=${l.base}`, `p=${fmtMm(l.pitch_mm)}`];
  if (l.half_width_mm !== undefined) fields.push(`h=${fmtMm(l.half_width_mm)}`);
  if (l.fiducial_shade !== undefined) fields.push(`shade=${l.fiducial_shade}`);
  if (l.gray_channel !== undefined) fields.push(`ch=${l.gray_channel}`);
  if (l.n_pins !== undefined) fields.push(`n=${l.n_pins}`);
  const body = fields.join(";");
  return `${body};k=${checkCharacter(body)}`;
}

/** The card's address: the phone page with #cal=<value>. */
export function linkUrl(link: ChainLink, presets: PresetNumbers, page = PHONE_PAGE): string {
  return `${page.split("#")[0]}#${PARAM}=${linkValue(link, presets)}`;
}

// --- parsing ------------------------------------------------------------------

const notALink = () => new ChainLinkError("not_a_link",
  "This is not a chain card's link (it has no cal=): scan the card's QR code again or type the pitch in Settings.");

/** The `cal` value in a link, a fragment or the value alone, percent-decoded
 * (decodeURIComponent: a bad escape throws URIError). */
export function linkText(text: string): string {
  text = text.trim();
  if (text.includes("#") || text.toLowerCase().startsWith(`${PARAM}=`)) {
    const fragment = text.includes("#") ? text.slice(text.indexOf("#") + 1) : text;
    for (const part of fragment.split("&")) {
      const eq = part.indexOf("=");
      if (eq >= 0 && part.slice(0, eq).toLowerCase() === PARAM) return decodeURIComponent(part.slice(eq + 1)).trim();
    }
  } else if (/^[0-9]/.test(text)) {
    return decodeURIComponent(text);
  }
  throw notALink();
}

/** Whether an address's fragment carries a chain card's link (to be read, and stripped, by the page). */
export function hasLink(hash: string): boolean {
  return hash.replace(/^#/, "").split("&").some((p) => p.toLowerCase().startsWith(`${PARAM}=`));
}

/** What a chain card's link says, checked; a ChainLinkError if it cannot be used.
 * `text` is the card's address, its fragment, or the cal value alone. */
export function parseLink(text: string, presets: PresetNumbers): ChainLink {
  let value: string;
  try {
    value = linkText(text).toLowerCase();
  } catch (err) {
    if (err instanceof URIError) throw new ChainLinkError("damaged", `This chain card's link is damaged (a bad % escape): ${RETRY}.`);
    throw err;
  }
  const fields = value.split(";");
  const first = fields[0]!;
  if (!DIGITS.test(first)) {
    throw new ChainLinkError("damaged", `This chain card's link is damaged (it does not start with a version number): ${RETRY}.`);
  }
  const version = Number(first);
  if (version > VERSION) {
    throw new ChainLinkError("version", `This chain card was made for a newer version of Spline Wire (link version ${first.replace(/^0+/, "")}): reload the page to update it.`);
  }
  if (first !== String(VERSION)) {
    throw new ChainLinkError("damaged", `This chain card's link is damaged (version ${first}): ${RETRY}.`);
  }
  const last = fields.length > 1 ? fields[fields.length - 1]! : "";
  const k = last.slice(2);
  if (!(last.startsWith("k=") && k.length === 1 && ALPHABET.includes(k))) {
    throw new ChainLinkError("cut_short",
      "This chain card's link is cut short (it does not end in a check character, ;k=): scan it again or copy the whole link.");
  }
  if (!checkOk(fields.slice(0, -1).join(";"), k)) {
    throw new ChainLinkError("check", `This chain card's code is damaged (check character): ${RETRY}.`);
  }

  const given = new Map<string, string>();
  for (const f of fields.slice(1, -1)) {
    const eq = f.indexOf("=");
    const name = eq >= 0 ? f.slice(0, eq) : f, v = eq >= 0 ? f.slice(eq + 1) : "";
    if (!f) throw new ChainLinkError("damaged", `This chain card's link is damaged (an empty field): ${RETRY}.`);
    if (eq >= 0 && NAME.test(name) && !v) throw new ChainLinkError("damaged", `This chain card's link is damaged (${name} has no value): ${RETRY}.`);
    if (!(eq >= 0 && NAME.test(name))) throw new ChainLinkError("damaged", `This chain card's link is damaged (a field "${f}"): ${RETRY}.`);
    if (given.has(name)) throw new ChainLinkError("damaged", `This chain card's link is damaged (${name} is given twice): ${RETRY}.`);
    given.set(name, v);
  }
  const unknown = [...given.keys()].filter((n) => !FIELDS.includes(n));
  if (unknown.length) {
    throw new ChainLinkError("unknown_field", "This chain card's link has a field this version of Spline Wire "
      + `does not know (${unknown[0]}=${given.get(unknown[0]!)}): reload the page to update it, or type the pitch in Settings.`);
  }
  const base = given.get("base"), p = given.get("p");
  if (base === undefined) throw new ChainLinkError("missing", `This chain card's link has no chain preset (base=): ${RETRY}.`);
  if (p === undefined) throw new ChainLinkError("missing", `This chain card's link has no pitch (p=): ${RETRY}.`);
  nominal(presets, base);
  const n = given.get("n");
  if (n !== undefined && !/^[1-9][0-9]*$/.test(n)) {
    throw new ChainLinkError("value", `The number of pins (n=${n}) must be a whole number from ${MIN_PINS} to ${MAX_PINS}.`);
  }
  const link: ChainLink = { base, pitch_mm: number("p", "pitch", p) };
  const h = given.get("h"), shade = given.get("shade"), ch = given.get("ch");
  if (h !== undefined) link.half_width_mm = number("h", "half width", h);
  if (shade !== undefined) link.fiducial_shade = shade as ChainLink["fiducial_shade"];
  if (ch !== undefined) link.gray_channel = ch as GrayChannel;
  if (n !== undefined) link.n_pins = Number(n);
  return checkLink(link, presets);
}

// --- using it -----------------------------------------------------------------

/** The preset's spec with the link's numbers and colours over it, checked. */
export function applyLink(spec: ChainSpec, link: ChainLink): ChainSpec {
  return makeSpec({
    ...spec,
    pitch_mm: link.pitch_mm,
    half_width_mm: link.half_width_mm ?? spec.half_width_mm,
    fiducial_shade: link.fiducial_shade ?? spec.fiducial_shade,
    gray_channel: link.gray_channel ?? spec.gray_channel,
    n_pins: link.n_pins ?? spec.n_pins,
  });
}
