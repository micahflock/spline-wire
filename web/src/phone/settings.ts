// The phone page's settings, kept in localStorage (store.ts) on this phone.
//
// The camera's focal length decides how well perspective is removed. Its
// sources, in the order the core uses them (session.Options): an override
// typed here, the photo's EXIF, the focal length saved here for photos that
// arrive without EXIF (iOS often strips it, S2), and last the estimate from
// the chain. So that the person rarely has to look the number up, the page
// remembers the focal length of any photo that did carry EXIF and offers it
// as the saved value.

import type { ChainSpec, PhotoInfo, Side } from "../core/types";
import { readJson, type KeyValueStore } from "../relay/store";

export interface Settings {
  /** Key of the chain preset in chains.json. */
  chain: string;
  /** Pins in the chain for new photos; null means the preset's own count. */
  pins: number | null;
  /** This phone's main camera in 35 mm terms, used for photos without EXIF. */
  savedFocal35: number | null;
  /** Used for every photo, EXIF or not. For a lens the EXIF gets wrong. */
  focalOverride35: number | null;
  /** The object side new photos start with. */
  side: Side;
  /** The focal length of the last photo that had one in its EXIF. */
  lastExif: { focal35: number; camera: string | null } | null;
  /** A remembered focal length the person said no to: not offered again. */
  declinedFocal35: number | null;
}

export const DEFAULT_SETTINGS: Settings = {
  chain: "default",
  pins: null,
  savedFocal35: null,
  focalOverride35: null,
  side: "inside",
  lastExif: null,
  declinedFocal35: null,
};

/** session._focal accepts 5 to 500 mm. */
export const FOCAL_RANGE = { min: 5, max: 500 };
/** chain.ChainSpec needs at least 3; more than this is a typing slip. */
export const PIN_RANGE = { min: 3, max: 999 };

/** How the presets are called on the page (the current app's names). */
const CHAIN_NAMES: Record<string, string> = { default: "O-ring chain", snap: "Snap-fit chain" };

export function chainLabel(key: string): string {
  return CHAIN_NAMES[key] ?? key;
}

const KEY = "phone-settings";

/** Settings from storage, every field checked: anything odd falls back to its default. */
export function loadSettings(store: KeyValueStore): Settings {
  const raw = readJson<Record<string, unknown>>(store, KEY, {});
  const s = { ...DEFAULT_SETTINGS };
  if (typeof raw.chain === "string" && raw.chain) s.chain = raw.chain;
  s.pins = isPins(raw.pins) ? raw.pins : null;
  s.savedFocal35 = isFocal(raw.savedFocal35) ? raw.savedFocal35 : null;
  s.focalOverride35 = isFocal(raw.focalOverride35) ? raw.focalOverride35 : null;
  if (raw.side === "inside" || raw.side === "outside") s.side = raw.side;
  const ex = raw.lastExif as Settings["lastExif"];
  if (ex && typeof ex === "object" && isFocal(ex.focal35)) {
    s.lastExif = { focal35: ex.focal35, camera: typeof ex.camera === "string" ? ex.camera : null };
  }
  s.declinedFocal35 = isFocal(raw.declinedFocal35) ? raw.declinedFocal35 : null;
  return s;
}

export function saveSettings(store: KeyValueStore, s: Settings): void {
  store.set(KEY, JSON.stringify(s));
}

/** A focal length typed by the person: a number in range, empty for none, or "invalid". */
export function parseFocal(text: string): number | null | "invalid" {
  const t = text.trim().replace(",", ".");
  if (!t) return null;
  const f = Number(t);
  return isFocal(f) ? f : "invalid";
}

export function parsePins(text: string): number | "invalid" {
  const n = Number(text.trim());
  return isPins(n) ? n : "invalid";
}

/** The preset to use: the chosen one if this build has it, else the first. */
export function presetKey(chains: Record<string, ChainSpec>, wanted: string): string {
  return wanted in chains ? wanted : Object.keys(chains)[0] ?? wanted;
}

/** Pins for a new photo: the setting, else the preset's own count. */
export function defaultPins(s: Settings, chains: Record<string, ChainSpec>): number {
  return s.pins ?? chains[presetKey(chains, s.chain)]?.n_pins ?? 13;
}

/** The chain the core measures a photo with: the preset, with the photo's pin count. */
export function chainSpec(chains: Record<string, ChainSpec>, preset: string, pins: number): ChainSpec {
  const spec = chains[presetKey(chains, preset)];
  if (!spec) throw new Error("no chain presets in this build (chains.json is empty)");
  return { ...spec, n_pins: pins };
}

/** After a measurement: remember the focal length if the photo carried one. */
export function rememberExif(s: Settings, info: PhotoInfo): Settings {
  if (!isFocal(info.focal_35mm)) return s;
  const camera = [info.make, info.model].filter(Boolean).join(" ") || null;
  if (s.lastExif?.focal35 === info.focal_35mm && s.lastExif.camera === camera) return s;
  return { ...s, lastExif: { focal35: info.focal_35mm, camera } };
}

/** The focal length to offer as the saved value ("use 26 mm from your last
 * photo?"), or null: there is none, it is already saved, or it was declined. */
export function focalOffer(s: Settings): { focal35: number; camera: string | null } | null {
  const ex = s.lastExif;
  if (!ex || ex.focal35 === s.savedFocal35 || ex.focal35 === s.declinedFocal35) return null;
  return ex;
}

function isFocal(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= FOCAL_RANGE.min && v <= FOCAL_RANGE.max;
}

function isPins(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= PIN_RANGE.min && v <= PIN_RANGE.max;
}
