// The phone page's settings, kept in localStorage (store.ts) on this phone.
//
// The camera's focal length decides how well perspective is removed. Its
// sources, in the order the core uses them (session.Options): an override
// typed here, the photo's EXIF, the focal length saved here for photos that
// arrive without EXIF (iOS often strips it, S2), and last the estimate from
// the chain. So that the person rarely has to look the number up, the page
// remembers the focal length of every lens a photo with EXIF came from
// (lenses.ts, D17) and offers the newest as the saved value.

import type { CameraInfo } from "../camera/choose";
import type { ChainSpec, Side } from "../core/types";
import { readJson, type KeyValueStore } from "../relay/store";
import type { CameraMemory, LensMemory, LensState } from "./lenses";

export interface Settings extends LensState {
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
  /** The lens offer the person said no to (lenses.offerKey): not offered again. */
  declinedOffer: string | null;
  /** The live view's camera, as last chosen (O10). */
  camera: CameraInfo | null;
  /** How photos are given: the page's live camera, or uploaded (library or system camera). */
  photoMode: "live" | "upload";
}

export const DEFAULT_SETTINGS: Settings = {
  chain: "snap",               // the snap-fit chain is the default (splinewire.chain.DEFAULT_PRESET)
  pins: null,
  savedFocal35: null,
  focalOverride35: null,
  side: "inside",
  lenses: [],
  cameras: {},
  declinedOffer: null,
  camera: null,
  photoMode: "live",
};

/** session._focal accepts 5 to 500 mm. */
export const FOCAL_RANGE = { min: 5, max: 500 };
/** chain.ChainSpec needs at least 3; more than this is a typing slip. */
export const PIN_RANGE = { min: 3, max: 999 };

/** How the presets are called on the page (the current app's names). */
const CHAIN_NAMES: Record<string, string> = { snap: "Snap-fit chain", oring: "O-ring chain" };

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
  // Before D17 the page kept one focal length, `lastExif`, with the phone's
  // make and model but not the lens. It may have come from a selfie, and no
  // lens can be named for it, so it is dropped (with `declinedFocal35`, the
  // answer to its offer). A value the person saved stays in savedFocal35.
  s.lenses = Array.isArray(raw.lenses) ? raw.lenses.filter(isLens) : [];
  s.cameras = {};
  if (raw.cameras && typeof raw.cameras === "object") {
    for (const [label, c] of Object.entries(raw.cameras as Record<string, unknown>)) if (isCameraMemory(c)) s.cameras[label] = c;
  }
  s.declinedOffer = typeof raw.declinedOffer === "string" ? raw.declinedOffer : null;
  const cam = raw.camera as CameraInfo | null;
  s.camera = cam && typeof cam === "object" && typeof cam.deviceId === "string" && typeof cam.label === "string"
    ? { deviceId: cam.deviceId, label: cam.label } : null;
  if (raw.photoMode === "upload") s.photoMode = "upload";
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

/** The preset to use: the chosen one if this build has it, else the first
 * (chains.json lists the default, the snap-fit chain, first). A setting saved
 * before the presets were renamed ("default", the O-ring chain) is not in it,
 * and so follows the default. */
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

function isFocal(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= FOCAL_RANGE.min && v <= FOCAL_RANGE.max;
}

function isPins(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= PIN_RANGE.min && v <= PIN_RANGE.max;
}

const num = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const numOrNull = (v: unknown) => v === null || num(v);

function isLens(v: unknown): v is LensMemory {
  const l = v as LensMemory;
  return !!l && typeof l === "object" && typeof l.key === "string" && typeof l.name === "string" && isFocal(l.focal35)
    && numOrNull(l.focalMm) && num(l.width) && num(l.height) && typeof l.ultraWide === "boolean" && num(l.at);
}

function isCameraMemory(v: unknown): v is CameraMemory {
  const c = v as CameraMemory;
  return !!c && typeof c === "object" && (c.key === null || typeof c.key === "string") && numOrNull(c.focal35)
    && numOrNull(c.focalMm) && typeof c.ultraWide === "boolean" && num(c.at);
}
