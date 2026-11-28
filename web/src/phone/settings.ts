// The phone page's settings, kept in localStorage (store.ts) on this phone.
//
// The camera's focal length decides how well perspective is removed. Its
// sources, in the order the core uses them (session.Options): an override
// (debug mode), the photo's EXIF, the focal length saved here for photos that
// arrive without EXIF (iOS often strips it, S2), and last the estimate from
// the chain. So that the person rarely has to look the number up, the page
// remembers the focal length of every lens a photo with EXIF came from
// (lenses.ts, D17), and with none typed the saved value is the newest main
// camera's (lenses.savedFocal).
//
// Settings for testing, not for using the page, are shown only in debug mode
// (`?debug`, remembered until Leave debug mode): the chain preset, the colour
// channel detection reads, the live stream's size, the focal override and the
// chain-only live view.

import type { CameraInfo } from "../camera/choose";
import type { ChainSpec, Side } from "../core/types";
import { readJson, type KeyValueStore } from "../relay/store";
import { KEEP_PHOTOS } from "./photos";
import type { CameraMemory, LensMemory, LensState } from "./lenses";

export interface Settings extends LensState {
  /** Key of the chain preset in chains.json. */
  chain: string;
  /** Pins in the chain for new photos; null means the preset's own count. */
  pins: number | null;
  /** This phone's main camera in 35 mm terms, used for photos without EXIF. */
  savedFocal35: number | null;
  /** Used for every photo, EXIF or not. For a lens the EXIF gets wrong (debug mode). */
  focalOverride35: number | null;
  /** The object side new photos start with. */
  side: Side;
  /** The live view's camera, as last chosen (O10). */
  camera: CameraInfo | null;
  /** The chain's colours: light dots or rims on dark links (as the presets print),
   * or dark ones on light links (ChainSpec fiducial_shade). */
  shade: Shade;
  /** What of a photo's colour detection reads (ChainSpec gray_channel): the gray (luma),
   * or red, green or blue alone, for coloured rims (debug mode). */
  channel: GrayChannel;
  /** The live camera's stream: "fast" (1440×1920 once the camera's focal length
   * is known, camera/stream.ts) or always the "full" sensor. */
  liveSize: "fast" | "full";
  /** Cameras (by label) whose takePhoto still came back small with the fast stream: full size for them. */
  fullStreamCameras: string[];
  /** What the live camera shows under the chain it finds: the video in colour, the video
   * grey and dimmed (the default: the trace stands out), or nothing ("chain"). */
  liveView: LiveView;
  /** How many of the newest photos keep their photo; older ones keep only their curve (photos.ts cleanupPlan). */
  keepPhotos: number;
  /** Debug mode: the settings for testing are shown (`?debug`). */
  debug: boolean;
}

/** The choices for Settings, Photos kept. */
export const KEEP_PHOTO_CHOICES = [1, 3, 10, 30] as const;

export type Shade = "light" | "dark";
export type LiveView = "color" | "grey" | "chain";
const LIVE_VIEWS: readonly LiveView[] = ["color", "grey", "chain"];

export const DEFAULT_SETTINGS: Settings = {
  chain: "snap",               // the snap-fit chain is the default (splinewire.chain.DEFAULT_PRESET)
  pins: null,
  savedFocal35: null,
  focalOverride35: null,
  side: "inside",
  lenses: [],
  cameras: {},
  camera: null,
  shade: "light",
  channel: "luma",
  liveSize: "fast",
  fullStreamCameras: [],
  liveView: "grey",
  keepPhotos: KEEP_PHOTOS,
  debug: false,
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
  // Earlier builds also kept `declinedOffer` (the offer of a learned lens
  // as the saved value, now taken without asking) and `photoMode` (the
  // Live/Upload switch the redesign replaced with two keys): both dropped.
  s.lenses = Array.isArray(raw.lenses) ? raw.lenses.filter(isLens) : [];
  s.cameras = {};
  if (raw.cameras && typeof raw.cameras === "object") {
    for (const [label, c] of Object.entries(raw.cameras as Record<string, unknown>)) if (isCameraMemory(c)) s.cameras[label] = c;
  }
  const cam = raw.camera as CameraInfo | null;
  s.camera = cam && typeof cam === "object" && typeof cam.deviceId === "string" && typeof cam.label === "string"
    ? { deviceId: cam.deviceId, label: cam.label } : null;
  if (raw.shade === "dark") s.shade = "dark";
  if (isChannel(raw.channel)) s.channel = raw.channel;
  if (raw.liveSize === "full") s.liveSize = "full";
  // Builds 31-32 kept `liveShows` ("video" or "chain"), saved whether chosen or not:
  // only a "chain" was chosen, so "video" takes the new default.
  if (LIVE_VIEWS.includes(raw.liveView as LiveView)) s.liveView = raw.liveView as LiveView;
  else if (raw.liveShows === "chain") s.liveView = "chain";
  s.fullStreamCameras = Array.isArray(raw.fullStreamCameras) ? raw.fullStreamCameras.filter((c): c is string => typeof c === "string").slice(0, 20) : [];
  if ((KEEP_PHOTO_CHOICES as readonly unknown[]).includes(raw.keepPhotos)) s.keepPhotos = raw.keepPhotos as number;
  s.debug = raw.debug === true;
  return s;
}

export function saveSettings(store: KeyValueStore, s: Settings): void {
  store.set(KEY, JSON.stringify(s));
}

/** Debug mode from the page's address: `?debug` turns it on, `?debug=0` off;
 * null when the address does not say. */
export function debugFromQuery(search: string): boolean | null {
  const q = new URLSearchParams(search);
  if (!q.has("debug")) return null;
  return !["0", "false", "off"].includes(q.get("debug")!.toLowerCase());
}

/** The debug-only settings that differ from their defaults, in words: said
 * outside debug mode, where they cannot be seen but still apply. */
export function debugLeftovers(s: Settings, chains: Record<string, ChainSpec>): string[] {
  const out: string[] = [];
  const first = Object.keys(chains)[0];
  if (first !== undefined && presetKey(chains, s.chain) !== first) out.push(chainLabel(presetKey(chains, s.chain)));
  if (s.focalOverride35 !== null) out.push(`focal override ${s.focalOverride35} mm`);
  if (s.liveSize === "full") out.push("full-resolution live stream");
  if (s.liveView === "chain") out.push("chain-only live view");
  if (s.channel !== "luma") out.push(`${CHANNEL_TEXT[s.channel].toLowerCase()} channel`);
  return out;
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

/** The chain the core measures a photo with: the preset, with the photo's pin
 * count, colours and channel (a photo stored before the colours could be chosen
 * is light on dark, and one stored before the channel could be, gray). */
export function chainSpec(chains: Record<string, ChainSpec>, preset: string, pins: number, shade: Shade = "light",
  channel: GrayChannel = "luma"): ChainSpec {
  const spec = chains[presetKey(chains, preset)];
  if (!spec) throw new Error("no chain presets in this build (chains.json is empty)");
  return { ...spec, n_pins: pins, fiducial_shade: shade, gray_channel: channel };
}

export type GrayChannel = NonNullable<ChainSpec["gray_channel"]>;
export const GRAY_CHANNELS: readonly GrayChannel[] = ["luma", "r", "g", "b"];

/** How the channels are called on the page. */
export const CHANNEL_TEXT: Record<GrayChannel, string> = { luma: "Gray", r: "Red", g: "Green", b: "Blue" };

function isChannel(v: unknown): v is GrayChannel {
  return (GRAY_CHANNELS as readonly unknown[]).includes(v);
}

/** How the colours are called on the page. */
export const SHADE_TEXT: Record<Shade, string> = {
  light: "Light dots or rims on dark links",
  dark: "Dark dots or rims on light links",
};

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
