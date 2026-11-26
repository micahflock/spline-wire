// Focal lengths remembered per lens (web-build.md D17), so that a photo or a
// video frame without a usable 35 mm focal length can still be measured with
// the right one.
//
// What the phones hand over (S2/S3, 2026-11-17):
//   iPhone takePhoto()   35 mm tag 24 and LensModel: nothing to look up.
//   Pixel takePhoto()    Make, Model and FocalLength 6.81 mm, but the 35 mm tag
//                        is 0 and there is no LensModel. The same phone's
//                        library photos have both (24 mm, "Pixel 7 Pro back
//                        camera 6.81mm f/1.85").
//   Firefox              no takePhoto: the still is a video frame, no EXIF at all.
//   file input camera    no focal length on iPhone, no EXIF at all on Pixel.
// So a lens is keyed by Make + Model + FocalLength (mm, to 0.01): a library
// photo teaches the 35 mm equivalent once, and a takePhoto still of the same
// lens finds it. A camera of the live view (by its label) is tied to the key
// its last still had, which gives the frames their focal length too.
//
// Never from a front camera: a selfie's 30 mm (S2) applied to the back
// camera's 24 mm would be a 25% focal error.

import { isFrontLabel } from "../camera/choose";
import { frame35, frameFocalPx, type Size } from "../camera/focal";
import type { Exif } from "./exif";

export interface LensMemory {
  key: string;
  /** For people: the EXIF lens name if there is one. */
  name: string;
  focal35: number;
  focalMm: number | null;
  /** The size of the photo the 35 mm figure came with. */
  width: number;
  height: number;
  ultraWide: boolean;
  /** ms since 1970. */
  at: number;
}

/** What the last still taken through a camera of the live view said. */
export interface CameraMemory {
  key: string | null;
  focal35: number | null;
  focalMm: number | null;
  ultraWide: boolean;
  at: number;
}

export interface LensState {
  lenses: LensMemory[];
  /** By camera label. */
  cameras: Record<string, CameraMemory>;
}

/** Lenses kept: a phone has a few, a person may use two phones. */
export const KEEP_LENSES = 12;

/** Under these, a still came from an ultra-wide (S23 Ultra: 13 mm, 2.2 mm;
 * main cameras are 23-26 mm, 5.7-6.9 mm). Ultra-wides also distort more
 * (experiments/lens_distortion.py), so the page moves off them. */
export const ULTRA_WIDE = { focal35: 18, focalMm: 3 };

/** Over this, a lens is a telephoto (2× and up: 48, 77, 120 mm; main cameras are 23-28 mm). */
export const TELEPHOTO_35 = 35;

export function isUltraWide(focal35: number | null, focalMm: number | null): boolean {
  return (focal35 !== null && focal35 < ULTRA_WIDE.focal35) || (focalMm !== null && focalMm < ULTRA_WIDE.focalMm);
}

/** The lens's key: Make + Model + FocalLength when they are there, else the
 * lens name; null when the photo says too little to tell lenses apart. */
export function lensKey(e: Exif): string | null {
  const phone = [e.make, e.model].filter(Boolean).join(" ");
  if (phone && e.focalMm !== null) return `${phone} @ ${e.focalMm.toFixed(2)} mm`;
  return e.lens;
}

export function lensName(e: Exif): string {
  if (e.lens) return e.lens;
  const phone = [e.make, e.model].filter(Boolean).join(" ") || "camera";
  return e.focalMm !== null ? `${phone}, ${+e.focalMm.toFixed(2)} mm lens` : phone;
}

export function isFront(e: Exif, facing?: string | null, cameraLabel?: string | null): boolean {
  return facing === "user" || /front/i.test(e.lens ?? "") || (!!cameraLabel && isFrontLabel(cameraLabel));
}

/** After a photo arrives: remember its lens's 35 mm focal length, and for a
 * still from the live view, which lens that camera is. Returns `s` itself when
 * nothing changes. */
export function learn<S extends LensState>(s: S, e: Exif, size: Size, from: { facing?: string | null; cameraLabel?: string | null; now?: number } = {}): S {
  if (isFront(e, from.facing, from.cameraLabel)) return s;
  const key = lensKey(e);
  const at = from.now ?? Date.now();
  let next = s;
  if (key && e.focal35 !== null) {
    const old = s.lenses.find((l) => l.key === key);
    const name = e.lens ?? old?.name ?? lensName(e);
    const same = old && old.focal35 === e.focal35 && old.name === name && old.width === size.width && old.height === size.height;
    if (!same) {
      const mem: LensMemory = { key, name, focal35: e.focal35, focalMm: e.focalMm, width: size.width, height: size.height,
        ultraWide: isUltraWide(e.focal35, e.focalMm), at };
      next = { ...next, lenses: [mem, ...s.lenses.filter((l) => l.key !== key)].slice(0, KEEP_LENSES) };
    }
  }
  if (from.cameraLabel) {
    const focal35 = e.focal35 ?? (key ? next.lenses.find((l) => l.key === key)?.focal35 ?? null : null);
    const cam: CameraMemory = { key, focal35, focalMm: e.focalMm, ultraWide: isUltraWide(focal35, e.focalMm), at };
    next = { ...next, cameras: { ...next.cameras, [from.cameraLabel]: cam } };
  }
  return next;
}

export interface KnownFocal {
  focal35: number;
  /** The size the 35 mm figure holds for. */
  size: Size;
  lens: string;
}

/** The 35 mm focal length for a photo whose own is missing (a Pixel's
 * takePhoto still): the lens remembered under its key. f35 is defined on the
 * image diagonal, so the core applies it to the photo's own size; a library
 * photo of 4080×3072 and a takePhoto still of 4032×3024 then get the same
 * focal length per diagonal pixel. Whether 4032 is a crop or a scale of 4080
 * is not known; the two differ by 1.2%. */
export function lensFocal(s: LensState, e: Exif): KnownFocal | null {
  if (e.focal35 !== null) return null;
  const key = lensKey(e);
  const lens = key ? s.lenses.find((l) => l.key === key) : undefined;
  return lens ? { focal35: lens.focal35, size: { width: lens.width, height: lens.height }, lens: lens.name } : null;
}

/** The focal length of a live-view camera, from the last still it took. */
export function cameraFocal(s: LensState, cameraLabel: string): KnownFocal | null {
  const cam = s.cameras[cameraLabel];
  if (!cam?.key) return null;
  const lens = s.lenses.find((l) => l.key === cam.key);
  return lens ? { focal35: lens.focal35, size: { width: lens.width, height: lens.height }, lens: lens.name } : null;
}

/** Labels of cameras a still showed to be ultra-wides. */
export function ultraWides(s: LensState): Set<string> {
  return new Set(Object.entries(s.cameras).filter(([, c]) => c.ultraWide).map(([label]) => label));
}

/** The newest lens learned that is a main camera: not an ultra-wide, not a
 * telephoto. A photo without EXIF is taken to come from it (the tips say
 * shoot at 1×): within the EXIF's ~5% (real-world-tests.md, phase 1) it costs
 * ≤0.12 mm at 25° tilt, where the estimate from the chain gives 0.17-0.55 mm
 * and is weak for a nearly straight chain (architecture.md, self_rectification). */
export function mainLens(s: LensState): LensMemory | null {
  return s.lenses.filter((l) => !l.ultraWide && l.focal35 <= TELEPHOTO_35).sort((a, b) => b.at - a.at)[0] ?? null;
}

export interface SavedFocal {
  focal35: number;
  /** The lens it was learned from; null when the person typed it. */
  lens: LensMemory | null;
}

/** The focal length for photos without EXIF (session's saved_focal_35mm): the
 * one typed in Settings, else the main camera's as learned (it used to be
 * offered, "use 24 mm from your last photo?", and was always taken). */
export function savedFocal(s: LensState & { savedFocal35: number | null }): SavedFocal | null {
  if (s.savedFocal35 !== null) return { focal35: s.savedFocal35, lens: null };
  const lens = mainLens(s);
  return lens ? { focal35: lens.focal35, lens } : null;
}

export interface FrameFocal {
  /** In px of the frame, for live.analyse. */
  px: number;
  /** As a 35 mm equivalent of the frame's own size: what a frame saved as the still is measured with. */
  focal35: number;
  from: "override" | "lens" | "saved";
  /** The lens it was learned from, for "lens". */
  lens: string | null;
}

/** The focal length for a live-view camera's frames (web-build.md step 5):
 * the override, else the lens of the last still this camera took, else the
 * saved focal length, typed or learned (taken to hold for the full 4:3 sensor), else null: estimate
 * it from the chain. */
export function frameFocal(s: LensState & { savedFocal35: number | null; focalOverride35: number | null }, cameraLabel: string | null,
  frame: Size): FrameFocal | null {
  const known = cameraLabel ? cameraFocal(s, cameraLabel) : null;
  const [f35, still, from, lens]: [number | null, Size | null, FrameFocal["from"], string | null] =
    s.focalOverride35 !== null ? [s.focalOverride35, null, "override", null]
    : known ? [known.focal35, known.size, "lens", known.lens]
    : [savedFocal(s)?.focal35 ?? null, null, "saved", null];
  if (f35 === null) return null;
  return { px: frameFocalPx(f35, still, frame), focal35: frame35(f35, still, frame), from, lens };
}
