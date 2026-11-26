import { describe, expect, it } from "vitest";
import type { Exif } from "./exif";
import { cameraFocal, frameFocal, isUltraWide, learn, lensFocal, lensKey, mainLens, savedFocal, ultraWides, type LensState } from "./lenses";

const ex = (e: Partial<Exif>): Exif => ({ make: null, model: null, lens: null, focalMm: null, focal35: null, ...e });
const empty = (): LensState & { savedFocal35: number | null } =>
  ({ lenses: [], cameras: {}, savedFocal35: null });

// What the phones handed over (web-build.md, S2/S3).
const pixelLibrary = ex({ make: "Google", model: "Pixel 7 Pro", focalMm: 6.81, focal35: 24, lens: "Pixel 7 Pro back camera 6.81mm f/1.85" });
const pixelTakePhoto = ex({ make: "Google", model: "Pixel 7 Pro", focalMm: 6.81 });
const iphoneTakePhoto = ex({ make: "Apple", model: "iPhone 16 Pro", focalMm: 6.765, focal35: 24, lens: "iPhone 16 Pro back triple camera 6.765mm f/1.78" });
const selfie = ex({ make: "Apple", model: "iPhone 16 Pro", focalMm: 2.69, focal35: 30, lens: "iPhone 16 Pro front TrueDepth camera 2.69mm f/1.9" });
const samsungWide = ex({ make: "samsung", model: "SM-S918U", focalMm: 2.2, focal35: 13 });
const samsungMain = ex({ make: "samsung", model: "SM-S918U", focalMm: 6.3, focal35: 23 });

describe("focal length per lens", () => {
  it("is keyed by make, model and focal length, so a takePhoto still finds what a library photo taught", () => {
    expect(lensKey(pixelLibrary)).toBe(lensKey(pixelTakePhoto));
    expect(lensKey(ex({ lens: "Some lens" }))).toBe("Some lens");
    expect(lensKey(ex({ make: "Google" }))).toBeNull();
    const s = learn(empty(), pixelLibrary, { width: 4080, height: 3072 }, { now: 1 });
    expect(lensFocal(s, pixelTakePhoto)).toEqual({ focal35: 24, size: { width: 4080, height: 3072 }, lens: "Pixel 7 Pro back camera 6.81mm f/1.85" });
    expect(lensFocal(s, iphoneTakePhoto)).toBeNull();                      // has its own 35 mm tag
    expect(lensFocal(s, ex({ make: "Google", model: "Pixel 7 Pro", focalMm: 2.2 }))).toBeNull();   // another lens
    expect(lensFocal(empty(), pixelTakePhoto)).toBeNull();                 // nothing taught yet: settings, then the estimate
  });

  it("learns nothing without the photo's size (a photo the browser could not read), which frames would divide by", () => {
    const s = empty();
    expect(learn(s, pixelLibrary, { width: 0, height: 0 }).lenses).toEqual([]);
  });

  it("never learns from a front camera", () => {
    expect(learn(empty(), selfie, { width: 4032, height: 3024 })).toEqual(empty());
    const s = empty();
    expect(learn(s, pixelLibrary, { width: 1, height: 1 }, { facing: "user" })).toBe(s);
    expect(learn(s, pixelLibrary, { width: 1, height: 1 }, { cameraLabel: "camera 1, facing front" })).toBe(s);
  });

  it("ties a live-view camera to its lens, for the frames", () => {
    let s = learn(empty(), pixelLibrary, { width: 4080, height: 3072 }, { now: 1 });
    s = learn(s, pixelTakePhoto, { width: 4032, height: 3024 }, { cameraLabel: "camera 0, facing back", now: 2 });
    expect(cameraFocal(s, "camera 0, facing back")).toMatchObject({ focal35: 24, lens: "Pixel 7 Pro back camera 6.81mm f/1.85" });
    expect(cameraFocal(s, "camera 2, facing back")).toBeNull();
    s = learn(s, iphoneTakePhoto, { width: 4032, height: 3024 }, { cameraLabel: "Back Triple Camera", now: 3 });
    expect(cameraFocal(s, "Back Triple Camera")?.focal35).toBe(24);
    expect(s.lenses).toHaveLength(2);
  });

  it("spots an ultra-wide by its 35 mm or real focal length", () => {
    expect(isUltraWide(13, 2.2)).toBe(true);
    expect(isUltraWide(null, 2.2)).toBe(true);
    expect(isUltraWide(23, 6.3)).toBe(false);
    expect(isUltraWide(24, null)).toBe(false);
    expect(isUltraWide(null, null)).toBe(false);
    let s = learn(empty(), samsungWide, { width: 4000, height: 3000 }, { cameraLabel: "camera 2, facing back" });
    expect([...ultraWides(s)]).toEqual(["camera 2, facing back"]);
    s = learn(s, samsungMain, { width: 4000, height: 3000 }, { cameraLabel: "camera 0, facing back" });
    expect([...ultraWides(s)]).toEqual(["camera 2, facing back"]);
  });

  it("takes the newest main lens as the saved value, never an ultra-wide or a telephoto, unless one is typed", () => {
    let s = learn(empty(), samsungWide, { width: 4000, height: 3000 }, { now: 1 });
    expect(mainLens(s)).toBeNull();                                         // an ultra-wide is not a main camera
    expect(savedFocal(s)).toBeNull();
    s = learn(s, pixelLibrary, { width: 4080, height: 3072 }, { now: 2 });
    expect(savedFocal(s)).toMatchObject({ focal35: 24, lens: { name: "Pixel 7 Pro back camera 6.81mm f/1.85" } });
    const tele = ex({ make: "Apple", model: "iPhone 16 Pro", focalMm: 15.66, focal35: 120, lens: "iPhone 16 Pro back triple camera 15.66mm f/2.8" });
    const twoX = ex({ make: "Apple", model: "iPhone 16 Pro", focalMm: 6.765, focal35: 48, lens: "iPhone 16 Pro back triple camera 6.765mm f/1.78" });
    s = learn(learn(s, tele, { width: 4032, height: 3024 }, { now: 3 }), twoX, { width: 4032, height: 3024 }, { now: 4 });
    expect(savedFocal(s)?.focal35).toBe(24);                                // newer, but telephotos (5× and the 2× crop)
    expect(savedFocal({ ...s, savedFocal35: 26 })).toEqual({ focal35: 26, lens: null });   // typed wins
    expect(savedFocal(learn(s, samsungMain, { width: 4000, height: 3000 }, { now: 5 }))?.focal35).toBe(23);
  });

  it("is not changed by the same photo twice", () => {
    const s = learn(empty(), pixelLibrary, { width: 4080, height: 3072 }, { now: 1 });
    expect(learn(s, pixelLibrary, { width: 4080, height: 3072 }, { now: 2 })).toBe(s);
  });
});

describe("the focal length of the live view's frames", () => {
  const settings = (o: Partial<{ savedFocal35: number | null; focalOverride35: number | null }> = {}) =>
    ({ ...empty(), focalOverride35: null, ...o });
  const frame = { width: 3024, height: 4032 };

  it("comes from the camera's last still, else the saved setting, else none (estimate)", () => {
    expect(frameFocal(settings(), "Back Camera", frame)).toBeNull();
    const learned = { ...settings(), ...learn(empty(), pixelLibrary, { width: 4080, height: 3072 }) };
    expect(frameFocal(learned, "Back Camera", frame)).toMatchObject({ from: "saved", focal35: 24, lens: null });   // the main camera, untyped
    expect(frameFocal(settings({ savedFocal35: 26 }), "Back Camera", frame)).toMatchObject({ from: "saved", focal35: 26, lens: null });
    const s = { ...settings({ savedFocal35: 26 }), ...learn(empty(), iphoneTakePhoto, { width: 4032, height: 3024 }, { cameraLabel: "Back Camera" }) };
    const f = frameFocal(s, "Back Camera", frame)!;
    expect(f).toMatchObject({ from: "lens", lens: iphoneTakePhoto.lens });
    expect(f.focal35).toBeCloseTo(24, 9);
    expect(f.px).toBeCloseTo(24 * 5040 / Math.hypot(36, 24), 6);
    expect(frameFocal({ ...s, focalOverride35: 28 }, "Back Camera", frame)).toMatchObject({ from: "override", focal35: 28 });
  });

  it("is scaled to a 16:9 frame by its width, not its diagonal", () => {
    const f = frameFocal(settings({ savedFocal35: 24 }), null, { width: 1920, height: 1080 })!;
    expect(f.px).toBeCloseTo(24 * 5040 / Math.hypot(36, 24) * 1920 / 4032, 6);
    expect(f.focal35).toBeGreaterThan(24);
  });
});
