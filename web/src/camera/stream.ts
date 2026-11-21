// The size the live camera's stream is opened at (web-build.md D16, step 5).
//
// Full: the whole 4:3 sensor (iPhone 3024×4032), so that a still's focal length
// applies to the frames unchanged and no second request at another size is
// needed (Firefox on a Pixel refused one, S3). Each frame then costs the frame
// grab a 12 MP draw: ~48 ms on the owner's iPhone (build 22), most of a frame's
// time.
//
// Fast: 1440×1920, a quarter of the pixels, the same 4:3 field. Only where the
// frames' focal length is known without the stream's full size (an override,
// or the lens of a still this camera took: lenses.frameFocal scales it by the
// long sides) and the still comes from takePhoto, not from a video frame (no
// ImageCapture: Firefox): a small stream must not make a small still. On by
// default (Settings, Live view); people mostly shoot the chain close up, and
// 1920 px reads a chain down to the coach's minSpan (0.22 of the short side:
// its rims ~18 work px at the finest scale, overlay/region.ts).
//
// Whether a phone's takePhoto still keeps its full size with a smaller stream
// is not known: the camera view reads each still's size (jpegSize), and a
// camera whose still came back small opens at full size from then on
// (Settings.fullStreamCameras).

export interface StreamSize {
  width: { ideal: number };
  height: { ideal: number };
}

/** Asked for in landscape terms, as cameras give their sizes; phones deliver portrait. */
export const FULL_STREAM: StreamSize = { width: { ideal: 4032 }, height: { ideal: 3024 } };
export const FAST_STREAM: StreamSize = { width: { ideal: 1920 }, height: { ideal: 1440 } };

/** A still whose long side is under this share of the full stream's is "small". */
export const SMALL_STILL = 0.9;

export interface StreamChoice {
  /** Settings: "fast" (when it may) or "full". */
  liveSize: "fast" | "full";
  /** The frames' focal length is known for this camera (an override or a still's lens). */
  focalKnown: boolean;
  /** The still is taken with takePhoto (ImageCapture), not saved from a frame. */
  imageCapture: boolean;
  /** This camera's still came back small at the fast size once. */
  refused: boolean;
}

/** Which size to open a camera's stream at, and why, for the info line. */
export function streamFor(c: StreamChoice): { size: StreamSize; fast: boolean; why: string } {
  const full = (why: string) => ({ size: FULL_STREAM, fast: false, why });
  if (c.liveSize === "full") return full("full size (Settings)");
  if (!c.imageCapture) return full("full size: the still is a video frame here");
  if (c.refused) return full("full size: this camera's still came back small at the fast size");
  if (!c.focalKnown) return full("full size until this camera's focal length is known");
  return { size: FAST_STREAM, fast: true, why: "fast" };
}

/** Is a still of `still` small next to a full stream of `full` (long sides)? */
export function smallStill(still: { width: number; height: number }, full: StreamSize): boolean {
  return Math.max(still.width, still.height) < SMALL_STILL * Math.max(full.width.ideal, full.height.ideal);
}

/** A JPEG's width and height from its start-of-frame marker, or null (not a JPEG, or cut short). */
export function jpegSize(bytes: Uint8Array): { width: number; height: number } | null {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  let i = 2;
  while (i + 9 < bytes.length) {
    if (bytes[i] !== 0xff) return null;
    const marker = bytes[i + 1]!;
    if (marker === 0xff) {                       // fill byte
      i++;
      continue;
    }
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      i += 2;                                    // markers with no length
      continue;
    }
    const length = (bytes[i + 2]! << 8) | bytes[i + 3]!;
    // SOF0-SOF15, but not DHT (c4), JPG (c8) or DAC (cc).
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      const height = (bytes[i + 5]! << 8) | bytes[i + 6]!, width = (bytes[i + 7]! << 8) | bytes[i + 8]!;
      return width && height ? { width, height } : null;
    }
    if (marker === 0xda) return null;            // scan data before any frame header
    i += 2 + length;
  }
  return null;
}

/** A PNG's width and height from its header, or null. */
export function pngSize(bytes: Uint8Array): { width: number; height: number } | null {
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length < 24 || sig.some((b, i) => bytes[i] !== b)) return null;
  const u32 = (i: number) => ((bytes[i]! << 24) | (bytes[i + 1]! << 16) | (bytes[i + 2]! << 8) | bytes[i + 3]!) >>> 0;
  const width = u32(16), height = u32(20);
  return width && height ? { width, height } : null;
}

/** A still's size: a JPEG (phones) or a PNG (Chromium's fake camera), or null. */
export function stillSize(bytes: Uint8Array): { width: number; height: number } | null {
  return jpegSize(bytes) ?? pngSize(bytes);
}
