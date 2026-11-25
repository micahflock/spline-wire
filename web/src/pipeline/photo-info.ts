// What the TypeScript engine (src/core/engine.ts) reports of a photo, as
// splinewire.session.photo_info does, and the focal length it measures with
// when the photo's EXIF gives one (camera.focal_px_from_exif).

import type { PhotoInfo } from "../core/types";
import { readExif } from "../phone/exif";
import type { Gray } from "./cv";
import { focalPxFrom35mm } from "./session";

/** The photo's info and its EXIF focal length in pixels, from its bytes and its decoded, upright gray image. */
export function photoInfo(bytes: Uint8Array, gray: Gray): { info: PhotoInfo; focalPx: number | null } {
  const exif = readExif(bytes);
  // The stored size, before turning upright (Pillow's im.size); the focal length
  // depends only on the diagonal, which turning keeps. A HEIF photo is turned by its
  // container, not its EXIF (pillow-heif resets that to 1), and pillow-heif reports
  // its size turned: so the decoded size (heif.ts).
  const turned = !isHeif(bytes) && (exif.orientation ?? 1) >= 5;
  const stored: [number, number] = turned ? [gray.height, gray.width] : [gray.width, gray.height];
  const info: PhotoInfo = {
    format: formatOf(bytes), width: stored[0], height: stored[1], make: exif.make, model: exif.model,
    focal_mm: exif.focalMm, focal_35mm: exif.focal35,
  };
  return { info, focalPx: exif.focal35 ? focalPxFrom35mm(exif.focal35, stored) : null };
}

/** The format Pillow would name, from the file's first bytes. */
export function formatOf(u: Uint8Array): string | null {
  const text = (a: number, b: number) => String.fromCharCode(...u.subarray(a, b));
  if (u[0] === 0xff && u[1] === 0xd8) return isMpo(u) ? "MPO" : "JPEG";
  if (text(1, 4) === "PNG") return "PNG";
  if (text(0, 4) === "RIFF" && text(8, 12) === "WEBP") return "WEBP";
  if (isHeif(u)) return "HEIF";
  if (text(0, 3) === "GIF") return "GIF";
  if (text(0, 2) === "BM") return "BMP";
  if (text(0, 4) === "II*\0" || text(0, 4) === "MM\0*") return "TIFF";
  return null;
}

/** A JPEG with a Multi-Picture (MPF) APP2 segment, which Pillow opens as "MPO" (many iPhone photos). */
function isMpo(u: Uint8Array): boolean {
  let off = 2;
  while (off + 8 <= u.length && u[off] === 0xff) {
    const m = u[off + 1]!;
    if (m === 0xda || m === 0xd9) return false;
    const len = (u[off + 2]! << 8) | u[off + 3]!;
    if (m === 0xe2 && String.fromCharCode(...u.subarray(off + 4, off + 7)) === "MPF") return true;
    off += 2 + len;
  }
  return false;
}

/** As worker.ts: an ISO base media file of a HEIF brand. */
export function isHeif(u: Uint8Array): boolean {
  if (u.length < 12) return false;
  const text = (a: number, b: number) => String.fromCharCode(...u.subarray(a, b));
  return text(4, 8) === "ftyp" && ["heic", "heix", "heim", "heis", "hevc", "hevx", "hevm", "hevs", "mif1", "msf1", "avif"].includes(text(8, 12));
}
