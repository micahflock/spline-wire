// The few EXIF tags the page needs to remember a focal length per lens (D17),
// read in the page from the photo's bytes, before and apart from the core
// (which reads the focal length again for itself).
//
//   Make, Model            IFD0
//   FocalLength            Exif IFD, mm of the real lens (Android's takePhoto keeps it)
//   FocalLengthIn35mmFilm  Exif IFD, else IFD0, as splinewire/camera.py looks it up;
//                          0 counts as missing (the Pixel's takePhoto writes 0)
//   LensModel              Exif IFD ("iPhone 16 Pro back triple camera 6.765mm f/1.78")
//   Orientation            IFD0: how the stored pixels turn upright (1-8), for the
//                          stored size the photo info reports (pipeline/engine.ts)
//
// JPEG keeps EXIF in an APP1 segment; HEIF (iPhone library photos) in an item
// that starts "Exif\0\0" somewhere in the file, which is searched for.

export interface Exif {
  make: string | null;
  model: string | null;
  lens: string | null;
  /** FocalLength, mm. */
  focalMm: number | null;
  /** FocalLengthIn35mmFilm; null when missing or 0. */
  focal35: number | null;
  /** Orientation, 1-8; null when missing. */
  orientation?: number | null;
}

export const NO_EXIF: Exif = { make: null, model: null, lens: null, focalMm: null, focal35: null, orientation: null };

const TAG = { make: 0x010f, model: 0x0110, orientation: 0x0112, exifIfd: 0x8769, focal: 0x920a, focal35: 0xa405, lens: 0xa434 };

/** The tags, or NO_EXIF if the bytes carry none (a frame, a screenshot, the
 * file input's camera on iPhone and Pixel). Never throws. */
export function readExif(bytes: ArrayBuffer | Uint8Array): Exif {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  try {
    const start = tiffStart(u8);
    return start === null ? NO_EXIF : parseTiff(new DataView(u8.buffer, u8.byteOffset + start, u8.length - start));
  } catch {
    return NO_EXIF;
  }
}

/** A photo file's tags. JPEG's EXIF is at the front, so only the first 256 KB
 * is read unless that does not find it. */
export async function exifOfFile(file: Blob): Promise<Exif> {
  try {
    const head = new Uint8Array(await file.slice(0, 256 * 1024).arrayBuffer());
    const e = readExif(head);
    if (e !== NO_EXIF || file.size <= head.length) return e;
    return readExif(await file.arrayBuffer());
  } catch {
    return NO_EXIF;
  }
}

function tiffStart(u: Uint8Array): number | null {
  if (u[0] === 0xff && u[1] === 0xd8) {
    let off = 2;
    while (off + 4 <= u.length && u[off] === 0xff) {
      const m = u[off + 1]!;
      if (m === 0xda || m === 0xd9) break;                     // image data: no EXIF before it
      const len = (u[off + 2]! << 8) | u[off + 3]!;
      if (m === 0xe1 && isExifHeader(u, off + 4)) return off + 10;
      off += 2 + len;
    }
    return null;
  }
  // HEIF, or anything else: look for "Exif\0\0" followed by a TIFF header.
  for (let i = 0; i + 10 <= u.length; i++) {
    if (u[i] === 0x45 && isExifHeader(u, i) && isTiff(u, i + 6)) return i + 6;
  }
  return null;
}

function isExifHeader(u: Uint8Array, i: number): boolean {
  return u[i] === 0x45 && u[i + 1] === 0x78 && u[i + 2] === 0x69 && u[i + 3] === 0x66 && u[i + 4] === 0 && u[i + 5] === 0;
}

function isTiff(u: Uint8Array, i: number): boolean {
  return (u[i] === 0x49 && u[i + 1] === 0x49 && u[i + 2] === 42 && u[i + 3] === 0)
    || (u[i] === 0x4d && u[i + 1] === 0x4d && u[i + 2] === 0 && u[i + 3] === 42);
}

type Value = number | string | null;

function parseTiff(v: DataView): Exif {
  const le = v.getUint8(0) === 0x49;
  const u16 = (o: number) => v.getUint16(o, le), u32 = (o: number) => v.getUint32(o, le);
  if (u16(2) !== 42) return NO_EXIF;
  const ifd = (o: number): Map<number, Value> => {
    const tags = new Map<number, Value>();
    const n = u16(o);
    for (let i = 0; i < n; i++) {
      const e = o + 2 + i * 12;
      tags.set(u16(e), value(v, le, u16(e + 2), u32(e + 4), e + 8));
    }
    return tags;
  };
  const ifd0 = ifd(u32(4));
  const ptr = ifd0.get(TAG.exifIfd);
  const sub = typeof ptr === "number" ? ifd(ptr) : new Map<number, Value>();
  const text = (x: Value | undefined) => (typeof x === "string" && x.trim() ? x.trim() : null);
  const num = (x: Value | undefined) => (typeof x === "number" && Number.isFinite(x) && x > 0 ? x : null);
  return {
    make: text(ifd0.get(TAG.make)),
    model: text(ifd0.get(TAG.model)),
    lens: text(sub.get(TAG.lens)),
    focalMm: num(sub.get(TAG.focal) ?? ifd0.get(TAG.focal)),
    focal35: num(sub.get(TAG.focal35) ?? ifd0.get(TAG.focal35)),
    orientation: num(ifd0.get(TAG.orientation)),
  };
}

/** One entry's value: ASCII, SHORT, LONG or RATIONAL; anything else null. */
function value(v: DataView, le: boolean, type: number, count: number, at: number): Value {
  const size = ({ 2: 1, 3: 2, 4: 4, 5: 8 } as Record<number, number>)[type];
  if (!size || count < 1) return null;
  const off = size * count <= 4 ? at : v.getUint32(at, le);
  if (off + size * count > v.byteLength) return null;
  if (type === 2) {
    let s = "";
    for (let i = 0; i < count; i++) {
      const c = v.getUint8(off + i);
      if (c === 0) break;
      s += String.fromCharCode(c);
    }
    return s;
  }
  if (type === 3) return v.getUint16(off, le);
  if (type === 4) return v.getUint32(off, le);
  const den = v.getUint32(off + 4, le);
  return den ? v.getUint32(off, le) / den : null;
}
