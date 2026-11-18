import { describe, expect, it } from "vitest";
import { NO_EXIF, exifOfFile, readExif } from "./exif";

type Entry = [tag: number, type: 2 | 3 | 5, value: string | number];

/** A TIFF block (little- or big-endian) with IFD0 and an Exif IFD, as cameras write them. */
export function tiff(ifd0: Entry[], exif: Entry[], le = true): Uint8Array {
  const out: number[] = [];
  const u16 = (v: number) => (le ? [v & 255, v >> 8] : [v >> 8, v & 255]);
  const u32 = (v: number) => (le ? [v & 255, (v >> 8) & 255, (v >> 16) & 255, v >>> 24] : [v >>> 24, (v >> 16) & 255, (v >> 8) & 255, v & 255]);
  const ifdSize = (n: number) => 2 + 12 * n + 4;
  const all0 = exif.length ? [...ifd0, [0x8769, 4 as never, 0] as Entry] : ifd0;
  const at0 = 8, atExif = at0 + ifdSize(all0.length);
  let data = atExif + (exif.length ? ifdSize(exif.length) : 0);
  const extra: number[] = [];
  const write = (entries: Entry[], at: number) => {
    const bytes: number[] = [...u16(entries.length)];
    for (const [tag, type, value] of entries) {
      if ((type as number) === 4) { bytes.push(...u16(tag), ...u16(4), ...u32(1), ...u32(atExif)); continue; }
      let payload: number[];
      let count: number;
      if (type === 2) { payload = [...String(value)].map((c) => c.charCodeAt(0)).concat(0); count = payload.length; }
      else if (type === 3) { payload = [...u16(Number(value)), 0, 0]; count = 1; }
      else { payload = [...u32(Math.round(Number(value) * 100)), ...u32(100)]; count = 1; }
      bytes.push(...u16(tag), ...u16(type), ...u32(count));
      if (type === 3 || payload.length <= 4) bytes.push(...payload.slice(0, 4), ...Array(Math.max(0, 4 - payload.length)).fill(0));
      else { bytes.push(...u32(data)); extra.push(...payload); data += payload.length; }
    }
    bytes.push(...u32(0));
    out.splice(at, bytes.length, ...bytes);
  };
  out.push(...(le ? [0x49, 0x49] : [0x4d, 0x4d]), ...u16(42), ...u32(at0));
  write(all0, at0);
  if (exif.length) write(exif, atExif);
  return Uint8Array.from([...out, ...extra]);
}

/** A JPEG that is only its markers: SOI, a JFIF APP0, the EXIF APP1, EOI. */
export function jpegWith(t: Uint8Array): Uint8Array {
  const app1 = [..."Exif"].map((c) => c.charCodeAt(0)).concat(0, 0, ...t);
  const len = app1.length + 2;
  return Uint8Array.from([0xff, 0xd8, 0xff, 0xe0, 0, 4, 0, 0, 0xff, 0xe1, len >> 8, len & 255, ...app1, 0xff, 0xd9]);
}

export const PIXEL_TAKEPHOTO = tiff([[0x010f, 2, "Google"], [0x0110, 2, "Pixel 7 Pro"]], [[0x920a, 5, 6.81], [0xa405, 3, 0]]);
export const PIXEL_LIBRARY = tiff([[0x010f, 2, "Google"], [0x0110, 2, "Pixel 7 Pro"]],
  [[0x920a, 5, 6.81], [0xa405, 3, 24], [0xa434, 2, "Pixel 7 Pro back camera 6.81mm f/1.85"]]);

describe("EXIF in the page", () => {
  it("reads make, model, lens and both focal lengths, either byte order", () => {
    for (const le of [true, false]) {
      const t = tiff([[0x010f, 2, "Apple"], [0x0110, 2, "iPhone 16 Pro"], [0x0112, 3, 6]],
        [[0x920a, 5, 6.765], [0xa405, 3, 24], [0xa434, 2, "iPhone 16 Pro back triple camera 6.765mm f/1.78"]], le);
      expect(readExif(jpegWith(t))).toEqual({
        make: "Apple", model: "iPhone 16 Pro", lens: "iPhone 16 Pro back triple camera 6.765mm f/1.78", focalMm: 6.77, focal35: 24,
        orientation: 6,
      });
    }
  });

  it("takes a 35 mm focal length of 0 as missing (the Pixel's takePhoto)", () => {
    expect(readExif(jpegWith(PIXEL_TAKEPHOTO))).toEqual({ make: "Google", model: "Pixel 7 Pro", lens: null, focalMm: 6.81, focal35: null, orientation: null });
  });

  it("finds EXIF inside a HEIF file, and says nothing for a file without", async () => {
    const heif = Uint8Array.from([0, 0, 0, 24, ...[..."ftypheic"].map((c) => c.charCodeAt(0)), ...Array(40).fill(7),
      0, 0, 0, 6, ...[..."Exif"].map((c) => c.charCodeAt(0)), 0, 0, ...PIXEL_LIBRARY]);
    expect(readExif(heif).lens).toBe("Pixel 7 Pro back camera 6.81mm f/1.85");
    expect(readExif(Uint8Array.from([0xff, 0xd8, 0xff, 0xda, 0, 2, 1, 2, 3]))).toBe(NO_EXIF);
    expect(readExif(new Uint8Array(0))).toBe(NO_EXIF);
    expect(readExif(Uint8Array.from([0xff, 0xd8, 0xff, 0xe1, 0, 40, 0x45, 0x78, 0x69, 0x66, 0, 0, 0x49, 0x49]))).toBe(NO_EXIF);   // cut short
    expect(await exifOfFile(new Blob([jpegWith(PIXEL_LIBRARY) as Uint8Array<ArrayBuffer>]))).toMatchObject({ focal35: 24, focalMm: 6.81 });
  });
});
