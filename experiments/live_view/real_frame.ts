// The live detector on a real iPhone frame: the owner's snap-fit chain (13
// pins) on a wooden table, seen at about 20 degrees (data/wood-table-iphone.png:
// the video area of a camera-view screenshot, build 22, 2026-11-21, 1184×1579
// for the 3024×4032 video; the coach's message over its top quarter is left
// out). The live view found the chain, lost it as the phone moved, and its scan
// never found it again: the grain gave each mask well over BLOBS.maxCandidates
// components, and the first that many in row order, all above the chain, were
// the only ones tested.
//
// The whole frame is drawn at a range of work sizes (by area; larger than the
// screenshot, bilinear, so those rows are only indicative), from the 800 px the
// camera view read until 2026-11-19 to the scan's quarters' scale (2016),
// and each is searched and analysed as the camera view does.
//
//   cd web && npx esbuild ../experiments/live_view/real_frame.ts --bundle --platform=node --format=esm --outfile=../out/live_view/real_frame.mjs
//   node ../out/live_view/real_frame.mjs ../experiments/live_view/data/wood-table-iphone.png
import { readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";
import { BLOBS, detectBlobs, shapeOf } from "../../web/src/overlay/blobs.ts";
import { REGIONS } from "../../web/src/overlay/region.ts";
import { analyse } from "../../web/src/pipeline/live.ts";
import { specFromDict } from "../../web/src/pipeline/chain.ts";
import type { GrayFrame } from "../../web/src/overlay/types.ts";

const SNAP = { pitch_mm: 13.0, half_width_mm: 4.39, contact: "round", fiducial: "ring", fiducial_mm: 8.79, ring_inner_mm: 7.07, n_pins: 13, max_bend_deg: 110 };
const VW = 3024, VH = 4032;
const FOCAL_PX = (24 * Math.hypot(VW, VH)) / Math.hypot(36, 24);   // "24.0 mm (this camera's last photo)"
const TEXT = 0.25;                                                   // the coach's message: the top quarter
const med = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1]!;

/** An 8-bit gray, non-interlaced PNG. */
function readPng(path: string): GrayFrame {
  const buf = readFileSync(path);
  let pos = 8, width = 0, height = 0;
  const idat: Buffer[] = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos), type = buf.toString("ascii", pos + 4, pos + 8), body = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = body.readUInt32BE(0);
      height = body.readUInt32BE(4);
      if (body[8] !== 8 || body[9] !== 0 || body[12] !== 0) throw new Error("not an 8-bit gray, non-interlaced PNG");
    } else if (type === "IDAT") idat.push(body);
    pos += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat)), data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (width + 1)]!, row = raw.subarray(y * (width + 1) + 1, (y + 1) * (width + 1));
    for (let x = 0; x < width; x++) {
      const a = x ? data[y * width + x - 1]! : 0, b = y ? data[(y - 1) * width + x]! : 0, c = x && y ? data[(y - 1) * width + x - 1]! : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const pred = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][f]!;
      data[y * width + x] = (row[x]! + pred) & 255;
    }
  }
  return { width, height, data };
}

/** `src` drawn at dw × dh: by area when smaller, bilinear when larger. */
function resize(src: GrayFrame, dw: number, dh: number): GrayFrame {
  const out = new Uint8Array(dw * dh), sx = src.width / dw, sy = src.height / dh;
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      if (sx >= 1) {
        const x0 = x * sx, x1 = x0 + sx, y0 = y * sy, y1 = y0 + sy;
        let s = 0, wsum = 0;
        for (let j = Math.floor(y0); j < Math.ceil(y1); j++) {
          const wy = Math.min(y1, j + 1) - Math.max(y0, j);
          for (let i = Math.floor(x0); i < Math.ceil(x1); i++) {
            const w = wy * (Math.min(x1, i + 1) - Math.max(x0, i));
            s += w * src.data[Math.min(src.height - 1, j) * src.width + Math.min(src.width - 1, i)]!;
            wsum += w;
          }
        }
        out[y * dw + x] = Math.round(s / wsum);
      } else {
        const fx = Math.max(0, (x + 0.5) * sx - 0.5), fy = Math.max(0, (y + 0.5) * sy - 0.5);
        const i = Math.min(src.width - 2, Math.floor(fx)), j = Math.min(src.height - 2, Math.floor(fy)), u = fx - i, v = fy - j;
        const at = (a: number, b: number) => src.data[b * src.width + a]!;
        out[y * dw + x] = Math.round((at(i, j) * (1 - u) + at(i + 1, j) * u) * (1 - v) + (at(i, j + 1) * (1 - u) + at(i + 1, j + 1) * u) * v);
      }
    }
  }
  return { width: dw, height: dh, data: out };
}

const shot = readPng(process.argv[2]!);
const shape = shapeOf(SNAP), spec = specFromDict(SNAP);
console.log(`BLOBS.maxCandidates ${BLOBS.maxCandidates}; the scan reads the whole frame at ${REGIONS.fullLongSide.ring} and its quarters at ${REGIONS.tileLongSide.ring}`);
console.log("long side  rim (work px)  blobs  chain  gaps  tilt   ms");
for (const L of [640, 800, 1008, 1200, 1400, 1600, 1800, 2016]) {
  const dh = L, dw = Math.round((L * shot.width) / shot.height), s = VW / dw;
  const work = resize(shot, dw, dh);
  const t: number[] = [];
  let blobs = detectBlobs(work, s, shape);
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    blobs = detectBlobs(work, s, shape);
    t.push(performance.now() - t0);
  }
  blobs = blobs.filter((b) => b.y > TEXT * VH);
  const a = analyse(blobs.map((b) => [b.x, b.y]), blobs.map((b) => b.d), [VW, VH], spec, FOCAL_PX);
  const rim = a.order.length ? med(a.order.map((i) => blobs[i]!.d)) / s : NaN;
  console.log(`${String(L).padStart(9)}  ${rim.toFixed(1).padStart(13)}  ${String(blobs.length).padStart(5)}  ${String(a.order.length).padStart(5)}  ${String(a.gaps.length).padStart(4)}  ${String(a.tilt_deg ?? "-").padStart(4)}  ${med(t).toFixed(1).padStart(4)}`);
}
