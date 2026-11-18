// Measurement.to_photo_px: where points on the chain plane (mm) fall in the
// photo, for drawing the curve over it. The plane maps to the photo by a
// homography fitted to the pins (cv2.findHomography with all points: a
// normalised linear fit refined on the reprojection error); a nearly straight
// chain leaves that undetermined across the chain, and then each point is
// placed by its nearest link's scale and direction instead.

import { leastSquaresLM } from "./lm";
import { clip, norm2, type Vec2 } from "./numeric";

export function toPhotoPx(points: readonly Vec2[], pinsMm: readonly Vec2[], pinsPx: readonly Vec2[]): Vec2[] {
  const n = pinsMm.length;
  const [s0, s1] = singularValues(pinsMm);
  if (n >= 5 && s1 > 0.1 * s0) {
    const H = fitHomography(pinsMm, pinsPx);
    if (H && Math.max(...pinsMm.map((p, i) => maxAbs(apply(H, p), pinsPx[i]!))) < 3.0) return points.map((p) => apply(H, p));
  }
  return points.map((p) => {
    let best = 0, bestD = Infinity;
    for (let k = 0; k + 1 < n; k++) {
      const a = pinsMm[k]!, d = [pinsMm[k + 1]![0] - a[0], pinsMm[k + 1]![1] - a[1]];
      const len2 = Math.max(d[0]! ** 2 + d[1]! ** 2, 1e-12);
      const t = clip(((p[0] - a[0]) * d[0]! + (p[1] - a[1]) * d[1]!) / len2, 0, 1);
      const dist = norm2(a[0] + t * d[0]! - p[0], a[1] + t * d[1]! - p[1]);
      if (dist < bestD) {
        bestD = dist;
        best = k;
      }
    }
    const a = pinsMm[best]!, b = pinsMm[best + 1]!, ap = pinsPx[best]!, bp = pinsPx[best + 1]!;
    const dm = [b[0] - a[0], b[1] - a[1]], dp = [bp[0] - ap[0], bp[1] - ap[1]];
    const len2 = Math.max(dm[0]! ** 2 + dm[1]! ** 2, 1e-12);
    const r = [p[0] - a[0], p[1] - a[1]];
    const along = (r[0]! * dm[0]! + r[1]! * dm[1]!) / len2;
    const left = (dm[0]! * r[1]! - dm[1]! * r[0]!) / len2;
    // mm runs y-up and the photo y-down: mm's left normal is (v, -u) of a photo direction (u, v).
    return [ap[0] + along * dp[0]! + left * dp[1]!, ap[1] + along * dp[1]! - left * dp[0]!];
  });
}

function maxAbs(a: Vec2, b: Vec2): number {
  return Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]));
}

function apply(H: number[], p: Vec2): Vec2 {
  const w = H[6]! * p[0] + H[7]! * p[1] + H[8]!;
  return [(H[0]! * p[0] + H[1]! * p[1] + H[2]!) / w, (H[3]! * p[0] + H[4]! * p[1] + H[5]!) / w];
}

/** Singular values of the centred points (an N x 2 matrix), largest first. */
function singularValues(p: readonly Vec2[]): [number, number] {
  const mx = p.reduce((s, q) => s + q[0], 0) / p.length, my = p.reduce((s, q) => s + q[1], 0) / p.length;
  let a = 0, b = 0, c = 0;
  for (const q of p) {
    const x = q[0] - mx, y = q[1] - my;
    a += x * x;
    b += x * y;
    c += y * y;
  }
  const tr = a + c, det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  return [Math.sqrt(Math.max(0, tr / 2 + disc)), Math.sqrt(Math.max(0, tr / 2 - disc))];
}

/** The homography from `src` to `dst`: a normalised linear fit (DLT), then
 * Levenberg–Marquardt on the reprojection error in `dst`, as
 * cv2.findHomography(src, dst, 0) does. Null if degenerate. */
export function fitHomography(src: readonly Vec2[], dst: readonly Vec2[]): number[] | null {
  const norm = (pts: readonly Vec2[]) => {
    const n = pts.length;
    const cx = pts.reduce((s, p) => s + p[0], 0) / n, cy = pts.reduce((s, p) => s + p[1], 0) / n;
    const sx = n / Math.max(pts.reduce((s, p) => s + Math.abs(p[0] - cx), 0), 1e-12);
    const sy = n / Math.max(pts.reduce((s, p) => s + Math.abs(p[1] - cy), 0), 1e-12);
    return { cx, cy, sx, sy };
  };
  const ns = norm(src), nd = norm(dst);
  // The normal equations of the DLT, 9x9; the solution is the eigenvector of the smallest eigenvalue.
  const L = Array.from({ length: 9 }, () => new Array<number>(9).fill(0));
  src.forEach((p, i) => {
    const X = (p[0] - ns.cx) * ns.sx, Y = (p[1] - ns.cy) * ns.sy;
    const x = (dst[i]![0] - nd.cx) * nd.sx, y = (dst[i]![1] - nd.cy) * nd.sy;
    for (const r of [[X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x], [0, 0, 0, X, Y, 1, -y * X, -y * Y, -y]]) {
      for (let a = 0; a < 9; a++) for (let b = 0; b < 9; b++) L[a]![b] = L[a]![b]! + r[a]! * r[b]!;
    }
  });
  const h = smallestEigenvector(L);
  // Undo the normalisations: H = Td^-1 * Hn * Ts.
  const Hn = [[h[0]!, h[1]!, h[2]!], [h[3]!, h[4]!, h[5]!], [h[6]!, h[7]!, h[8]!]];
  const Ts = [[ns.sx, 0, -ns.sx * ns.cx], [0, ns.sy, -ns.sy * ns.cy], [0, 0, 1]];
  const TdInv = [[1 / nd.sx, 0, nd.cx], [0, 1 / nd.sy, nd.cy], [0, 0, 1]];
  const H = mul3(mul3(TdInv, Hn), Ts).flat();
  if (Math.abs(H[8]!) < 1e-12) return null;
  const H0 = H.map((v) => v / H[8]!);
  // Refine the 8 free entries on the reprojection error.
  const residuals = (x: readonly number[]) => {
    const Hx = [...x, 1];
    return src.flatMap((p, i) => {
      const q = apply(Hx, p);
      return [q[0] - dst[i]![0], q[1] - dst[i]![1]];
    });
  };
  const fit = leastSquaresLM(residuals, H0.slice(0, 8), { maxNfev: 200 });
  return [...fit.x, 1];
}

function mul3(a: number[][], b: number[][]): number[][] {
  return a.map((row) => [0, 1, 2].map((j) => row[0]! * b[0]![j]! + row[1]! * b[1]![j]! + row[2]! * b[2]![j]!));
}

/** The eigenvector of a symmetric matrix's smallest eigenvalue (cyclic Jacobi). */
function smallestEigenvector(A: number[][]): number[] {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const v: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p]![q]! ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p]![q]!) < 1e-300) continue;
        const theta = (a[q]![q]! - a[p]![p]!) / (2 * a[p]![q]!);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k]![p]!, akq = a[k]![q]!;
          a[k]![p] = c * akp - s * akq;
          a[k]![q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p]![k]!, aqk = a[q]![k]!;
          a[p]![k] = c * apk - s * aqk;
          a[q]![k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k]![p]!, vkq = v[k]![q]!;
          v[k]![p] = c * vkp - s * vkq;
          v[k]![q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let m = 0;
  for (let i = 1; i < n; i++) if (a[i]![i]! < a[m]![m]!) m = i;
  return v.map((row) => row[m]!);
}
