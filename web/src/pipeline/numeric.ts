// Small numeric helpers that reproduce NumPy's and Python's results to the
// last bit where the pipeline's decisions depend on them (ties between walks
// in order.ts, medians of steps). Each says which Python call it stands for.

export type Vec2 = readonly [number, number];

/** np.linalg.norm of a 2-vector: sqrt(x*x + y*y), not Math.hypot, which can
 * differ in the last bit. */
export function norm2(x: number, y: number): number {
  return Math.sqrt(x * x + y * y);
}

/** np.median: the middle value, or the mean of the two middle values. */
export function median(values: ArrayLike<number>): number {
  const v = Array.from(values).sort((a, b) => a - b);
  return medianSorted(v);
}

/** The median of values already sorted in increasing order. */
export function medianSorted(v: ArrayLike<number>): number {
  const n = v.length;
  if (n === 0) return NaN;
  const h = n >> 1;
  return n % 2 ? v[h]! : (v[h - 1]! + v[h]!) / 2;
}

/** np.round: halves go to the even neighbour. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Python's math.fsum: the correctly rounded sum (Shewchuk's algorithm), so
 * that the same values summed in another order give the same result. */
export function fsum(values: Iterable<number>): number {
  const partials: number[] = [];
  for (let x of values) {
    let i = 0;
    for (let y of partials) {
      if (Math.abs(x) < Math.abs(y)) [x, y] = [y, x];
      const hi = x + y;
      const lo = y - (hi - x);
      if (lo !== 0) partials[i++] = lo;
      x = hi;
    }
    partials.length = i;
    partials.push(x);
  }
  // Add the partials from the largest, with Python's correction for the
  // half-way case.
  let n = partials.length;
  if (n === 0) return 0;
  let hi = partials[--n]!;
  let lo = 0;
  while (n > 0) {
    const x = hi;
    const y = partials[--n]!;
    hi = x + y;
    const yr = hi - x;
    lo = y - yr;
    if (lo !== 0) break;
  }
  if (n > 0 && ((lo < 0 && partials[n - 1]! < 0) || (lo > 0 && partials[n - 1]! > 0))) {
    const y = lo * 2;
    const x = hi + y;
    const yr = x - hi;
    if (y === yr) hi = x;
  }
  return hi;
}

/** Python's f"{x:.{digits}f}": like toFixed, but an exact tie rounds to even
 * (toFixed rounds it up: 0.125 gives "0.13" against Python's "0.12"). */
export function fixed(x: number, digits: number): string {
  const s = x.toFixed(digits);
  // toFixed(100) is the exact decimal value for the magnitudes measured here
  // (a double of at least 1e-3 has at most ~62 fractional digits).
  const exact = x.toFixed(100);
  const dot = exact.indexOf(".");
  const tail = exact.slice(dot + 1 + digits);
  if (!/^50*$/.test(tail)) return s;
  // An exact tie: keep the even last digit, which truncation gives when the
  // digit before the 5 is even.
  const kept = exact.slice(0, dot + 1 + digits).replace(/\.$/, "");
  const last = Number(kept[kept.length - 1]);
  return last % 2 === 0 ? kept : s;
}

export function argmin(v: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i]! < v[best]!) best = i;
  return best;
}

export function clip(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

export const DEG = Math.PI / 180;
