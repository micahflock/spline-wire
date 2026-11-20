// For the tests only: the snap-fit chain pressed against a profile with sharp
// corners (a staircase, a step), as it would sit. Used to see what the round
// contact model makes of a sharp corner and how well corners.ts recovers it.
//
// The link is the snap-fit chain's: a round sleeve of radius `sleeve` about
// every pin and an arm of half-width `arm` between two pins (docs/snap-chain.md:
// 4.39 mm and 2.5 mm). The profile is a polyline; the object lies on its
// right as it runs. Pins are laid one at a time: each link is turned about the
// last pin, from well clear of the object towards it, until its sleeve or its
// arm touches the profile: the chain wrapped tight.

export type P = [number, number];

export interface Link { pitch: number; sleeve: number; arm: number }

export const SNAP_LINK: Link = { pitch: 13, sleeve: 4.39, arm: 2.5 };

function segDist(p: P, a: P, b: P): number {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
}

/** Distance from a point to the profile. */
export function profileDist(p: P, profile: readonly P[]): number {
  let d = Infinity;
  for (let i = 0; i + 1 < profile.length; i++) d = Math.min(d, segDist(p, profile[i]!, profile[i + 1]!));
  return d;
}

/** Distance between segments ab and cd (they do not cross in any use here). */
function segSegDist(a: P, b: P, c: P, d: P): number {
  return Math.min(segDist(a, c, d), segDist(b, c, d), segDist(c, a, b), segDist(d, a, b));
}

function clear(p: P, q: P, profile: readonly P[], link: Link): boolean {
  if (profileDist(q, profile) < link.sleeve) return false;
  for (let i = 0; i + 1 < profile.length; i++) {
    if (segSegDist(p, q, profile[i]!, profile[i + 1]!) < link.arm) return false;
  }
  return true;
}

/** Pins of the chain wrapped over `profile`, the first `start` mm along its
 * first edge. Stops before a pin would pass the profile's last vertex. */
export function wrap(profile: readonly P[], start: number, link: Link = SNAP_LINK): P[] {
  const [a, b] = [profile[0]!, profile[1]!];
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len;
  // The chain on the left of the profile as it runs.
  const first: P = [a[0] + start * ux - link.sleeve * uy, a[1] + start * uy + link.sleeve * ux];
  const pins: P[] = [first];
  let heading = Math.atan2(uy, ux);
  const last = profile[profile.length - 1]!;
  const end = profile[profile.length - 2]!;
  const ex = last[0] - end[0], ey = last[1] - end[1];
  for (let guard = 0; guard < 200; guard++) {
    const p = pins[pins.length - 1]!;
    const at = (th: number): P => [p[0] + link.pitch * Math.cos(th), p[1] + link.pitch * Math.sin(th)];
    // From 120° to the left of straight on (clear of the object) turn right
    // (towards it) until something touches.
    let lo = heading + (120 * Math.PI) / 180;
    if (!clear(p, at(lo), profile, link)) break;
    let hi = lo;
    const step = (0.25 * Math.PI) / 180;
    while (clear(p, at(hi), profile, link) && hi > heading - Math.PI) {
      lo = hi;
      hi -= step;
    }
    if (hi <= heading - Math.PI) break;
    for (let k = 0; k < 50; k++) {
      const mid = (lo + hi) / 2;
      if (clear(p, at(mid), profile, link)) lo = mid;
      else hi = mid;
    }
    const q = at(lo);
    // Past the last vertex along the last edge: done.
    if ((q[0] - last[0]) * ex + (q[1] - last[1]) * ey > 0) break;
    pins.push(q);
    heading = lo;
  }
  return pins;
}

/** A staircase: `steps` treads of `tread` mm and risers of `rise` mm, going
 * up to the right, the object below. Ends with a lead-in and lead-out edge. */
export function staircase(steps: number, tread: number, rise: number, lead = 40): P[] {
  const out: P[] = [[-lead, 0], [0, 0]];
  let x = 0, y = 0;
  for (let i = 0; i < steps; i++) {
    y += rise;
    out.push([x, y]);
    x += tread;
    out.push([x, y]);
  }
  out[out.length - 1] = [x + lead - tread, y];
  return out;
}
