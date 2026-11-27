// The chain as the pages draw it (the phone's strip under its header, the
// desktop page's mark): drawn from the chain preset, so that it looks like the
// chain the person has chosen, in the colours they print it in.
//
// Two kinds of link, by what the preset says of them:
// - "snap" (contact: round, ring fiducial: the snap-fit chain, docs/snap-chain.md):
//   seen from above, a round head on each pin with the rim round its top edge,
//   a 5 mm arm stepping down to the split sleeve one pitch on, which clips under
//   the next link's head. Every hook has a pin over it, the last one too, so the
//   drawing has one head per pin of the chain and one link fewer. The step is
//   where data/snap-link.stl puts it, seen from above. The rim is as wide as the
//   head, so on the page it would melt into the paper round it: the head is
//   drawn with a thin edge in the link's colour outside the rim, so the rim
//   shows as a ring in the mark's colour, dark on both sides.
// - "plates" (any other preset, such as the O-ring chain, docs/printed-chain.md):
//   stadium plates the link's full width with round ends on the pins, the outer
//   links (every other one, on top) and the inner ones between them (recessed,
//   drawn a shade lighter and a little narrower), a dot or ring on every pin.
//
// Pure: SVG strings, so it is tested without a page. Colours are CSS variables
// the page defines (--ca-link, --ca-low, --ca-mark, --ca-pale, --ca-pale-2).

import type { ChainSpec, Point } from "../core/types";

export type ChainLook = Pick<ChainSpec, "pitch_mm" | "half_width_mm" | "fiducial" | "fiducial_mm" | "ring_inner_mm" | "contact">
  & { fiducial_shade?: "light" | "dark" };
export type LinkStyle = "snap" | "plates";

/** The snap-fit link from above (data/snap-link.stl, 3DP_Link_Ver_MF), mm. */
export const SNAP_ART = {
  armHalf: 2.5,
  /** Where the arm steps down from the head's level, along the link from the head, at each edge of the arm. */
  step: [8.39, 4.27] as const,
  /** The edge drawn outside the rim, so the rim contrasts with the page as with the head (not on the part). */
  edge: 0.8,
};

export function linkStyle(c: ChainLook): LinkStyle {
  return c.contact === "round" && c.fiducial === "ring" ? "snap" : "plates";
}

/** The snap-fit chain's head as drawn: the rim's outer radius and the edge outside it, mm. */
function headMm(c: ChainLook): number {
  return Math.max(c.fiducial_mm / 2, c.half_width_mm) + SNAP_ART.edge;
}

/** How far the drawing reaches from a pin's centre, mm. */
export function reachMm(c: ChainLook): number {
  return linkStyle(c) === "snap" ? headMm(c) : Math.max(c.half_width_mm, c.fiducial_mm / 2);
}

const f = (n: number) => (Math.round(n * 100) / 100).toString();
const pt = (p: Point) => `${f(p[0])} ${f(p[1])}`;

interface Tones { link: string; low: string; mark: string }

/** The tones: dark links with light marks as the presets print them, or the other way round (fiducial_shade dark). */
function tones(c: ChainLook): Tones {
  return c.fiducial_shade === "dark"
    ? { link: "var(--ca-pale)", low: "var(--ca-pale-2)", mark: "var(--ca-link)" }
    : { link: "var(--ca-link)", low: "var(--ca-low)", mark: "var(--ca-mark)" };
}

/**
 * The chain's elements, no <svg> round them. `at`: the pins, one per pin of the
 * chain, in the drawing's units, y down; `s`: units per mm. `grew`: the pin just
 * added, whose link is drawn popping in (class "new").
 */
export function chainArt(c: ChainLook, at: readonly Point[], s: number, opts: { grew?: number } = {}): string {
  const t = tones(c);
  const n = at.length;
  if (n < 1) return "";
  const isNew = (k: number) => (k === opts.grew ? ` class="new"` : "");
  return linkStyle(c) === "snap" ? snapArt(c, at, n, s, t, isNew) : platesArt(c, at, n, s, t, isNew);
}

function snapArt(c: ChainLook, at: readonly Point[], n: number, s: number, t: Tones, isNew: (k: number) => string): string {
  const A = SNAP_ART;
  const out: string[] = [];
  // Link j: its head on pin j, its arm to its sleeve on pin j + 1, under that pin's head.
  const frame = (j: number) => {
    const h = at[j]!, e = at[j + 1]!;
    const L = Math.hypot(e[0] - h[0], e[1] - h[1]) || 1;
    const u: Point = [(e[0] - h[0]) / L, (e[1] - h[1]) / L];
    const v: Point = [u[1], -u[0]];        // across the link, to the left seen from above (y is down)
    const P = (along: number, across: number): Point => [h[0] + (u[0] * along + v[0] * across) * s, h[1] + (u[1] * along + v[1] * across) * s];
    return { h, e, u, v, L, P };
  };
  // Lowest first: the arms where they are down at the sleeve's level, then each arm's raised part, the heads, the rims.
  for (let j = 0; j < n - 1; j++) {
    const { h, e } = frame(j);
    out.push(`<g${isNew(j + 1)}><line x1="${f(h[0])}" y1="${f(h[1])}" x2="${f(e[0])}" y2="${f(e[1])}" style="stroke:${t.low}" stroke-width="${f(2 * A.armHalf * s)}"/></g>`);
  }
  for (let j = 0; j < n - 1; j++) {
    const { P } = frame(j);
    const d = [P(0, -A.armHalf), P(A.step[0], -A.armHalf), P(A.step[1], A.armHalf), P(0, A.armHalf)].map(pt).join("L");
    out.push(`<path${isNew(j + 1)} d="M${d}Z" style="fill:${t.link}"/>`);
  }
  const head = c.fiducial_mm / 2, inner = (c.ring_inner_mm ?? 0) / 2;
  for (let k = 0; k < n; k++) {
    const [x, y] = at[k]!;
    out.push(`<g${isNew(k)}><circle cx="${f(x)}" cy="${f(y)}" r="${f(headMm(c) * s)}" style="fill:${t.link}"/>`
      + `<circle cx="${f(x)}" cy="${f(y)}" r="${f(((head + inner) / 2) * s)}" style="fill:none;stroke:${t.mark}" stroke-width="${f((head - inner) * s)}"/></g>`);
  }
  return out.join("");
}

function platesArt(c: ChainLook, at: readonly Point[], n: number, s: number, t: Tones, isNew: (k: number) => string): string {
  const out: string[] = [];
  const w = 2 * c.half_width_mm * s;
  const link = (j: number, inner: boolean) => {
    const [a, b] = [at[j]!, at[j + 1]!];
    return `<line${isNew(j + 1)} x1="${f(a[0])}" y1="${f(a[1])}" x2="${f(b[0])}" y2="${f(b[1])}" style="stroke:${inner ? t.low : t.link}" stroke-width="${f(inner ? w * 0.86 : w)}" stroke-linecap="round"/>`;
  };
  for (let j = 1; j < n - 1; j += 2) out.push(link(j, true));       // the inner links, under the outer ones
  for (let j = 0; j < n - 1; j += 2) out.push(link(j, false));
  const r = (c.fiducial_mm / 2) * s, ri = ((c.ring_inner_mm ?? 0) / 2) * s;
  for (let k = 0; k < n; k++) {
    const [x, y] = at[k]!;
    out.push(ri > 0
      ? `<circle${isNew(k)} cx="${f(x)}" cy="${f(y)}" r="${f((r + ri) / 2)}" style="fill:none;stroke:${t.mark}" stroke-width="${f(r - ri)}"/>`
      : `<circle${isNew(k)} cx="${f(x)}" cy="${f(y)}" r="${f(r)}" style="fill:${t.mark}"/>`);
  }
  return out.join("");
}

/** A short straight piece of the chain, for a logo: `pins` pins, in mm. */
export function chainMarkSvg(c: ChainLook, pins = 3, label = ""): string {
  const R = reachMm(c), p = c.pitch_mm;
  const at: Point[] = Array.from({ length: pins }, (_, k): Point => [k * p, 0]);
  const x1 = (at.length - 1) * p + R;
  const aria = label ? `role="img" aria-label="${label}"` : `aria-hidden="true"`;
  return `<svg class="chain-mark" viewBox="${f(-R - 0.5)} ${f(-R - 0.5)} ${f(x1 + R + 1)} ${f(2 * R + 1)}" ${aria}>${chainArt(c, at, 1)}</svg>`;
}
