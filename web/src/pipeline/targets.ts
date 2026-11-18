// splinewire/targets.py in TypeScript: the real-world test targets
// (docs/real-world-tests.md). The gauges' shapes come from targets.json,
// which web/scripts/golden_scoring.py writes from Python's targets_json; the
// plaque poses are computed here as Python computes them (pinned by
// scoring.test.ts).

import { DEG, type Vec2 } from "./numeric";
import targetsJson from "./targets.json";

export type Segment =
  | { arc: { center: [number, number]; radius: number; start_deg: number; sweep_deg: number } }
  | { line: { start: [number, number]; end: [number, number] } };

export interface Gauge {
  name: string;
  label: string;
  closed: boolean;
  segments: Segment[];
  radius_mm?: number;
  concave?: boolean;
}

export interface PlaquePose {
  name: string;
  label: string;
}

export const GAUGES: readonly Gauge[] = (targetsJson as unknown as { gauges: Gauge[] }).gauges;
export const PLAQUES: readonly PlaquePose[] = (targetsJson as unknown as { plaques: PlaquePose[] }).plaques;

export function gaugeByName(name: string): Gauge | undefined {
  return GAUGES.find((g) => g.name === name);
}

export function isPlaque(name: string): boolean {
  return PLAQUES.some((p) => p.name === name);
}

function segmentLength(s: Segment): number {
  if ("arc" in s) return Math.abs(s.arc.sweep_deg * DEG) * s.arc.radius;
  return Math.hypot(s.line.end[0] - s.line.start[0], s.line.end[1] - s.line.start[1]);
}

function segmentAt(s: Segment, d: number): Vec2 {
  if ("arc" in s) {
    const { center, radius, start_deg, sweep_deg } = s.arc;
    const a = start_deg * DEG + Math.sign(sweep_deg) * d / radius;
    return [center[0] + radius * Math.cos(a), center[1] + radius * Math.sin(a)];
  }
  const { start, end } = s.line;
  const t = d / segmentLength(s);
  return [start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])];
}

/** Dense points along the gauge's edge, as Gauge.edge: each segment sampled
 * with numpy's linspace, the segments one after another. */
export function gaugeEdge(g: Gauge, spacingMm = 0.02): Vec2[] {
  const out: Vec2[] = [];
  for (const s of g.segments) {
    const len = segmentLength(s);
    const n = Math.max(2, Math.ceil(len / spacingMm) + 1);
    const step = len / (n - 1);
    for (let k = 0; k < n; k++) out.push(segmentAt(s, k === n - 1 ? len : k * step));
  }
  return out;
}

/** The joint turns (degrees) of a plaque pose, as targets.plaque_turns. */
export function plaqueTurns(pose: string, nPins: number): number[] {
  const n = nPins - 2;
  if (n < 1) throw new Error(`a plaque needs at least 3 pins, got ${nPins}`);
  if (pose === "plaque-s") {
    return Array.from({ length: n }, (_, i) => 45.0 * Math.sin(2 * Math.PI * (i + 0.5) / n));
  }
  if (pose === "plaque-tight") {
    const bends = [100.0, -95.0, 98.0, -100.0];
    const turns = new Array<number>(n).fill(0);
    for (let i = 1, k = 0; i < n; i += 3, k++) turns[i] = bends[k % bends.length]!;
    return turns;
  }
  throw new Error(`no plaque pose ${pose}`);
}

/** Pin positions of a plaque, as targets.plaque_pins (chain.pins_from_turns). */
export function plaquePins(pose: string, pitchMm: number, nPins: number): Vec2[] {
  const turns = plaqueTurns(pose, nPins);
  const pins: Vec2[] = [[0, 0]];
  let heading = 0, x = 0, y = 0;
  for (let i = 0; i < nPins - 1; i++) {
    if (i > 0) heading += turns[i - 1]! * DEG;
    x += pitchMm * Math.cos(heading);
    y += pitchMm * Math.sin(heading);
    pins.push([x, y]);
  }
  return pins;
}
