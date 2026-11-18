// What a tap on the photo does to a photo's edits, and the undo history.
//
// The rules are those of the retired Windows app's desktop editor
// (onEditorClick), which were used on real photos:
//   - empty space: add a pin there (the core snaps it to the dot under it);
//   - a pin on the chain: remove it;
//   - a detection left out of the chain (red): use it as a pin;
//   - a pin the person added (blue): delete that edit;
//   - a removed pin (ghost): bring it back.
// Edits are always sent whole, and the core measures the photo again from them
// (session.PhotoSession.measure), so an edit is never applied locally.

import type { Edits, Marker, PhotoView, Point } from "../core/types";
import type { Pt } from "./viewport";

export type HitKind = "pin" | "mine" | "unused" | "removed";

export interface Hit {
  kind: HitKind;
  marker: Marker | { x: number; y: number; edit: number; r?: number };
  distance: number;
}

/** The smallest target, in screen px, around any marker. Fingers need more than a mouse. */
export const MIN_TARGET_PX = { mouse: 8, pen: 10, touch: 22 };

/** The nearest marker under a photo point: within 1.25 of its own radius, or
 * within the minimum target size converted to photo px (`slackPhotoPx`). */
export function hitTest(view: PhotoView, at: Pt, slackPhotoPx: number): Hit | null {
  let best: Hit | null = null;
  const consider = (kind: HitKind, m: Hit["marker"]) => {
    const d = Math.hypot(m.x - at.x, m.y - at.y);
    const reach = Math.max((m.r ?? view.r) * 1.25, slackPhotoPx);
    if (d <= reach && (!best || d < best.distance)) best = { kind, marker: m, distance: d };
  };
  for (const m of view.pins) consider(m.edit === undefined ? "pin" : "mine", m);
  for (const m of view.unused) consider(m.edit === undefined ? "unused" : "mine", m);
  for (const m of view.removed) consider("removed", m);
  return best;
}

/** The edits after a tap at `at` (photo px), and the point to mark as pending. */
export function editsAfterTap(view: PhotoView, at: Pt, slackPhotoPx: number): { edits: Edits; mark: Pt } {
  const e: Edits = { add: view.edits.add.map(copy), remove: view.edits.remove.map(copy) };
  const h = hitTest(view, at, slackPhotoPx);
  if (!h) {
    e.add.push([at.x, at.y]);
    return { edits: e, mark: at };
  }
  const m = h.marker;
  if (h.kind === "pin") e.remove.push([m.x, m.y]);
  else if (h.kind === "unused") e.add.push([m.x, m.y]);
  else if (h.kind === "mine") e.add.splice(m.edit!, 1);
  else e.remove.splice(m.edit!, 1);
  return { edits: e, mark: { x: m.x, y: m.y } };
}

/** What a tap on this would do, for a hint or an accessible label. */
export function describe(h: Hit | null): string {
  if (!h) return "Tap to add a pin here.";
  const m = h.marker as Marker;
  switch (h.kind) {
    case "pin":
      return `Pin ${m.k! + 1}. Tap to remove it.`;
    case "mine":
      return m.k === undefined
        ? "Your pin doesn't fit the chain (it must be about one link from its neighbours). Tap to delete it."
        : `Pin ${m.k + 1}, added by you${m.snapped ? "" : "; no dot to snap to, so it is where you tapped"}. Tap to delete it.`;
    case "unused":
      return "A stray dot that isn't on the chain. Tap to use it as a pin.";
    case "removed":
      return "A pin you removed. Tap to bring it back.";
  }
}

/** Undo history of one photo's edits. `commit` before sending new edits;
 * `rollback` if the send failed, so history matches what was measured. */
export class EditHistory {
  private readonly stack: Edits[] = [];

  get canUndo(): boolean {
    return this.stack.length > 0;
  }

  commit(current: Edits): void {
    this.stack.push(clone(current));
  }

  rollback(): void {
    this.stack.pop();
  }

  /** The edits to go back to, or null. If sending them fails, `restore` puts them back. */
  undo(): Edits | null {
    return this.stack.pop() ?? null;
  }

  restore(edits: Edits): void {
    this.stack.push(edits);
  }
}

function copy(p: Point): Point {
  return [p[0], p[1]];
}

function clone(e: Edits): Edits {
  return { add: e.add.map(copy), remove: e.remove.map(copy) };
}
