// Which curve the page shows.
//
// A new curve from the phone is shown at once, which is what someone who has
// just measured wants, unless the person picked a curve themselves in the last
// minute: then they are reading or copying it, and a curve arriving must not
// pull it away from under them. The new one is marked in the list instead.

/** How long a person's own pick holds against arriving curves. */
export const HOLD_MS = 60_000;

export class Selection {
  private picked: number | null = null;

  constructor(public id: string | null = null) {}

  /** The person chose a curve (click, keyboard). */
  pick(id: string, now: number): void {
    this.id = id;
    this.picked = now;
  }

  /** A curve arrived. True if it is now the one shown. */
  arrived(id: string, now: number): boolean {
    if (this.id !== null && this.picked !== null && now - this.picked < HOLD_MS) return false;
    this.id = id;
    return true;
  }

  /** Make sure the shown curve is one in `ids` (newest first): after a load or
   * a delete. A deleted curve's place goes to the next older one, else the newer. */
  keepIn(ids: readonly string[], removedAt = -1): void {
    if (this.id !== null && ids.includes(this.id)) return;
    if (ids.length === 0) {
      this.id = null;
    } else if (removedAt >= 0) {
      this.id = ids[Math.min(removedAt, ids.length - 1)]!;
    } else {
      this.id = ids[0]!;
    }
    // Not the person's pick: the next curve to arrive may take its place.
    this.picked = null;
  }
}
