// What the phone and the desktop keep in step about each curve, beside the
// curve itself: its star, its name and whether it was deleted.
//
// The relay keeps nothing and the pages are seldom open together, so each page
// keeps its own copy and they merge whenever they meet (messages.ts `state`):
//
//   - the curve: only the phone changes it. Each change is a new revision; the
//     desktop keeps the highest it has seen (inbox.ts), the phone resends until
//     a state from the desktop shows it (outbox.ts);
//   - the star: either side sets it. The later setting wins (by the clock of
//     the device that set it; a tie: starred). Setting it twice is harmless;
//   - the name: the same, a person's name for the curve in place of the photo's
//     file name ("" gives the file name back). A tie: the greater text;
//   - deletion: either side deletes. A deleted curve stays deleted on both, and
//     its id is remembered (DELETED_CAP) so a late copy of it is not taken back in.
//
// Merging is the same whichever order the states arrive in, and a state may be
// repeated or lost: the next meeting repairs it.

import type { KeyValueStore } from "./store";
import { readJson } from "./store";
import { STATE_LIMITS } from "./messages";

export interface Star {
  on: boolean;
  /** ms since 1970, when it was set; 0: never. */
  at: number;
}

export const NO_STAR: Star = { on: false, at: 0 };

/** The setting that wins of two. */
export function laterStar(a: Star, b: Star): Star {
  if (a.at !== b.at) return a.at > b.at ? a : b;
  return a.on ? a : b;
}

export function sameStar(a: Star, b: Star): boolean {
  return a.on === b.on && a.at === b.at;
}

/** A person's name for a curve; "" (and at 0) when none was given. */
export interface Name {
  text: string;
  at: number;
}

export const NO_NAME: Name = { text: "", at: 0 };
/** The longest name kept. */
export const NAME_CHARS = 100;

export function laterName(a: Name, b: Name): Name {
  if (a.at !== b.at) return a.at > b.at ? a : b;
  return a.text >= b.text ? a : b;
}

export function sameName(a: Name, b: Name): boolean {
  return a.text === b.text && a.at === b.at;
}

/** A name as typed, tidied: one line, no control characters, at most NAME_CHARS. */
export function cleanName(text: string): string {
  return text.replace(/[\u0000-\u001f\u007f]+/g, " ").replace(/\s+/g, " ").trim().slice(0, NAME_CHARS);
}

/** A time for a new setting, later than `before`'s even when this device's clock is behind. */
export function stamp<T extends { at: number }>(before: T, now = Date.now()): number {
  return Math.max(now, before.at + 1);
}

/** Deleted ids remembered. */
export const DELETED_CAP = 1000;

/** The ids of deleted curves, oldest first, kept in a store. */
export class DeletedIds {
  private ids: string[];

  constructor(private readonly store: KeyValueStore, private readonly key = "deleted") {
    this.ids = readJson<string[]>(store, key, []);
  }

  has(id: string): boolean {
    return this.ids.includes(id);
  }

  /** Remember these; returns those that were new. */
  add(ids: readonly string[]): string[] {
    const fresh = [...new Set(ids)].filter((id) => !this.ids.includes(id));
    if (!fresh.length) return [];
    this.ids.push(...fresh);
    if (this.ids.length > DELETED_CAP) this.ids.splice(0, this.ids.length - DELETED_CAP);
    this.store.set(this.key, JSON.stringify(this.ids));
    return fresh;
  }

  /** The most recent, as many as a state message carries. */
  recent(): string[] {
    return this.ids.slice(-STATE_LIMITS.deleted);
  }
}
