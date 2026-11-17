// Which photo the core measures next, and each photo's status.
//
//   loading    waiting for the core to finish loading (a photo taken while
//              Pyodide loads is measured as soon as it is ready)
//   queued     waiting for another photo's measurement
//   measuring
//   done       measured (document ok)
//   error      the document says what went wrong (with `view`, the photo can
//              still be fixed by hand), or there is no document: not a photo,
//              or a bug
//
// The worker runs one call at a time and Python cannot be interrupted, so the
// page asks for one measurement at a time too and coalesces: a photo whose
// options change while it is being measured (taps on the side buttons, a pin
// count typed digit by digit) is measured once more afterwards with the
// newest options, and the result in between is not shown. The photo the
// person is editing goes ahead of photos picked in bulk from the library.

import type { PhotoSource } from "../core/client";
import { CoreError, type Failure } from "../core/protocol";
import type { MeasureOptions, MeasurementDocument } from "../core/types";
import type { Status } from "./photos";

export interface CoreLike {
  measure(photo: string, options: MeasureOptions, source: PhotoSource): Promise<MeasurementDocument>;
}

export interface Update {
  status: Status;
  doc?: MeasurementDocument;
  failure?: Failure | null;
  /** How long the measurement took, ms. */
  ms?: number;
}

export interface MeasurerDeps {
  core: CoreLike;
  /** The options to measure a photo with now; null if it was deleted. */
  options(id: string): MeasureOptions | null;
  source(id: string): PhotoSource;
  update(id: string, update: Update): void;
  now?: () => number;
}

export class Measurer {
  private ready = false;
  private readonly waiting: string[] = [];
  private running: string | null = null;
  private readonly again = new Set<string>();
  private readonly now: () => number;

  constructor(private readonly deps: MeasurerDeps) {
    this.now = deps.now ?? (() => performance.now());
  }

  /** The core has loaded: start on what is waiting. */
  coreReady(): void {
    this.ready = true;
    for (const id of this.waiting) this.deps.update(id, { status: "queued" });
    void this.pump();
  }

  /** The core is loading again (its worker died): photos wait for it. */
  coreLoading(): void {
    this.ready = false;
    for (const id of this.waiting) this.deps.update(id, { status: "loading" });
  }

  /** Measure this photo with its current options. `urgent`: the person is
   * looking at it, so it goes before the others. */
  request(id: string, urgent = false): void {
    if (this.running === id) {
      this.again.add(id);
      return;
    }
    const at = this.waiting.indexOf(id);
    if (at >= 0) this.waiting.splice(at, 1);
    if (urgent) this.waiting.unshift(id);
    else this.waiting.push(id);
    this.deps.update(id, { status: this.ready ? "queued" : "loading" });
    void this.pump();
  }

  /** The photo was deleted. */
  forget(id: string): void {
    const at = this.waiting.indexOf(id);
    if (at >= 0) this.waiting.splice(at, 1);
    this.again.delete(id);
  }

  /** Whether a measurement of this photo is waiting or under way: the page
   * takes no taps on it meanwhile, since a tap's meaning depends on the
   * pins the last measurement found. */
  busy(id: string): boolean {
    return this.running === id || this.waiting.includes(id);
  }

  private async pump(): Promise<void> {
    if (!this.ready || this.running !== null) return;
    const id = this.waiting.shift();
    if (id === undefined) return;
    this.running = id;
    let update: Update | null = null;
    try {
      update = await this.run(id);
    } finally {
      this.running = null;
    }
    // Options changed meanwhile: this result is already out of date, and the
    // next measurement follows at once. Otherwise it is published once the
    // photo is no longer busy, so the page can act on it at once.
    if (this.again.delete(id)) this.waiting.unshift(id);
    else if (update) this.deps.update(id, update);
    void this.pump();
  }

  private async run(id: string): Promise<Update | null> {
    const options = this.deps.options(id);
    if (!options) return null;
    this.deps.update(id, { status: "measuring" });
    const t0 = this.now();
    let update: Update;
    try {
      const doc = await this.deps.core.measure(id, options, this.deps.source(id));
      update = { status: doc.ok ? "done" : "error", doc, failure: null, ms: this.now() - t0 };
    } catch (err) {
      update = { status: "error", failure: toFailure(err), ms: this.now() - t0 };
    }
    return update;
  }
}

export function toFailure(err: unknown): Failure {
  if (err instanceof CoreError) return err.failure;
  const e = err as { name?: string; message?: string };
  return { kind: "bug", type: e?.name ?? "Error", message: String(e?.message ?? err) };
}
