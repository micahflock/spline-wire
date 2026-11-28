// The page's handle on the core worker: typed calls that return promises.
//
//   const core = new CoreClient(() => makeCoreWorker());
//   core.onProgress = (p) => ...;
//   const doc = await core.measure(photoId, options, () => blobFromIndexedDb(photoId));
//
// Calls run one at a time in the worker. A stale measure (the person changed
// the options again before the last one finished) is not cancelled, since a
// worker's work cannot be interrupted; instead `latest` lets the page skip
// work that is already out of date.

import {
  CoreError, type Command, type Commands, type EnginePreviews, type Progress, type Request, type Response,
} from "./protocol";
import type { MeasureOptions, MeasurementDocument } from "./types";

/** The parts of Worker the client uses, so tests can supply a fake. The
 * handlers take the DOM's event types so that a real Worker fits; a fake
 * passes `{ data } as MessageEvent`. */
export interface WorkerLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent) => void) | null;
  onerror: ((ev: ErrorEvent) => void) | null;
  terminate(): void;
}

export interface PhotoSource {
  name: string;
  /** A fresh copy of the photo's bytes: they are transferred to the worker. */
  bytes(): Promise<ArrayBuffer>;
}

export class CoreClient {
  onProgress: (progress: Progress) => void = () => {};
  /** Set it to have the engine make the previews of photos the browser cannot draw
   * (HEIC outside Safari) when it opens them for measuring: called with each. */
  onPreviews: ((photo: string, previews: EnginePreviews) => void) | null = null;
  private worker: WorkerLike | null = null;
  private nextId = 1;
  private readonly waiting = new Map<number, { resolve(v: unknown): void; reject(e: Error): void }>();
  private readonly generation = new Map<string, number>();

  constructor(private readonly make: () => WorkerLike) {}

  call<C extends Command>(cmd: C, args: Commands[C]["args"], transfer: Transferable[] = []): Promise<Commands[C]["result"]> {
    const worker = this.ensure();
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve: resolve as (v: unknown) => void, reject });
      worker.postMessage({ id, cmd, args } as Request, transfer);
    });
  }

  /** Start loading now, e.g. while the person is still taking the photo. */
  preload(): Promise<Commands["load"]["result"]> {
    return this.call("load", {});
  }

  /** Measure a photo, opening it in the worker first if it is not (or no longer) open there. */
  async measure(photo: string, options: MeasureOptions, source: PhotoSource): Promise<MeasurementDocument> {
    try {
      return await this.call("measure", { photo, options });
    } catch (err) {
      if (!(err instanceof CoreError) || err.failure.kind !== "closed") throw err;
    }
    const data = await source.bytes();
    const opened = await this.call("open", {
      photo, name: source.name, data, previews: !!this.onPreviews, channel: options.chain.gray_channel ?? "luma",
    }, [data]);
    if (opened.previews) this.onPreviews?.(photo, opened.previews);
    return this.call("measure", { photo, options });
  }

  /** For "measure the latest options only": returns a function that says
   * whether a newer request for this photo has been made since. */
  latest(photo: string): () => boolean {
    const g = (this.generation.get(photo) ?? 0) + 1;
    this.generation.set(photo, g);
    return () => this.generation.get(photo) === g;
  }

  /** Throw the worker away and free its memory (all of the wasm heap). The
   * next call starts a new one. */
  restart(): void {
    this.worker?.terminate();
    this.worker = null;
    this.failAll(new CoreError({ kind: "bug", message: "the measuring engine was restarted" }));
  }

  private ensure(): WorkerLike {
    if (this.worker) return this.worker;
    const w = this.make();
    w.onmessage = (ev) => this.receive(ev.data as Response);
    // A worker that dies (out of memory on a phone, most likely) takes its
    // requests with it: fail them, and let the next call start a new worker.
    w.onerror = (ev) => {
      if (this.worker === w) this.worker = null;
      w.terminate();
      this.failAll(new CoreError({ kind: "bug", type: "WorkerError",
        message: ev.message || "the measuring engine stopped (the browser may be out of memory)" }));
    };
    return (this.worker = w);
  }

  private receive(msg: Response): void {
    if ("event" in msg) {
      this.onProgress(msg.progress);
      return;
    }
    const w = this.waiting.get(msg.id);
    if (!w) return;
    this.waiting.delete(msg.id);
    if (msg.ok) w.resolve(msg.result);
    else w.reject(new CoreError(msg.error));
  }

  private failAll(err: Error): void {
    for (const w of this.waiting.values()) w.reject(err);
    this.waiting.clear();
  }
}
