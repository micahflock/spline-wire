// The page's side of live-worker.ts: analyse(points) as a promise, answers
// matched by id. The worker is made on first use; if it dies, the next call
// makes a new one.

import type { FrameAnalysis, FramePoints } from "../core/protocol";

interface WorkerLike {
  postMessage(msg: unknown): void;
  terminate(): void;
  onmessage: ((e: MessageEvent) => void) | null;
  onerror: ((e: ErrorEvent) => void) | null;
}

export class LiveAnalyser {
  private worker: WorkerLike | null = null;
  private next = 1;
  private readonly waiting = new Map<number, { resolve: (a: FrameAnalysis) => void; reject: (e: Error) => void }>();

  constructor(private readonly makeWorker: () => WorkerLike) {}

  analyse(points: FramePoints): Promise<FrameAnalysis> {
    const w = this.worker ?? this.start();
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      w.postMessage({ id, points });
    });
  }

  close(): void {
    this.worker?.terminate();
    this.worker = null;
    this.failAll("the live analysis was closed");
  }

  private start(): WorkerLike {
    const w = this.makeWorker();
    w.onmessage = (e: MessageEvent<{ id: number; result?: FrameAnalysis; error?: string }>) => {
      const p = this.waiting.get(e.data.id);
      if (!p) return;
      this.waiting.delete(e.data.id);
      if (e.data.error !== undefined) p.reject(new Error(e.data.error));
      else p.resolve(e.data.result!);
    };
    w.onerror = (e) => {
      this.worker = null;
      w.terminate();
      this.failAll(`the live analysis stopped: ${e.message}`);
    };
    this.worker = w;
    return w;
  }

  private failAll(message: string): void {
    for (const p of this.waiting.values()) p.reject(new Error(message));
    this.waiting.clear();
  }
}
