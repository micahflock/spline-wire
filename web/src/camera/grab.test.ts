import { describe, expect, it, vi } from "vitest";
import { fullRegion } from "../overlay/region";
import { compareGrays, FrameGrab, GRAB, type GrabReply, type GrabRequest, type GrabWorkerLike, whyNotWorker } from "./grab";

/** A worker that answers each request as `answer` says (or never, with null). */
function fakeWorker(answer: (req: GrabRequest) => GrabReply | null) {
  const sent: GrabRequest[] = [];
  let terminated = 0;
  const w: GrabWorkerLike = {
    onmessage: null, onerror: null,
    postMessage(msg) {
      sent.push(msg);
      const reply = answer(msg);
      if (reply) queueMicrotask(() => w.onmessage?.({ data: reply } as MessageEvent<GrabReply>));
    },
    terminate: () => void terminated++,
  };
  return { w, sent, terminated: () => terminated };
}

const frame = () => ({ close: vi.fn() }) as unknown as VideoFrame;
const region = fullRegion(3024, 4032, 1200);
const shape = { inner: 0.8 };

describe("the frame grab in a worker", () => {
  it("is tried only where VideoFrame, OffscreenCanvas and workers exist, and not with ?grab=main; says why not", () => {
    const all = { VideoFrame: class {}, OffscreenCanvas: class {}, Worker: class {} };
    expect(whyNotWorker("", all)).toBeNull();
    expect(whyNotWorker("?grab=main", all)).toBe("?grab=main");
    expect(whyNotWorker("?engine=python", all)).toBeNull();
    expect(whyNotWorker("", { OffscreenCanvas: class {}, Worker: class {} })).toBe("no VideoFrame here");    // Firefox without WebCodecs
    expect(whyNotWorker("", { VideoFrame: class {}, Worker: class {} })).toBe("no OffscreenCanvas here");
    const page = new FrameGrab(() => fakeWorker(() => null).w, "no VideoFrame here");
    expect(page.mode).toBe("main");
    expect(page.describe()).toBe("grab on the page: no VideoFrame here");
    const tried = new FrameGrab(() => fakeWorker(() => null).w, null);
    expect(tried.describe()).toBe("grab on the page (checking the worker)");
    tried.checked("same");
    expect(tried.describe()).toBe("grab in a worker");
  });

  it("checks first, then hands frames over one at a time, transferring them", async () => {
    const fw = fakeWorker((req) => ({ id: req.id, blobs: [{ x: 1, y: 2, d: 3, contrast: 4, sharpness: 1 }], grabMs: 5, blobMs: 1 }));
    const grab = new FrameGrab(() => fw.w, null);
    expect(grab.mode).toBe("check");
    expect(grab.wants()).toBe(true);
    grab.checked("unsure");                     // too plain a frame: the page grabs a while, then checks again
    const turns = Array.from({ length: GRAB.recheckFrames + 1 }, () => grab.wants());
    expect(turns.filter(Boolean)).toHaveLength(1);
    expect(turns.at(-1)).toBe(true);
    grab.checked("same");
    expect(grab.mode).toBe("worker");
    expect(grab.wants()).toBe(true);
    const p = grab.run(frame(), region, shape);
    expect(grab.busy).toBe(true);
    const reply = await p;
    expect(grab.busy).toBe(false);
    expect(reply.blobs).toHaveLength(1);
    expect(fw.sent[0]).toMatchObject({ id: 1, region, shape, gray: false });
    await grab.run(frame(), region, shape, true);
    expect(fw.sent[1]!.gray).toBe(true);
  });

  it("rejects on the worker's error and when it does not answer, and the page can then grab", async () => {
    const failing = fakeWorker((req) => ({ id: req.id, error: "InvalidStateError: closed" }));
    const a = new FrameGrab(() => failing.w, null);
    await expect(a.run(frame(), region, shape)).rejects.toThrow("InvalidStateError");
    a.fallBack("it failed");
    expect(a.mode).toBe("main");
    expect(a.wants()).toBe(false);
    expect(a.why).toBe("it failed");
    expect(failing.terminated()).toBe(1);

    vi.useFakeTimers();
    try {
      const silent = fakeWorker(() => null);
      const b = new FrameGrab(() => silent.w, null, 500);
      const p = b.run(frame(), region, shape);
      const caught = expect(p).rejects.toThrow("did not answer");
      vi.advanceTimersByTime(600);
      await caught;
      expect(b.busy).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it("closes a frame it could not post", async () => {
    const f = frame();
    const w = fakeWorker(() => null).w;
    w.postMessage = () => { throw new DOMException("cannot transfer", "DataCloneError"); };
    await expect(new FrameGrab(() => w, null).run(f, region, shape)).rejects.toThrow("cannot transfer");
    expect(f.close).toHaveBeenCalled();
  });

  it("compares the worker's copy of a frame with the page's", () => {
    const pic = Uint8Array.from({ length: 48 * 64 }, (_, i) => ((i % 48) * 5 + Math.floor(i / 48) * 2) % 256);
    const near = pic.map((v) => Math.min(255, v + 2));
    expect(compareGrays(pic, near)).toBe("same");
    expect(compareGrays(pic, pic.slice().reverse())).toBe("different");          // turned half round
    expect(compareGrays(pic, pic.slice(1))).toBe("different");
    expect(compareGrays(new Uint8Array(100).fill(30), new Uint8Array(100).fill(30))).toBe("unsure");   // too plain to tell
    const c = new FrameGrab(() => fakeWorker(() => null).w, null);
    c.checked("different");
    expect(c.mode).toBe("main");
    expect(GRAB.maxDifference).toBeLessThan(GRAB.minContrast);
  });
});
