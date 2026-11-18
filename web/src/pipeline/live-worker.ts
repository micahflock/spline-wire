// A small module worker that answers the live overlay's analyses with the
// TypeScript pipeline (live.ts): no Pyodide, so the camera view coaches from
// its first frame instead of after the engine's 8-11 s load. An analysis can
// take ~160 ms (focal length estimated, busy background), too long for the
// page's own thread. Messages: { id, points: FramePoints } in, { id, result }
// or { id, error } out.

import type { FramePoints } from "../core/protocol";
import { specFromDict } from "./chain";
import { analyse } from "./live";

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<{ id: number; points: FramePoints }>) => void) | null;
  postMessage(msg: unknown): void;
};

scope.onmessage = (e) => {
  const { id, points: p } = e.data;
  try {
    const spec = specFromDict(p.chain as unknown as Record<string, unknown>);
    scope.postMessage({ id, result: analyse(p.points, p.diameters, [p.width, p.height], spec, p.focal_px) });
  } catch (err) {
    scope.postMessage({ id, error: err instanceof Error ? err.message : String(err) });
  }
};
