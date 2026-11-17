// A minimal phone for desktop-check.ts: the real backbone (link, outbox,
// pairing, packed files, curveFrom) with no page around it. The check calls
// window.sendCurve with a Python measurement document and its output files.
import type { MeasurementDocument } from "../../src/core/types";
import { curveFrom, type CurveContext } from "../../src/phone/curve";
import { deriveKeys } from "../../src/relay/crypto";
import { pack, type FileKind } from "../../src/relay/files";
import { RelayLink, type LinkStatus } from "../../src/relay/link";
import { Outbox } from "../../src/relay/outbox";
import { phoneSecret } from "../../src/relay/pairing";
import { MemoryStore } from "../../src/relay/store";

const store = new MemoryStore();
const secret = phoneSecret(store, location.hash, () => history.replaceState(null, "", location.pathname + location.search));
if (!secret) throw new Error("no pairing secret in the address");
const relayUrl = new URLSearchParams(location.search).get("relay");
if (!relayUrl) throw new Error("no ?relay= in the address");

let last: LinkStatus = { state: "off", peers: 0 };
const link: RelayLink = new RelayLink({
  relayUrl,
  keys: await deriveKeys(secret),
  role: "phone",
  events: {
    status: (s) => {
      if (s.state === "open" && (last.state !== "open" || s.peers > last.peers)) {
        void link.send({ type: "hello", role: "phone", device: "Test phone", app: "desktop check" });
      }
      outbox.onStatus(s, last);
      last = s;
      document.getElementById("status")!.textContent = `${s.state} ${s.peers}`;
    },
    message: (m) => outbox.onMessage(m),
  },
});
const outbox: Outbox = new Outbox(store, link);
link.start();

Object.assign(window, {
  async sendCurve(doc: MeasurementDocument, outputs: Partial<Record<FileKind | "tsv", string>>, photo: string,
    ctx: CurveContext): Promise<string> {
    const curve = curveFrom({ ...doc, name: photo }, ctx, await pack(outputs))!;
    await outbox.send(curve);
    return curve.id;
  },
  delivered: () => outbox.delivered.length,
});
