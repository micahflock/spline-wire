// The phone half of the S4 demo: take the pairing from the link, send sample
// curves through the outbox, show what is still waiting for the desktop.
import { deriveKeys } from "../../src/relay/crypto";
import { RelayLink, type LinkStatus } from "../../src/relay/link";
import { newCurveId, type Curve } from "../../src/relay/messages";
import { Outbox } from "../../src/relay/outbox";
import { phoneSecret } from "../../src/relay/pairing";
import { LocalStore } from "../../src/relay/store";
import { $, describe, relayUrl } from "./common";

const store = new LocalStore("splinewire-demo-phone:");
const secret = phoneSecret(store, location.hash, () => history.replaceState(null, "", location.pathname + location.search));
if (!secret) {
  $("pairing").textContent = "Not paired: open the link the desktop page shows.";
} else {
  const keys = await deriveKeys(secret);
  $("pairing").textContent = `Paired (room ${keys.room.slice(0, 6)}…).`;
  let last: LinkStatus = { state: "off", peers: 0 };
  const relay = new RelayLink({
    relayUrl: relayUrl(),
    keys,
    role: "phone",
    events: {
      status: (s) => {
        outbox.onStatus(s, last);
        last = s;
        $("status").textContent = describe(s);
      },
      message: (m) => outbox.onMessage(m),
    },
  });
  const outbox: Outbox = new Outbox(store, relay, () => {
    $("outbox").replaceChildren(
      ...outbox.items.map((i) => {
        const li = document.createElement("li");
        li.textContent = `${i.curve.photo}: ${i.state}`;
        return li;
      }),
    );
    $("delivered").textContent = String(outbox.delivered.length);
  });
  relay.start();

  let n = 0;
  const button = $("send") as HTMLButtonElement;
  button.disabled = false;
  button.onclick = () => void outbox.put(sample(++n));
}

function sample(n: number): Curve {
  const points: [number, number][] = Array.from({ length: 12 }, (_, i) => [i * 9.8, 20 * Math.sin(i / 3)]);
  return {
    v: 1, id: newCurveId(), measured_at: new Date().toISOString(), photo: `demo-${n}.jpg`,
    chain: { name: "O-ring chain", pitch_mm: 10, half_width_mm: 4, n_pins: 13 },
    points_mm: points, pins_mm: points.map(([x, y]) => [x, y + 4]),
    object_side: "outside", warnings: [], notes: [],
    diagnostics: { focal_source: "exif", focal_35mm: 26, tilt_deg: 10, missing_pins: 0,
      link_residual_rms_mm: 0.03, link_residual_max_mm: 0.06 },
  };
}
