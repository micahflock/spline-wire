// The desktop half of the S4 demo: show the pairing link, list what arrives.
// Remembers the pairing and the curves in localStorage, as the real page will.
import { deriveKeys } from "../../src/relay/crypto";
import { Inbox } from "../../src/relay/inbox";
import { RelayLink } from "../../src/relay/link";
import { desktopSecret, forgetPairing, pairingUrl } from "../../src/relay/pairing";
import { LocalStore } from "../../src/relay/store";
import { $, describe, relayUrl } from "./common";

const store = new LocalStore("splinewire-demo-desktop:");
const secret = desktopSecret(store);
const keys = await deriveKeys(secret);

const phonePage = new URL("phone.html", location.href);
phonePage.search = location.search;
const link = pairingUrl(phonePage.href, secret);
$("pair").textContent = link;
$("pair").setAttribute("href", link);

function show(): void {
  $("curves").replaceChildren(
    ...inbox.curves.map((c) => {
      const li = document.createElement("li");
      li.textContent = `${c.photo} (${c.points_mm.length} points, id ${c.id})`;
      li.dataset.id = c.id;
      return li;
    }),
  );
}

const relay = new RelayLink({
  relayUrl: relayUrl(),
  keys,
  role: "desktop",
  events: {
    status: (s) => ($("status").textContent = describe(s)),
    message: (m) => inbox.onMessage(m),
  },
});
const inbox = new Inbox(store, relay, { arrived: show });
show();
relay.start();

$("unpair").onclick = () => {
  forgetPairing(store);
  location.reload();
};
