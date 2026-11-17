// The desktop page: pair a phone by its QR code, receive its curves, show
// them, copy their points for Fusion and save their files.
//
// The logic lives in the modules beside this one and in src/relay/; this file
// wires them to the page and draws. Everything drawn from a curve passes
// through esc(): curves come from the paired phone, which may run another build.

import { deriveKeys } from "../relay/crypto";
import { phonePageUrl, pageRelaySetting, relayAddress } from "../relay/config";
import { deviceName } from "../relay/device";
import { unpack } from "../relay/files";
import { Inbox } from "../relay/inbox";
import { RelayLink, type LinkStatus } from "../relay/link";
import { pointsTsv, type Curve, type Message } from "../relay/messages";
import { desktopSecret, forgetPairing, pairingUrl, storedSecret } from "../relay/pairing";
import { LocalStore, MemoryStore, readJson, type KeyValueStore } from "../relay/store";
import { DOWNLOADS, downloadName, type DownloadKind } from "./names";
import { plotSvg } from "./plot";
import { qrSvg } from "./qr";
import { Selection } from "./selection";
import { splineSamples } from "./spline";
import { statusLine, type PhoneSeen } from "./status";
import { UnseenTitle } from "./title";
import { exactTime, relativeTime } from "./when";
import { FOCAL_SOURCE, esc, plural, readNote, readWarning } from "./wording";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const APP = `Spline Wire ${import.meta.env.APP_VERSION}`;
const BUILD = import.meta.env.BUILD_ID;
// This page lives at desktop/ under the site's root for now; at the root it becomes "./".
const SITE_ROOT = new URL("../", location.href).href;
const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform);
/** Where "a phone has paired" is remembered, so the big QR code does not come back on every visit. */
const PHONE_KEY = "phone";

// Its own prefix: a phone page open in the same browser (testing on one
// computer) keeps its own pairing apart from this one.
const persistent = storageWorks();
const store: KeyValueStore = persistent ? new LocalStore("splinewire-desktop:") : new MemoryStore();
const relayUrl = relayAddress(pageRelaySetting());
const selection = new Selection();
const unseen = new UnseenTitle("Spline Wire");

let secret = desktopSecret(store);
let link: RelayLink | null = null;
let inbox: Inbox;
let linkStatus: LinkStatus = { state: "off", peers: 0 };
let retryAt = 0;
/** The phone in the room now, from its hello. */
let phoneHere: PhoneSeen | null = null;
let flashId: string | null = null;
/** Downloads of the curve on show; their blob: addresses are freed when it changes. */
let blobUrls: string[] = [];

$("storage-note").hidden = persistent;
$("relay-note").hidden = relayUrl !== null;
$("build").textContent = `${APP} · build ${BUILD}`;
$("relay-name").textContent = relayUrl ? `relay ${new URL(relayUrl).host}` : "no relay";

await connect();
selection.keepIn(ids());
renderAll();

// ---- the relay ------------------------------------------------------------

/** Join the pairing's room (again, after unpairing). */
async function connect(): Promise<void> {
  link?.stop();
  phoneHere = null;
  const keys = await deriveKeys(secret);
  // Only the current link speaks: a frame the old one was still opening when it
  // stopped must not mark the new pairing as paired.
  const next: RelayLink | null = relayUrl
    ? new RelayLink({ relayUrl, keys, role: "desktop", events: {
        status: (s) => void (link === next && onStatus(s)),
        message: (m) => void (link === next && onMessage(m)),
      } })
    : null;
  link = next;
  loadInbox();
  link?.start();
}

/** The inbox reads its list from the store, so a new one keeps every curve. */
function loadInbox(): void {
  inbox = new Inbox(store, { send: (m) => link?.send(m) ?? Promise.resolve(false) }, arrived);
}

function onStatus(s: LinkStatus): void {
  const before = linkStatus;
  linkStatus = s;
  if (s.state !== "open" || s.peers < 1) phoneHere = null;
  if (s.state === "waiting") retryAt = Date.now() + (s.retryInMs ?? 0);
  // Say hello whenever a phone may have appeared: its outbox sends what it holds on hearing it.
  if (s.state === "open" && (before.state !== "open" || s.peers > before.peers)) {
    void link?.send({ type: "hello", role: "desktop", device: deviceName(navigator.userAgent), app: `${APP} (${BUILD})` });
  }
  renderStatus();
}

function onMessage(m: Message): void {
  if (m.type === "hello" && m.role === "phone") {
    phoneHere = { device: m.device, app: m.app };
    rememberPhone(m.device);
    renderStatus();
  }
  inbox.onMessage(m);
}

function arrived(curve: Curve): void {
  if (!phoneHere) phoneHere = { device: "" };
  rememberPhone(phoneHere.device);
  const shown = selection.arrived(curve.id, Date.now());
  unseen.arrived(document.hidden);
  document.title = unseen.title;
  flashId = curve.id;
  renderList();
  renderPairing();
  if (shown) renderMain();
  renderStatus();
  toast(shown ? `New curve: ${curve.photo}` : `New curve: ${curve.photo}, at the top of the list`);
}

function rememberPhone(device: string): void {
  const had = lastPhone();
  store.set(PHONE_KEY, JSON.stringify({ device: device || had?.device || "", at: new Date().toISOString() }));
  if (!had) renderAll();
}

/** The phone that last said hello or sent a curve on this pairing; null before any has. */
function lastPhone(): { device: string; at: string } | null {
  return readJson<{ device: string; at: string } | null>(store, PHONE_KEY, null);
}

function paired(): boolean {
  return lastPhone() !== null;
}

function ids(): string[] {
  return inbox.curves.map((c) => c.id);
}

function current(): Curve | null {
  return inbox.curves.find((c) => c.id === selection.id) ?? null;
}

// ---- drawing --------------------------------------------------------------

function renderAll(): void {
  renderStatus();
  renderList();
  renderPairing();
  renderMain();
}

function renderStatus(): void {
  if (!link) {
    $("lamp").className = "lamp bad";
    $("link-text").textContent = "No relay";
    $("link-wait").textContent = "";
    $("retry").hidden = true;
    return;
  }
  const line = statusLine(linkStatus, phoneHere, retryAt - Date.now());
  $("lamp").className = `lamp ${line.lamp}`;
  if ($("link-text").textContent !== line.text) $("link-text").textContent = line.text;
  $("link-wait").textContent = line.countdown ?? "";
  $("link-state").title = phoneHere?.app ? `The phone runs ${phoneHere.app}` : "";
  $("retry").hidden = !line.retry;
}

function renderList(): void {
  const list = $("list");
  const focused = (document.activeElement as HTMLElement | null)?.dataset?.id;
  const now = Date.now();
  list.innerHTML = inbox.curves.map((c) => {
    const t = Date.parse(c.measured_at);
    const w = c.warnings.length;
    return `<li><button type="button" class="item${c.id === flashId ? " flash" : ""}" data-id="${esc(c.id)}"
      aria-current="${c.id === selection.id}">
      <span class="row1"><span class="name">${esc(c.photo)}</span>${w ? `<span class="warns">${plural(w, "warning")}</span>` : ""}</span>
      <span class="sub"><time datetime="${esc(c.measured_at)}" title="${esc(exactTime(t))}">${esc(relativeTime(t, now))}</time>
        <span>· ${esc(c.chain.name)}</span><span>· ${c.chain.n_pins} pins</span></span>
    </button></li>`;
  }).join("");
  if (flashId) {
    const id = flashId;
    flashId = null;
    setTimeout(() => list.querySelector(`[data-id="${CSS.escape(id)}"]`)?.classList.remove("flash"), 1800);
  }
  $("count").textContent = inbox.curves.length ? String(inbox.curves.length) : "";
  $("none").hidden = inbox.curves.length > 0;
  if (focused) list.querySelector<HTMLElement>(`[data-id="${CSS.escape(focused)}"]`)?.focus();
}

function renderPairing(): void {
  const side = $("pair-side");
  const url = pairingUrl(phonePageUrl(SITE_ROOT, location.search), secret);
  const phone = lastPhone();
  if (!paired() && inbox.curves.length === 0) {
    side.innerHTML = "";            // the welcome in the middle shows the code
    return;
  }
  // Open until a phone has paired, then folded away; a person's own open or close is kept while that holds.
  const was = side.querySelector("details");
  const open = was && side.dataset.paired === String(paired()) ? was.open : !paired();
  side.dataset.paired = String(paired());
  side.innerHTML = `<details class="pair"${open ? " open" : ""}>
    <summary>${paired() ? "Pair another phone" : "Pair a phone"}</summary>
    <div class="body">
      ${phone?.device ? `<p>Paired with ${esc(phone.device)}.</p>` : ""}
      ${qrSvg(url, "QR code: the phone page, paired with this computer")}
      <p>Scan it with the phone's Camera app. Any number of phones can pair with this computer.</p>
      <div class="url" data-pair-link>${esc(url)}</div>
      <div class="row"><button type="button" class="btn small" data-act="copy-link">Copy link</button>
        ${paired() ? `<button type="button" class="btn small plain" data-act="unpair">Unpair…</button>` : ""}</div>
    </div>
  </details>`;
}

function renderMain(): void {
  const main = $("main");
  for (const u of blobUrls) URL.revokeObjectURL(u);
  blobUrls = [];
  const c = current();
  main.classList.toggle("empty", !c);
  if (!c) {
    main.innerHTML = paired() ? waitingHtml() : welcomeHtml();
    return;
  }
  main.innerHTML = curveHtml(c);
  void fillDownloads(c);
}

function welcomeHtml(): string {
  const link = pairingUrl(phonePageUrl(SITE_ROOT, location.search), secret);
  return `<section class="welcome" aria-labelledby="welcome-title">
    ${qrSvg(link, "QR code: the phone page, paired with this computer")}
    <div><h2 id="welcome-title">Pair your phone</h2>
      <ol><li>Scan this code with the phone's Camera app and open the page.</li>
        <li>Photograph the chain there. The phone measures it.</li>
        <li>The curve shows up here, ready for <b>Copy points</b> and Fusion's <b>Paste points</b>.</li></ol>
      <p class="then">Done once per phone: both remember the pairing. The phone and this computer can be on different networks.</p>
      <div class="url" data-pair-link>${esc(link)}</div>
      <button type="button" class="btn small" data-act="copy-link">Copy link</button>
    </div>
  </section>`;
}

function waitingHtml(): string {
  return `<section class="welcome waiting" aria-labelledby="welcome-title">
    <div><h2 id="welcome-title">Waiting for a curve</h2>
      <p>Measure the chain on the paired phone. Each curve shows up here within seconds, and stays on this computer until you delete it.</p>
      <p class="then">If the phone measures while this page is closed, it keeps the curve and sends it when the page is open again.</p></div>
  </section>`;
}

function curveHtml(c: Curve): string {
  const t = Date.parse(c.measured_at);
  const d = c.diagnostics;
  const spline = splineSamples(c.points_mm);
  const xs = [...spline, ...c.pins_mm].map((p) => p[0]);
  const ys = [...spline, ...c.pins_mm].map((p) => p[1]);
  const size = `${Math.round(Math.max(...xs) - Math.min(...xs))} × ${Math.round(Math.max(...ys) - Math.min(...ys))} mm`;
  const plot = plotSvg(c.points_mm, c.pins_mm, spline, `The measured curve, ${plural(c.points_mm.length, "point")}, ${size}, on a millimetre grid`);
  const found = c.chain.n_pins - d.missing_pins;
  const cells: [string, string, string, boolean][] = [
    ["Points", `${c.points_mm.length}`, "on the object's edge", false],
    ["Pins", `${found}<small>/${c.chain.n_pins}</small>`, d.missing_pins ? `${d.missing_pins} missing` : "all found", d.missing_pins > 0],
    ["Tilt", `${d.tilt_deg.toFixed(1)}<small>°</small>`, "camera to table", false],
    ["Link error", d.link_residual_rms_mm.toFixed(3), `rms · max ${d.link_residual_max_mm.toFixed(3)} mm`,
      // Python warns above 5% of the pitch (pipeline.py).
      d.link_residual_max_mm > 0.05 * c.chain.pitch_mm],
    ["Focal length", `${+d.focal_35mm.toFixed(1)}<small>mm</small>`, `35 mm equiv. · ${FOCAL_SOURCE[d.focal_source] ?? d.focal_source}`,
      d.focal_source === "estimated"],
    ["Object", c.object_side === "inside" ? "Inside" : "Outside",
      c.object_side === "inside" ? "chain wrapped around it" : "chain pressed into it", false],
  ];
  const notes = [...c.warnings.map(readWarning), ...c.notes.map(readNote)]
    .map((r) => `<div class="note ${r.level}"><b>${esc(r.title)}</b>${esc(r.text)}</div>`).join("");
  return `<section class="view" aria-labelledby="curve-title">
    <div class="head"><h2 id="curve-title">${esc(c.photo)}</h2>
      <span class="meta"><time datetime="${esc(c.measured_at)}" title="${esc(exactTime(t))}">measured ${esc(relativeTime(t, Date.now()))}</time> · ${esc(c.chain.name)}</span>
      <span class="spacer"></span>
      <button type="button" class="btn plain small" data-act="delete">Delete…</button></div>
    <div class="plot">${plot}</div>
    <div class="legend" aria-hidden="true">
      <span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="5" fill="none" stroke="var(--screen-ink)" stroke-opacity=".55"/></svg>Pin of the chain</span>
      <span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="3.5" fill="var(--plot-pt)"/></svg>Point on the object's edge</span>
      <span><svg viewBox="0 0 14 14"><line x1="1" y1="7" x2="13" y2="7" stroke="var(--trace)" stroke-width="2"/></svg>Spline through the points</span>
    </div>
  </section>
  <aside class="read" aria-label="Measurement">
    <div class="sec"><span>Curve · mm</span><span class="mono">${esc(size)}</span></div>
    <div class="scroll">
      <div class="nums">${cells.map(([k, v, s, warn]) =>
        `<div class="cell${warn ? " warn" : ""}"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${esc(s)}</div></div>`).join("")}</div>
      ${notes ? `<div class="notes">${notes}</div>` : ""}
      <details class="about"><summary>About this curve</summary>
        <div class="kv"><span>Measured</span><span>${esc(exactTime(t))}</span></div>
        <div class="kv"><span>Chain</span><span>${esc(c.chain.name)}</span></div>
        <div class="kv"><span>Link length</span><span>${c.chain.pitch_mm} mm</span></div>
        <div class="kv"><span>Pin to edge</span><span>${c.chain.half_width_mm} mm</span></div>
        <div class="kv"><span>Id</span><span>${esc(c.id)}</span></div>
      </details>
    </div>
    <div class="out">
      <button type="button" class="copy" data-act="copy"><span>Copy points</span><kbd>${IS_MAC ? "⌘ C" : "Ctrl C"}</kbd></button>
      <div class="then">then <b>Paste points</b> in a Fusion sketch</div>
      <div class="exp" id="exp">Save as …</div>
    </div>
  </aside>`;
}

/** The downloads: the phone's files unpacked, and the clipboard table. */
async function fillDownloads(c: Curve): Promise<void> {
  const texts: Partial<Record<DownloadKind, string>> = c.files ? await unpack(c.files) : {};
  texts.tsv = pointsTsv(c.points_mm);
  if (selection.id !== c.id || !$("exp")) return;      // another curve is on show by now
  const links = DOWNLOADS.filter((d) => texts[d.kind] !== undefined).map((d) => {
    const url = URL.createObjectURL(new Blob([texts[d.kind]!], { type: d.type }));
    blobUrls.push(url);
    const name = downloadName(d.kind, c.photo);
    return `<a class="btn" href="${url}" download="${esc(name)}" data-kind="${d.kind}" title="${esc(name)}">${d.label}</a>`;
  });
  const missing = c.files ? "" : `<span>The phone sent this curve without its files (too long for one message): only the points can be saved.</span>`;
  $("exp").innerHTML = `Save as ${links.join("")}${missing}`;
}

// ---- actions --------------------------------------------------------------

function pick(id: string, focus = false): void {
  const changed = selection.id !== id;
  selection.pick(id, Date.now());          // a click on the curve on show holds it too
  if (!changed) return;
  renderList();
  renderMain();
  if (focus) $("list").querySelector<HTMLElement>(`[data-id="${CSS.escape(id)}"]`)?.focus();
}

/** The text a copy now puts on the clipboard, for the copy event when the Clipboard API is refused. */
let copying: string | null = null;

async function copyPoints(): Promise<void> {
  const c = current();
  if (!c) return;
  const text = pointsTsv(c.points_mm);
  let ok = false;
  try {
    await navigator.clipboard.writeText(text);
    ok = true;
  } catch {
    // No Clipboard API (a page served over plain http on the network): the old way, through the copy event below.
    copying = text;
    try {
      ok = document.execCommand("copy");
    } finally {
      copying = null;
    }
  }
  toast(ok ? `Copied ${plural(c.points_mm.length, "point")}. In Fusion, click Paste points while editing a sketch.`
    : "Couldn't copy. Use Save as TSV instead.");
}

async function copyLink(): Promise<void> {
  const text = pairingUrl(phonePageUrl(SITE_ROOT, location.search), secret);
  try {
    await navigator.clipboard.writeText(text);
    toast("Link copied. Open it on the phone.");
  } catch {
    toast("Couldn't copy the link. Select it and copy it by hand.");
  }
}

async function remove(): Promise<void> {
  const c = current();
  if (!c) return;
  const sure = await confirmBox(`Delete ${c.photo}?`,
    "Its points and files go from this computer. Save them first if you need them; the phone does not keep a copy.", "Delete");
  if (!sure) return;
  const at = ids().indexOf(c.id);
  inbox.remove(c.id);
  selection.keepIn(ids(), at);
  renderList();
  renderPairing();
  renderMain();
  const next = selection.id && $("list").querySelector<HTMLElement>(`[data-id="${CSS.escape(selection.id)}"]`);
  (next || $("main")).focus();
  toast(`Deleted ${c.photo}`);
}

async function unpair(): Promise<void> {
  const sure = await confirmBox("Unpair every phone?",
    "This computer gets a new pairing code. Phones paired with the old one can't send curves here until they scan the new code. The curves already here stay.",
    "Unpair");
  if (!sure) return;
  forgetPairing(store);
  store.remove(PHONE_KEY);
  secret = desktopSecret(store);
  await connect();
  renderAll();
  toast("Unpaired. Scan the new code with the phone to pair it again.");
}

function confirmBox(title: string, body: string, yes: string): Promise<boolean> {
  const d = $<HTMLDialogElement>("confirm");
  $("confirm-title").textContent = title;
  $("confirm-body").textContent = body;
  $("confirm-yes").textContent = yes;
  d.returnValue = "";
  d.showModal();
  return new Promise((resolve) => d.addEventListener("close", () => resolve(d.returnValue === "yes"), { once: true }));
}

function toast(message: string, ms = 3200): void {
  const t = $("toast");
  t.textContent = message;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), ms);
}
let toastTimer: ReturnType<typeof setTimeout> | undefined;

function storageWorks(): boolean {
  try {
    const k = "splinewire-desktop:probe";
    localStorage.setItem(k, "1");
    localStorage.removeItem(k);
    return true;
  } catch {
    return false;
  }
}

// ---- events ---------------------------------------------------------------

document.addEventListener("click", (ev) => {
  const t = (ev.target as Element).closest<HTMLElement>("[data-id], [data-act]");
  if (!t) return;
  if (t.dataset.id) return pick(t.dataset.id);
  const act = t.dataset.act;
  if (act === "copy") void copyPoints();
  else if (act === "copy-link") void copyLink();
  else if (act === "delete") void remove();
  else if (act === "unpair") void unpair();
});

$("retry").addEventListener("click", () => link?.retryNow());

// Arrow keys move through the list, as in a file list; Delete deletes.
$("list").addEventListener("keydown", (ev) => {
  const at = ids().indexOf((ev.target as HTMLElement).dataset.id ?? "");
  if (at < 0) return;
  const to = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: ids().length - 1 }[ev.key];
  if (to !== undefined) {
    ev.preventDefault();
    const id = ids()[Math.max(0, Math.min(ids().length - 1, to))];
    if (id) pick(id, true);
  } else if (ev.key === "Delete") {
    ev.preventDefault();
    pick(ids()[at]!);                     // the focused curve, which Tab may have reached without showing it
    void remove();
  }
});

// Ctrl+C (⌘C) with nothing selected in the page copies the curve's points, as in today's app.
function nothingSelected(): boolean {
  const el = document.activeElement;
  const typing = el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || (el as HTMLElement | null)?.isContentEditable;
  return !typing && !String(getSelection() ?? "") && !$<HTMLDialogElement>("confirm").open;
}
document.addEventListener("keydown", (ev) => {
  if ((ev.ctrlKey || ev.metaKey) && !ev.altKey && !ev.shiftKey && ev.key.toLowerCase() === "c" && nothingSelected() && current()) {
    ev.preventDefault();
    void copyPoints();
  }
});
document.addEventListener("copy", (ev) => {
  // Edit > Copy with nothing selected, and copyPoints' fallback.
  const text = copying ?? (nothingSelected() ? (current() && pointsTsv(current()!.points_mm)) : null);
  if (!text || !ev.clipboardData) return;
  ev.clipboardData.setData("text/plain", text);
  ev.preventDefault();
});

// Back on the network or back to the tab: try the relay now, not at the end of a 30 s wait.
window.addEventListener("online", () => link?.retryNow());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  link?.retryNow();
  unseen.seen();
  document.title = unseen.title;
});

// The retry countdown, and "2 min ago" in the list.
setInterval(() => {
  if (linkStatus.state === "waiting") renderStatus();
}, 1000);
setInterval(() => {
  const now = Date.now();
  for (const el of document.querySelectorAll<HTMLTimeElement>("time[datetime]")) {
    const t = Date.parse(el.dateTime);
    el.textContent = (el.closest(".meta") ? "measured " : "") + relativeTime(t, now);
  }
}, 30_000);

// Another tab of this page unpaired, received or deleted: follow it, so the
// two never disagree and neither writes back a list the other has changed.
window.addEventListener("storage", (ev) => {
  const key = ev.key?.replace(/^splinewire-desktop:/, "");
  if (key === "pairing") {
    // Only a new secret: between the other tab's forget and its new one there is none,
    // and making one here would start a second pairing.
    const now = storedSecret(store);
    if (!now || now === secret) return;
    secret = now;
    void connect().then(renderAll);
    return;
  }
  if (key !== "inbox" && key !== "inbox-seen" && key !== PHONE_KEY) return;
  const shown = selection.id;
  loadInbox();
  selection.keepIn(ids());
  renderList();
  renderPairing();
  if (selection.id !== shown || !current()) renderMain();
});

