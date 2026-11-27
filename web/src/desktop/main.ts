// The desktop page: pair a phone by its QR code, receive its curves, show
// them, copy their points for Fusion and save their files. A photo dropped on
// the page (or picked with its button) is measured here instead (D9), by
// here.ts, which is loaded only then: it brings the measuring engine.
//
// With ?fusion in its address it is the page inside the Spline Wire add-in's
// palette in Fusion (fusion.ts; docs/fusion-palette.md): one narrow column, its
// own pairing, no clipboard, downloads or measuring here, and each curve that
// arrives goes into the sketch being edited (or waits for Insert).
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
import {
  AUTO_KEY, FusionBridge, InsertLog, addinHasCorners, autoInsert, fusionMode, markText, waitForSend,
  type FusionStatus, type How, type Inserted,
} from "./fusion";
import { DOWNLOADS, downloadName, type DownloadKind } from "./names";
import { plotSvg } from "./plot";
import { curveDrawing } from "../phone/drawings";
import { dxfText } from "../pipeline/dxf";
import { pieces, shape, toggleMark, type Shaped } from "./corners";
import { qrSvg } from "./qr";
import { Selection } from "./selection";
import { splineSamples } from "./spline";
import { statusLine, type PhoneSeen } from "./status";
import { UnseenTitle } from "./title";
import type { Draft, DraftHost } from "./here";
import { exactTime, relativeTime } from "./when";
import { FOCAL_SOURCE, FOCAL_SOURCE_HERE, esc, plural, readNote, readWarning } from "./wording";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const APP = `Spline Wire ${import.meta.env.APP_VERSION}`;
const BUILD = import.meta.env.BUILD_ID;
// This page is the site's root (index.html); the phone page is phone/ under it.
/** Paste points is the Spline Wire add-in's button: how to get it, for whoever
 * does not have it yet. The zip is fusion/SplineWire, put on the site by
 * `npm run assets`. Fusion runs an add-in from any folder it is pointed at,
 * so nothing has to be copied into its own folders (decision O5). */
/** Under the drawing: how to mark a corner (corners.ts). */
const CORNER_HINT = "Step or ledge rounded off? Click the point at its corner to make it sharp.";
const ADDIN_HELP = `<details class="addin"><summary>No Paste points button in Fusion?</summary>
  <ol>
    <li><a href="SplineWire-fusion-addin.zip" download>Download the add-in</a> and unzip it anywhere you will keep it.</li>
    <li>In Fusion: <b>Utilities</b> → <b>Add-Ins</b> → <b>Scripts and Add-Ins</b>, the <b>Add-Ins</b> tab, then <b>+</b> and pick the unzipped <b>SplineWire</b> folder.</li>
    <li>Select <b>SplineWire</b>, tick <b>Run on Startup</b> and click <b>Run</b>. <b>Paste points</b> and <b>Add from phone</b> appear in the sketch toolbar.</li>
  </ol>
</details>`;

const SITE_ROOT = new URL("./", location.href).href;
const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform);
/** Where "a phone has paired" is remembered, so the big QR code does not come back on every visit. */
const PHONE_KEY = "phone";

/** In Fusion's palette (?fusion). */
const FUSION = fusionMode(location.search);
const UNDO = IS_MAC ? "⌘Z" : "Ctrl+Z";

// Its own prefix: a phone page open in the same browser (testing on one
// computer) keeps its own pairing apart from this one, and so does the Fusion
// mode (its palette is a pairing of its own).
const PREFIX = FUSION ? "splinewire-fusion:" : "splinewire-desktop:";
const persistent = storageWorks();
const store: KeyValueStore = persistent ? new LocalStore(PREFIX) : new MemoryStore();
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
/** A photo being measured on this computer, not yet added to the list; one at a time. */
let draft: Draft | null = null;
/** Whether the draft, rather than a curve, is on show. */
let showingDraft = false;
/** The curve whose name is being edited in its heading, if any. */
let renaming: string | null = null;
/** The add-in, once adsk has appeared (Fusion mode). */
let bridge: FusionBridge | null = null;
let fusionStatus: FusionStatus | null = null;
/** Why the page can't insert, for the person; null when it can. */
let fusionProblem: string | null = null;
const marks = new InsertLog(store);

/** here.ts and what it imports (the core client, the editor): loaded on the first photo. */
const here = () => import("./here");
// For the end-to-end tests and a look from the console: desktop.engine() loads
// the measuring code (here.ts, and the engine worker when asked to load or measure)
// and returns its engine; desktop.timings() each measurement's time.
Object.assign(window, {
  desktop: {
    engine: async () => (await here()).engine(SITE_ROOT),
    timings: async () => (await here()).timings,
  },
});

if (FUSION) {
  document.body.classList.add("fusion");
  $("pick-label").hidden = true;
  // The add-in pushes how each insert ended (palette.sendInfoToHTML). Fusion wants a string back.
  Object.assign(window, { fusionJavaScriptHandler: { handle: (a: string, d: string) => bridge?.handle(a, d) ?? "OK" } });
}
$("storage-note").hidden = persistent;
$("relay-note").hidden = relayUrl !== null;
$("build").textContent = `${APP} · build ${BUILD}`;
$("relay-name").textContent = relayUrl ? `relay ${new URL(relayUrl).host}` : "no relay";

await connect();
selection.keepIn(ids());
renderAll();
if (FUSION) void startFusion();

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
  inbox = new Inbox(store, { send: (m) => link?.send(m) ?? Promise.resolve(false) }, { arrived, removed: removedOnPhone, edited: editedOnPhone });
}

function onStatus(s: LinkStatus): void {
  const before = linkStatus;
  linkStatus = s;
  if (s.state !== "open" || s.peers < 1) phoneHere = null;
  if (s.state === "waiting") retryAt = Date.now() + (s.retryInMs ?? 0);
  // Say hello whenever a phone may have appeared: its outbox sends what it holds on hearing it.
  if (s.state === "open" && (before.state !== "open" || s.peers > before.peers)) {
    void link?.send({ type: "hello", role: "desktop", device: FUSION ? "Fusion" : deviceName(navigator.userAgent), app: `${APP} (${BUILD})` });
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
  // A phone that says all it holds hears all this computer holds: stars set and curves deleted here while it was away.
  if (m.type === "state" && m.full) void link?.send(inbox.fullState());
}

function arrived(curve: Curve, updated: boolean): void {
  if (!phoneHere) phoneHere = { device: "" };
  rememberPhone(phoneHere.device);
  if (updated) return changedOnPhone(curve);
  // A photo being measured here is the person's own pick: a curve arriving does not take its place.
  const shown = !showingDraft && selection.arrived(curve.id, Date.now());
  unseen.arrived(document.hidden);
  document.title = unseen.title;
  flashId = curve.id;
  renderList();
  renderPairing();
  if (shown) renderMain();
  renderStatus();
  marks.keepOnly(ids());
  if (FUSION && bridge && autoInsert(store)) {
    void insertCurve(curve, "auto");
    return;
  }
  toast(shown ? `New curve: ${title(curve)}` : `New curve: ${title(curve)}, at the top of the list`);
}

/** A newer revision of a curve in the list: the person changed it on the phone (a pin, the side). */
function changedOnPhone(curve: Curve): void {
  flashId = curve.id;
  renderList();
  if (current()?.id === curve.id) renderMain();
  if (FUSION) {
    const mark = marks.get(curve.id);
    if (mark?.state === "inserted") {
      // The sketch keeps what was inserted; this revision is not in it.
      marks.forget(curve.id);
      renderMark(curve.id);
      toast(`Changed on the phone: ${title(curve)}. ${mark.sketch} has the earlier curve; Insert puts this one in.`, 6000);
      return;
    }
    if (bridge && autoInsert(store) && mark?.state !== "inserting") {
      void insertCurve(curve, "auto");
      return;
    }
  }
  toast(`Changed on the phone: ${title(curve)}`);
}

/** The phone deleted these. */
function removedOnPhone(curves: Curve[]): void {
  marks.keepOnly(ids());
  selection.keepIn(ids());
  renderAll();
  // Their names went with them (the inbox forgets a deleted curve's name): the photos' names.
  toast(curves.length === 1 ? `Deleted on the phone: ${curves[0]!.photo}` : `${curves.length} curves deleted on the phone`);
}

function editedOnPhone(): void {
  renderList();
  if (!renaming) renderMain();
}

/** What a curve is called: the person's name for it, else its photo's. */
function title(c: Curve): string {
  return inbox.nameOf(c.id) || c.photo;
}

function startRename(): void {
  const c = current();
  if (!c) return;
  renaming = c.id;
  renderMain();
  const input = document.querySelector<HTMLInputElement>("#rename-input");
  input?.focus();
  input?.select();
}

/** Enter or leaving the field keeps the name; Escape does not. An empty name gives the photo's back.
 * The heading and the list's entry change in place, and when a press elsewhere
 * ended the edit, only once it is released: the heading's row reflows, and the
 * press's click must still reach what was pressed. */
function endRename(keep: boolean): void {
  const id = renaming;
  const input = document.querySelector<HTMLInputElement>("#rename-input");
  const c = inbox.curves.find((x) => x.id === id);
  renaming = null;
  if (!id || !c || !input) return;
  if (keep) inbox.setName(id, input.value);
  const name = $("list").querySelector(`[data-id="${CSS.escape(id)}"] .name`);
  if (name) name.textContent = title(c);
  if (!pressing) return showName(c, input);
  // On release, or soon at the latest (a press the browser took over, to scroll, never ends in a pointerup).
  const show = () => {
    clearTimeout(late);
    removeEventListener("pointerup", soon);
    showName(c, input);
  };
  const soon = () => setTimeout(show);
  const late = setTimeout(show, 1500);
  addEventListener("pointerup", soon, { once: true });
}

/** Whether a mouse button or finger is down. */
let pressing = false;
addEventListener("pointerdown", () => (pressing = true), true);
addEventListener("pointerup", () => (pressing = false), true);

/** The heading again in place of the name's field. */
function showName(c: Curve, input: HTMLInputElement): void {
  if (!input.isConnected) return;                // the view was drawn again meanwhile
  input.insertAdjacentHTML("afterend", titleHtml(c));
  input.remove();
  const head = document.querySelector(".view .head");
  head?.querySelector<HTMLElement>('[data-act="rename"]')?.style.removeProperty("visibility");
  const meta = head?.querySelector(".meta");
  if (meta) meta.innerHTML = metaHtml(c);
  document.querySelector(".view")?.setAttribute("aria-label", title(c));
  void fillDownloads(c);                      // their names follow the curve's
  if (document.activeElement === document.body) head?.querySelector<HTMLElement>('[data-act="rename"]')?.focus();
}

function titleHtml(c: Curve): string {
  return `<h2 id="curve-title">${esc(title(c))}</h2>`;
}

/** Hidden, not left out, while the name is edited: the buttons beside it stay put for a click that ends the edit. */
function renameButtonHtml(c: Curve, hidden = false): string {
  return `<button type="button" class="btn plain small" data-act="rename"${hidden ? ` style="visibility: hidden"` : ""}
    title="${inbox.isLocal(c.id) ? "Renamed here" : "Renamed here and on the phone"}">Rename</button>`;
}

function metaHtml(c: Curve): string {
  const t = Date.parse(c.measured_at);
  return `${inbox.isLocal(c.id) ? "measured here" : "measured"} <time datetime="${esc(c.measured_at)}" title="${esc(exactTime(t))}">${esc(relativeTime(t, Date.now()))}</time> · ${esc(c.chain.name)}${inbox.nameOf(c.id) ? ` · ${esc(c.photo)}` : ""}`;
}

function toggleStar(): void {
  const c = current();
  if (!c) return;
  inbox.setStar(c.id, !inbox.isStarred(c.id));
  renderList();
  renderMain();
}

// ---- Fusion (?fusion) -----------------------------------------------------

/** Wait for adsk, then keep the add-in's status (the sketch being edited) fresh. */
async function startFusion(): Promise<void> {
  const send = await waitForSend(window as { adsk?: { fusionSendData?: unknown } }, 5000);
  if (!send) {
    fusionProblem = "This is the page for Fusion's Spline Wire palette. Open it there: Add from phone, in the sketch toolbar.";
    renderFusion();
    return;
  }
  bridge = new FusionBridge(send, onInserted);
  await refreshStatus();
  renderMain();
  setInterval(() => void (!document.hidden && refreshStatus()), 2000);
}

async function refreshStatus(): Promise<void> {
  if (!bridge) return;
  try {
    const s = await bridge.status();
    const changed = s.sketch !== fusionStatus?.sketch || fusionProblem !== null;
    fusionStatus = s;
    fusionProblem = null;
    if (changed) renderFusion();
  } catch (e) {
    fusionStatus = null;
    fusionProblem = (e as Error).message;
    renderFusion();
  }
}

async function insertCurve(c: Curve, how: How): Promise<void> {
  if (!bridge) {
    toast(fusionProblem ?? "Fusion isn't answering.");
    return;
  }
  marks.set(c.id, { state: "inserting" });
  renderMark(c.id);
  try {
    // A new sketch is named after the photo with its extension cut off (palette.py
    // sketch_name): a person's name gets a final dot, so all of it is kept.
    const named = inbox.nameOf(c.id);
    const s = shapedOf(c);
    if (s.corners.length && fusionStatus && !addinHasCorners(fusionStatus.addin)) {
      toast("This add-in puts in one smooth spline, without the sharp corners: download it again from the Spline Wire website.", 7000);
    }
    const shaped = { ...c, points_mm: s.points, corners: s.corners };
    if (await bridge.insert(named ? { ...shaped, photo: `${named}.` } : shaped, how) === "no-sketch") {
      marks.set(c.id, { state: "waiting" });
      toast(`New curve: ${title(c)}. Open a sketch and press Insert to put it there.`, 5000);
    }
  } catch (e) {
    const error = (e as Error).message;
    marks.set(c.id, { state: "failed", error, cancelled: false });
    toast(`Not inserted: ${error}`, 6000);
  }
  renderMark(c.id);
}

function onInserted(r: Inserted): void {
  marks.set(r.id, r.ok ? { state: "inserted", sketch: r.sketch, at: new Date().toISOString() }
    : { state: "failed", error: r.error, cancelled: r.cancelled });
  renderMark(r.id);
  if (r.ok) toast(`${plural(r.points, "point")} into ${r.sketch}. ${UNDO} undoes it.`, 5000);
  else if (!r.cancelled) toast(`Not inserted: ${r.error}`, 6000);
  void refreshStatus();
}

function renderMark(id: string): void {
  renderList();
  if (current()?.id === id) renderMain();
}

function renderFusion(): void {
  if (!FUSION) return;
  $("fusion-note").hidden = !fusionProblem;
  $("fusion-note").textContent = fusionProblem ?? "";
  $("fusion-bar").hidden = !bridge;
  const sketch = fusionStatus?.sketch ?? null;
  $("fusion-sketch").textContent = sketch ? `Editing ${sketch}` : "No sketch open: new curves wait here";
  $<HTMLInputElement>("auto-insert").checked = autoInsert(store);
  const button = document.querySelector<HTMLButtonElement>('[data-act="insert"]');
  if (button) {
    button.disabled = !bridge;
    button.querySelector("span")!.textContent = insertLabel();
  }
  const hint = $("insert-hint");
  if (hint && !markText(marks.get(current()?.id ?? ""))) hint.textContent = insertHint();
  $("addin-name").textContent = fusionStatus ? `add-in ${fusionStatus.addin}` : "";
}

function insertLabel(): string {
  return fusionStatus?.sketch ? `Insert into ${fusionStatus.sketch}` : "Insert into a new sketch…";
}

function insertHint(): string {
  return fusionStatus?.sketch ? `One undo step (${UNDO}).` : "Fusion asks for a plane or face (XY if you pick none).";
}

function fusionOutHtml(c: Curve): string {
  const said = markText(marks.get(c.id));
  return `<div class="out">
      <button type="button" class="copy" data-act="insert"${bridge ? "" : " disabled"}><span>${esc(insertLabel())}</span></button>
      <div class="then" id="insert-hint">${esc(said || insertHint())}</div>
    </div>`;
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

/** The curve on show; null when there is none, or the draft is on show. */
function current(): Curve | null {
  if (showingDraft) return null;
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
  const draftItem = draft ? `<li><button type="button" class="item draft" data-draft aria-current="${showingDraft}">
      <span class="thumb" aria-hidden="true">…</span><span class="row1"><span class="name">${esc(draft.name)}</span><span class="tag-here">not added</span></span>
      <span class="sub">${esc(draft.state)}</span></button></li>` : "";
  list.innerHTML = draftItem + inbox.curves.map((c) => {
    const t = Date.parse(c.measured_at);
    const w = c.warnings.length;
    const mine = inbox.isLocal(c.id);
    return `<li><button type="button" class="item${c.id === flashId ? " flash" : ""}" data-id="${esc(c.id)}"
      aria-current="${!showingDraft && c.id === selection.id}">
      <span class="thumb" aria-hidden="true">${curveDrawing({ pins: c.pins_mm, spline: c.points_mm }, 1, { pad: 8 })}${w ? `<span class="dot"></span>` : ""}</span>
      <span class="row1">${inbox.isStarred(c.id) ? `<span class="starred" title="Starred">★</span>` : ""}<span class="name">${esc(title(c))}</span>${mine ? `<span class="tag-here">measured here</span>` : ""}${FUSION ? markTag(c.id) : ""}${w ? `<span class="warns">${plural(w, "warning")}</span>` : ""}</span>
      <span class="sub"><time datetime="${esc(c.measured_at)}" title="${esc(exactTime(t))}">${esc(relativeTime(t, now))}</time>
        · ${c.chain.n_pins} pins · ${esc(c.chain.name)}</span>
    </button></li>`;
  }).join("");
  if (flashId) {
    const id = flashId;
    flashId = null;
    setTimeout(() => list.querySelector(`[data-id="${CSS.escape(id)}"]`)?.classList.remove("flash"), 1800);
  }
  $("count").textContent = inbox.curves.length ? String(inbox.curves.length) : "";
  $("none").hidden = inbox.curves.length > 0 || !!draft;
  if (focused) list.querySelector<HTMLElement>(`[data-id="${CSS.escape(focused)}"]`)?.focus();
}

/** In Fusion: where the curve went, in a word or two. */
function markTag(id: string): string {
  const m = marks.get(id);
  if (!m) return "";
  const [cls, text] = m.state === "inserted" ? ["tag-in", m.sketch]
    : m.state === "inserting" ? ["tag-in", "inserting…"]
    : ["tag-wait", m.state === "waiting" ? "waiting" : "not inserted"];
  return `<span class="${cls}" title="${esc(markText(m))}">${esc(text)}</span>`;
}

function renderPairing(): void {
  const side = $("pair-side");
  const url = pairingUrl(phonePageUrl(SITE_ROOT, location.search), secret);
  const phone = lastPhone();
  if (!paired() && inbox.curves.length === 0 && !showingDraft) {
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
      <div class="row">${FUSION ? "" : `<button type="button" class="btn small" data-act="copy-link">Copy link</button>`}
        ${paired() ? `<button type="button" class="btn small plain" data-act="unpair">Unpair…</button>` : ""}</div>
    </div>
  </details>`;
}

function renderMain(): void {
  const main = $("main");
  for (const u of blobUrls) URL.revokeObjectURL(u);
  blobUrls = [];
  if (showingDraft && draft) {
    main.classList.remove("empty");
    main.replaceChildren(draft.root);
    draft.shown();
    return;
  }
  const c = current();
  main.classList.toggle("empty", !c);
  if (!c) {
    main.innerHTML = paired() ? waitingHtml() : welcomeHtml();
    return;
  }
  main.innerHTML = curveHtml(c);
  if (!FUSION) void fillDownloads(c);
}

function welcomeHtml(): string {
  const link = pairingUrl(phonePageUrl(SITE_ROOT, location.search), secret);
  if (FUSION) {
    return `<section class="welcome" aria-labelledby="welcome-title">
    ${qrSvg(link, "QR code: the phone page, paired with Fusion")}
    <div><h2 id="welcome-title">Pair your phone</h2>
      <ol><li>Scan this code with the phone's Camera app and open the page.</li>
        <li>Open a sketch here, and photograph the chain on the phone.</li>
        <li>The curve goes straight into the sketch, as one undo step.</li></ol>
      <p class="then">Done once per phone. A phone pairs with one place at a time: pairing it here replaces its pairing with the Spline Wire website on a computer.</p>
      <div class="url" data-pair-link>${esc(link)}</div>
    </div>
  </section>`;
  }
  return `<section class="welcome" aria-labelledby="welcome-title">
    ${qrSvg(link, "QR code: the phone page, paired with this computer")}
    <div><h2 id="welcome-title">Pair your phone</h2>
      <ol><li>Scan this code with the phone's Camera app and open the page.</li>
        <li>Photograph the chain there. The phone measures it.</li>
        <li>The curve shows up here, ready for <b>Copy points</b> and Fusion's <b>Paste points</b>.</li></ol>
      <p class="then">Done once per phone: both remember the pairing. The phone and this computer can be on different networks.</p>
      <div class="url" data-pair-link>${esc(link)}</div>
      <button type="button" class="btn small" data-act="copy-link">Copy link</button>
      ${measureHereHtml()}
    </div>
  </section>`;
}

function waitingHtml(): string {
  if (FUSION) {
    return `<section class="welcome waiting" aria-labelledby="welcome-title">
    <div><h2 id="welcome-title">Waiting for a curve</h2>
      <p>Open a sketch, then measure the chain on the paired phone: the curve goes straight into the sketch.</p>
      <p class="then">With no sketch open, curves wait here for Insert. If the phone measures while Fusion is closed, it keeps the curve and sends it when the palette is open again.</p></div>
  </section>`;
  }
  return `<section class="welcome waiting" aria-labelledby="welcome-title">
    <div><h2 id="welcome-title">Waiting for a curve</h2>
      <p>Measure the chain on the paired phone. Each curve shows up here within seconds, and stays on this computer until you delete it.</p>
      <p class="then">If the phone measures while this page is closed, it keeps the curve and sends it when the page is open again.</p>
      ${measureHereHtml()}</div>
  </section>`;
}

function measureHereHtml(): string {
  return `<p class="then here-offer">Have the photo on this computer? Drop it on this page, or
    <button type="button" class="btn small" data-act="pick">Choose a photo…</button>, to measure it here.</p>`;
}

/** The curve with the corners marked on it made sharp (corners.ts). */
function shapedOf(c: Curve): Shaped {
  return shape(c.points_mm, c.pins_mm, c.chain.half_width_mm, inbox.cornersOf(c.id));
}

function curveHtml(c: Curve): string {
  const t = Date.parse(c.measured_at);
  const mine = inbox.isLocal(c.id);
  const d = c.diagnostics;
  const s = shapedOf(c);
  const splines = pieces(s).map((p) => splineSamples(p));
  const spline = splines.flat();
  const xs = [...spline, ...c.pins_mm].map((p) => p[0]);
  const ys = [...spline, ...c.pins_mm].map((p) => p[1]);
  const size = `${Math.round(Math.max(...xs) - Math.min(...xs))} × ${Math.round(Math.max(...ys) - Math.min(...ys))} mm`;
  const sharp = s.corners.length ? `, ${plural(s.corners.length, "sharp corner")}` : "";
  const plot = plotSvg(s.points, c.pins_mm, spline, `The measured curve, ${plural(s.points.length, "point")}${sharp}, ${size}, on a millimetre grid`, {
    pieces: splines, targets: c.points_mm, replaced: s.replaced,
    corners: s.corners.map((i, j) => ({ at: s.points[i]!, source: s.zones[j]!.source })),
  });
  const found = c.chain.n_pins - d.missing_pins;
  const cells: [string, string, string, boolean][] = [
    ["Points", `${s.points.length}`, s.corners.length ? `${plural(s.corners.length, "sharp corner")}` : "on the object's edge", false],
    ["Pins", `${found}<small>/${c.chain.n_pins}</small>`, d.missing_pins ? `${d.missing_pins} missing` : "all found", d.missing_pins > 0],
    ["Tilt", `${d.tilt_deg.toFixed(1)}<small>°</small>`, "camera to table", false],
    ["Link error", d.link_residual_rms_mm.toFixed(3), `rms · max ${d.link_residual_max_mm.toFixed(3)} mm`,
      // Python warns above 5% of the pitch (pipeline.py).
      d.link_residual_max_mm > 0.05 * c.chain.pitch_mm],
    ["Focal length", `${+d.focal_35mm.toFixed(1)}<small>mm</small>`, `35 mm equiv. · ${(mine ? FOCAL_SOURCE_HERE : FOCAL_SOURCE)[d.focal_source] ?? d.focal_source}`,
      d.focal_source === "estimated"],
    ["Object", c.object_side === "inside" ? "Inside" : "Outside",
      c.object_side === "inside" ? "chain wrapped around it" : "chain pressed into it", false],
  ];
  const notes = [...c.warnings.map((w) => readWarning(w, mine)), ...c.notes.map(readNote)]
    .map((r) => `<div class="note ${r.level}"><b>${esc(r.title)}</b>${esc(r.text)}</div>`).join("");
  return `<section class="view" aria-label="${esc(title(c))}">
    <div class="head">${renaming === c.id
      ? `<input id="rename-input" class="rename" value="${esc(title(c))}" maxlength="100" aria-label="Name of the curve" placeholder="${esc(c.photo)}">`
      : titleHtml(c)}
      ${c.warnings.length ? `<span class="tag chk" title="${plural(c.warnings.length, "warning")}">CHECK</span>` : `<span class="tag ok">OK</span>`}
      <span class="meta">${metaHtml(c)}</span>
      <span class="spacer"></span>
      <button type="button" class="btn plain small star${inbox.isStarred(c.id) ? " on" : ""}" data-act="star" aria-pressed="${inbox.isStarred(c.id)}"
        title="${mine ? "Starred here" : "Starred here and on the phone"}"><span class="sg">${inbox.isStarred(c.id) ? "★" : "☆"}</span> ${inbox.isStarred(c.id) ? "Starred" : "Star"}</button>
      ${renameButtonHtml(c, renaming === c.id)}
      <button type="button" class="btn plain small" data-act="delete">Delete…</button></div>
    <div class="plot">${plot}</div>
    <div class="legend" aria-hidden="true">
      <span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="5" fill="none" stroke="var(--plot-ink)" stroke-opacity=".6"/></svg>Pin of the chain</span>
      <span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="3.5" fill="var(--curve)"/></svg>Point on the object's edge</span>
      <span><svg viewBox="0 0 14 14"><line x1="1" y1="7" x2="13" y2="7" stroke="var(--curve)" stroke-width="2.5"/></svg>Spline through the points</span>
      <span><svg viewBox="0 0 14 14"><rect x="3.5" y="3.5" width="7" height="7" transform="rotate(45 7 7)" fill="var(--ink)"/></svg>Sharp corner</span>
    </div>
    <p class="corner-hint">${CORNER_HINT}</p>
  </section>
  <aside class="read" aria-label="Measurement">
    <div class="sec"><span>Curve · mm</span><span class="mono">${esc(size)}</span></div>
    <div class="scroll">
      <div class="nums">${cells.map(([k, v, s, warn]) =>
        `<div class="cell${warn ? " warn" : ""}"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${esc(s)}</div></div>`).join("")}</div>
      ${notes ? `<div class="notes">${notes}</div>` : ""}
      <details class="about"><summary>About this curve</summary>
        <div class="kv"><span>Measured</span><span>${esc(exactTime(t))}${mine ? ", on this computer" : ""}</span></div>
        <div class="kv"><span>Photo</span><span>${esc(c.photo)}</span></div>
        <div class="kv"><span>Chain</span><span>${esc(c.chain.name)}</span></div>
        <div class="kv"><span>Link length</span><span>${c.chain.pitch_mm} mm</span></div>
        <div class="kv"><span>Pin to edge</span><span>${c.chain.half_width_mm} mm</span></div>
        <div class="kv"><span>Id</span><span>${esc(c.id)}</span></div>
      </details>
    </div>
    ${FUSION ? fusionOutHtml(c) : `<div class="out">
      <button type="button" class="copy" data-act="copy"><span>Copy points</span><kbd>${IS_MAC ? "⌘ C" : "Ctrl C"}</kbd></button>
      <div class="then">then <b>Paste points</b> in a Fusion sketch</div>
      ${ADDIN_HELP}
      <div class="exp" id="exp">Save as …</div>
    </div>`}
  </aside>`;
}

/** The downloads: the phone's files unpacked, and the clipboard table. With
 * sharp corners, the files of points are made here from the curve with its
 * corners; the SVG and the Fusion CSV have one smooth spline, so they go. */
async function fillDownloads(c: Curve): Promise<void> {
  const texts: Partial<Record<DownloadKind, string>> = c.files ? await unpack(c.files) : {};
  const s = shapedOf(c);
  texts.tsv = pointsTsv(s.points, s.corners);
  if (s.corners.length) {
    texts.dxf = dxfText(s.points, s.corners);
    texts.csv = pointsTsv(s.points, s.corners).replaceAll("\t", ",").replaceAll("\n", "\r\n");
    delete texts.svg;
    delete texts.fusion_csv;
  }
  if (selection.id !== c.id || !$("exp")) return;      // another curve is on show by now
  const links = DOWNLOADS.filter((d) => texts[d.kind] !== undefined).map((d) => {
    const url = URL.createObjectURL(new Blob([texts[d.kind]!], { type: d.type }));
    blobUrls.push(url);
    const name = downloadName(d.kind, c.photo, inbox.nameOf(c.id));
    return `<a class="btn" href="${url}" download="${esc(name)}" data-kind="${d.kind}" title="${esc(name)}">${d.label}</a>`;
  });
  const missing = c.files ? (s.corners.length ? `<span>With sharp corners: no SVG or Fusion CSV, which hold one smooth spline.</span>` : "")
    : `<span>The phone sent this curve without its files (too long for one message): only ${s.corners.length ? "the files of points" : "the points"} can be saved.</span>`;
  $("exp").innerHTML = `Save as ${links.join("")}${missing}`;
}

// ---- actions --------------------------------------------------------------

function pick(id: string, focus = false): void {
  const changed = selection.id !== id || showingDraft;
  showingDraft = false;
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
  const s = shapedOf(c);
  const text = pointsTsv(s.points, s.corners);
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
  const sharp = s.corners.length ? ` and ${plural(s.corners.length, "sharp corner")}` : "";
  toast(ok ? `Copied ${plural(s.points.length, "point")}${sharp}. In Fusion, click Paste points while editing a sketch.`
    : "Couldn't copy. Use Save as TSV instead.");
}

function clipboardOf(c: Curve): string {
  const s = shapedOf(c);
  return pointsTsv(s.points, s.corners);
}

/** A click on a point of the drawing: a sharp corner there, or a smooth curve again. */
function toggleCorner(i: number): void {
  const c = current();
  if (!c || !Number.isInteger(i)) return;
  const marks = inbox.cornersOf(c.id);
  const before = shapedOf(c).corners.length;
  inbox.setCorners(c.id, toggleMark(c.points_mm, shapedOf(c), marks, i));
  renderMain();
  const after = shapedOf(c).corners.length;
  if (after !== before) toast(after > before ? "Sharp corner. Click it again to make the curve smooth there." : "Smooth again there.");
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

// ---- measuring here (D9) ---------------------------------------------------

/** Measure a photo on this computer. One at a time: a new photo replaces one not yet added. */
async function measureHere(files: File[]): Promise<void> {
  const file = files[0];
  if (!file) return;
  let mod: typeof import("./here");
  try {
    mod = await here();
  } catch (e) {
    toast(`Couldn't load the measuring code: ${(e as Error).message}`);
    return;
  }
  if (files.length > 1) toast(`One photo at a time: measuring ${file.name}.`);
  else if (draft) toast(`${draft.name} was not added; measuring ${file.name} instead.`);
  draft?.close();
  draft = mod.openDraft(file, draftHost);
  showingDraft = true;
  renderList();
  renderPairing();
  renderMain();
}

const draftHost: DraftHost = {
  siteRoot: SITE_ROOT,
  store,
  toast: (m) => toast(m),
  add(curve) {
    draft = null;
    showingDraft = false;
    inbox.addLocal(curve);
    selection.pick(curve.id, Date.now());
    flashId = curve.id;
    renderList();
    renderPairing();
    renderMain();
    toast(`Added ${curve.photo}. Copy points, then Paste points in a Fusion sketch.`);
  },
  discard() {
    draft = null;
    showingDraft = false;
    selection.keepIn(ids());
    renderList();
    renderPairing();
    renderMain();
  },
  changed() {
    renderList();
  },
};

function showDraft(): void {
  if (!draft || showingDraft) return;
  showingDraft = true;
  renderList();
  renderMain();
}

async function remove(): Promise<void> {
  const c = current();
  if (!c) return;
  const sure = await confirmBox(`Delete ${title(c)}?`, inbox.isLocal(c.id)
    ? "Its points and files go from this computer. Save them first if you need them."
    : "Its points and files go from this computer, and the photo goes from the phone too (when the phone's page next opens, if it is closed). Save them first if you need them.",
    "Delete");
  if (!sure) return;
  const at = ids().indexOf(c.id);
  inbox.remove(c.id);
  marks.keepOnly(ids());
  selection.keepIn(ids(), at);
  renderList();
  renderPairing();
  renderMain();
  const next = selection.id && $("list").querySelector<HTMLElement>(`[data-id="${CSS.escape(selection.id)}"]`);
  (next || $("main")).focus();
  toast(`Deleted ${title(c)}`);
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
  const t = (ev.target as Element).closest<HTMLElement>("[data-id], [data-act], [data-draft]");
  if (!t) return;
  if (t.dataset.id) return pick(t.dataset.id);
  if (t.hasAttribute("data-draft")) return showDraft();
  const act = t.dataset.act;
  if (act === "copy") void copyPoints();
  else if (act === "copy-link") void copyLink();
  else if (act === "delete") void remove();
  else if (act === "star") toggleStar();
  else if (act === "rename") startRename();
  else if (act === "unpair") void unpair();
  else if (act === "pick") $("pick").click();
  else if (act === "insert" && current()) void insertCurve(current()!, "ask");
  else if (act === "corner") toggleCorner(Number(t.dataset.i));
});

$<HTMLInputElement>("pick").addEventListener("change", (ev) => {
  const input = ev.target as HTMLInputElement;
  const files = [...(input.files ?? [])];
  input.value = "";                         // the same photo can be picked again
  void measureHere(files);
});

// A photo dragged anywhere onto the page is measured here. Only drags that
// carry files count, and the browser must not open the photo in the tab.
let dragDepth = 0;
const carriesFiles = (ev: DragEvent) => !!ev.dataTransfer && [...ev.dataTransfer.types].includes("Files");
document.addEventListener("dragenter", (ev) => {
  if (!carriesFiles(ev) || FUSION) return;
  ev.preventDefault();
  dragDepth++;
  $("drop-hint").hidden = false;
});
document.addEventListener("dragover", (ev) => {
  if (!carriesFiles(ev)) return;
  ev.preventDefault();
  ev.dataTransfer!.dropEffect = FUSION ? "none" : "copy";
});
document.addEventListener("dragleave", (ev) => {
  if (!carriesFiles(ev) || FUSION) return;
  if (--dragDepth <= 0) {
    dragDepth = 0;
    $("drop-hint").hidden = true;
  }
});
document.addEventListener("drop", (ev) => {
  if (!carriesFiles(ev)) return;
  ev.preventDefault();
  dragDepth = 0;
  $("drop-hint").hidden = true;
  if (FUSION) toast("Measure photos on the phone; the Spline Wire website measures them on a computer.");
  else void measureHere([...ev.dataTransfer!.files]);
});
window.addEventListener("resize", () => draft?.refit());

$("retry").addEventListener("click", () => link?.retryNow());
$<HTMLInputElement>("auto-insert").addEventListener("change", (ev) => {
  store.set(AUTO_KEY, (ev.target as HTMLInputElement).checked ? "on" : "off");
  renderFusion();
});

// The name being edited: Enter keeps it, Escape does not, leaving the field keeps it.
document.addEventListener("keydown", (ev) => {
  if ((ev.target as HTMLElement).id !== "rename-input") return;
  if (ev.key === "Enter" || ev.key === "Escape") {
    ev.preventDefault();
    endRename(ev.key === "Enter");
  }
});
document.addEventListener("focusout", (ev) => {
  if ((ev.target as HTMLElement).id === "rename-input" && renaming) endRename(true);
});

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
  if (!FUSION && (ev.ctrlKey || ev.metaKey) && !ev.altKey && !ev.shiftKey && ev.key.toLowerCase() === "c" && nothingSelected() && current()) {
    ev.preventDefault();
    void copyPoints();
  }
});
document.addEventListener("copy", (ev) => {
  // Edit > Copy with nothing selected, and copyPoints' fallback.
  const text = copying ?? (nothingSelected() ? (current() && clipboardOf(current()!)) : null);
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
    el.textContent = relativeTime(t, now);
  }
}, 30_000);

// Another tab of this page unpaired, received or deleted: follow it, so the
// two never disagree and neither writes back a list the other has changed.
window.addEventListener("storage", (ev) => {
  if (!ev.key?.startsWith(PREFIX)) return;
  const key = ev.key.slice(PREFIX.length);
  if (key === "pairing") {
    // Only a new secret: between the other tab's forget and its new one there is none,
    // and making one here would start a second pairing.
    const now = storedSecret(store);
    if (!now || now === secret) return;
    secret = now;
    void connect().then(renderAll);
    return;
  }
  if (key !== "inbox" && key !== "inbox-seen" && key !== "inbox-here" && key !== PHONE_KEY) return;
  const shown = selection.id;
  loadInbox();
  selection.keepIn(ids());
  renderList();
  renderPairing();
  if (selection.id !== shown || !current()) renderMain();
});

