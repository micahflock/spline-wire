// The phone page (web-build.md, step 2): take or pick a photo, measure it in
// the core worker, fix the pins by touch, then copy, share, download or send
// the curve. The decisions live in the modules beside this one (photos,
// measurer, settings, send, wording), in src/editor/ and src/photo/ (the
// photo editor, shared with the desktop page); this file wires them to the page.

import "./phone.css";
import "../photo/photo.css";
import type { CameraInfo } from "../camera/choose";
import { fovRatio, stillSearchBox } from "../camera/focal";
import { CoreClient, type PhotoSource } from "../core/client";
import { makeCoreWorker } from "../core/make-worker";
import { CoreError, type Commands, type EnginePreviews, type Progress } from "../core/protocol";
import type { ChainSpec, Edits, MeasureOptions, MeasurementDocument, OutputKind, Side } from "../core/types";
import { LiveAnalyser } from "../pipeline/live-client";
import { EditHistory, editsAfterTap } from "../editor/edits";
import type { Pt } from "../editor/viewport";
import { greyCopy, previews } from "../photo/preview";
import { PhotoEditor } from "../photo/view";
import { pageRelaySetting, relayAddress } from "../relay/config";
import { deriveKeys } from "../relay/crypto";
import { deviceName } from "../relay/device";
import { pack } from "../relay/files";
import { RelayLink, type LinkStatus } from "../relay/link";
import { pointsTsv, type Message } from "../relay/messages";
import { DeletedIds } from "../relay/sync";
import { Outbox } from "../relay/outbox";
import { codeMeaning } from "../camera/qr";
import { forgetPairing, phoneSecret, secretFromHash } from "../relay/pairing";
import { LocalStore } from "../relay/store";
import { CameraView, type Still } from "./camera-view";
import { curveFrom } from "./curve";
import { cardCrop, cardSvg, clampBox, curveDrawing, fitBox, stripSvg, type Box } from "./drawings";
import { TapFilter } from "./taps";
import { NO_EXIF, exifOfFile, type Exif } from "./exif";
import { cameraLines, isFront, learn, lensFocal, lensName, mainLens, savedFocal } from "./lenses";
import { Measurer, type Update } from "./measurer";
import {
  KEEP_CURVES, PHOTO_FILES, PhotoStore, cleanupPlan, hasCurve, newPhotoId, type BlobKind, type PhotoFocal, type PhotoRecord,
} from "./photos";
import { applyState, curveKey, delivery, deliveryText, displayName, nameOf, nextRevision, phoneState, sendOffer, setName, setStar, starOf, syncOf } from "./send";
import { renamedFile } from "../desktop/names";
import {
  KEEP_PHOTO_CHOICES, chainLabel, chainSpec, debugFromQuery, debugLeftovers, defaultPins, loadSettings, parseFocal, parsePins, presetKey,
  saveSettings,
  type Settings, type Shade,
} from "./settings";
import {
  ENGINE_STAGE, errorAdvice, errorTitle, failureText, focalStatus, measurementRows, photoFocalText, readNote, readWarning, spaceText,
} from "./wording";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const el = <K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, ...kids: (Node | string)[]) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids);
  return e;
};

const NO_EDITS: Edits = { add: [], remove: [] };
const history_ = window.history;

const store = new LocalStore("splinewire-phone:");
let settings: Settings = loadSettings(store);
// `?debug` turns debug mode on (and `?debug=0` off) and the phone remembers it (settings.ts).
const debugAsked = debugFromQuery(location.search);
if (debugAsked !== null && debugAsked !== settings.debug) {
  settings = { ...settings, debug: debugAsked };
  saveSettings(store, settings);
}
let chains: Record<string, ChainSpec> = {};

const core = new CoreClient(() => makeCoreWorker());
core.onPreviews = (id, p) => void takePreviews(id, p);
const live = new LiveAnalyser(() => new Worker(new URL("../pipeline/live-worker.ts", import.meta.url), { type: "module" }));
const timings: { id: string; status: string; ms: number }[] = [];
// For the end-to-end test, and for looking at a phone's timings and memory
// from its console: phone.core.call("stats", {}).
const debug = { core, timings, engine: null as null | { readyAt: number; ms: number }, link: [] as (LinkStatus & { t: number })[],
  camera: null as null | CameraView["stats"], cameraView: null as null | CameraView };
Object.assign(window, { phone: debug });

// ---- photos ----

const records = new Map<string, PhotoRecord>();
let order: string[] = [];                      // newest first
let photos: PhotoStore | null = null;           // null: IndexedDB unavailable; photos then last for the visit
const memBlobs = new Map<string, Blob>();
const histories = new Map<string, EditHistory>();
/** Photos whose measurement under way carries an edit just committed to the history. */
const editing = new Set<string>();

function focalKey(): string {
  return JSON.stringify([savedFocal(settings)?.focal35 ?? null, settings.focalOverride35]);
}

function options(id: string): MeasureOptions | null {
  const r = records.get(id);
  if (!r) return null;
  // A photo from the live camera: look for the chain where the live view last saw it first (much faster on 12 MP).
  const search = r.camera && r.pixels ? stillSearchBox(r.camera.frameChain, r.camera.frameSize, r.pixels) : null;
  return {
    chain: chainSpec(chains, r.chain, r.pins, r.shade), side: r.side, edits: r.edits,
    saved_focal_35mm: savedFocal(settings)?.focal35 ?? null, focal_override_35mm: settings.focalOverride35 ?? r.focal?.focal35 ?? null,
    ...(search ? { search_px: search } : {}),
  };
}

function source(id: string): PhotoSource {
  return {
    name: records.get(id)?.name ?? "photo.jpg",
    bytes: async () => {
      const blob = await getBlob(id, "original");
      if (!blob) throw new CoreError({ kind: "person", message: "this photo is no longer on the phone" });
      return blob.arrayBuffer();
    },
  };
}

async function getBlob(id: string, kind: BlobKind): Promise<Blob | undefined> {
  return memBlobs.get(`${id}:${kind}`) ?? (await photos?.blob(id, kind).catch(() => undefined));
}

/** Measure a photo (again), unless its photo was deleted to save space: its curve then stays as it is. */
function remeasure(id: string, urgent = false): void {
  if (records.get(id)?.photoDeleted) return;
  measurer.request(id, urgent);
}

function persist(r: PhotoRecord): void {
  photos?.put(r).catch((e: Error) => toast(`Couldn't save the photo's state: ${e.message}`));
}

const measurer = new Measurer({
  core,
  options,
  source,
  update(id, u) {
    const r = records.get(id);
    if (!r) return;
    onUpdate(r, u);
  },
});

function onUpdate(r: PhotoRecord, u: Update): void {
  r.status = u.status;
  const finished = u.status === "done" || u.status === "error";
  if (finished) {
    if (u.ms !== undefined) timings.push({ id: r.id, status: u.status, ms: Math.round(u.ms) });
    if (u.doc) {
      r.doc = u.doc;
      r.failure = null;
      r.measuredWith = focalKey();
      const pins = u.doc.view?.pins;
      if (r.camera && pins && u.doc.view) {
        const still = [...pins].sort((a, b) => (a.k ?? 0) - (b.k ?? 0)).map((p): [number, number] => [p.x, p.y]);
        r.camera.fov = fovRatio(r.camera.frameChain, r.camera.frameSize, still, { width: u.doc.view.width, height: u.doc.view.height });
      }
    } else if (u.failure) {
      r.failure = u.failure;
      // An edit the core refused: the history and the edits go back to what was last measured.
      if (editing.has(r.id)) {
        histories.get(r.id)?.rollback();
        r.edits = r.doc?.view?.edits ?? NO_EDITS;
      }
      if (u.failure.type === "WorkerError") engineLost();
    }
    editing.delete(r.id);
    if (r.id === selected) pending = null;
    persist(r);
    if (u.doc?.ok) {
      // Its files are kept while the photo is still open in the core, so that the photo can go later.
      void outputsOf(r.id).then(() => tidySoon(), () => {});
      void syncCurve(r);
    }
  }
  renderList();
  if (r.id === selected) renderResult();
}

async function addFiles(files: File[], via: PhotoRecord["via"], still: Still | null = null): Promise<void> {
  for (const [i, file] of files.entries()) {
    const id = newPhotoId(Date.now() + i);
    const [made, exif] = await Promise.all([previews(file, { thumb: true }), exifOfFile(file)]);
    const focal = learnFrom(exif, made?.pixels ?? null, still);
    const r: PhotoRecord = {
      id, name: file.name || "photo.jpg", type: file.type, size: file.size, added: Date.now() + i, via,
      chain: presetKey(chains, settings.chain), pins: defaultPins(settings, chains), shade: settings.shade, side: settings.side, edits: NO_EDITS,
      status: "queued", doc: null, failure: null, pixels: made?.pixels ?? null, preview: made?.previewSize ?? null, autoSend: paired(),
      measuredWith: null, exif, focal,
      camera: still ? { label: still.camera, still: still.kind, frameChain: still.frameChain, frameSize: still.frameSize, fov: null, ms: Math.round(still.ms),
        ...(still.why ? { why: still.why } : {}) } : null,
    };
    const blobs: Partial<Record<BlobKind, Blob>> = { original: file, preview: made?.preview, thumb: made?.thumb };
    let kept = false;
    if (photos) {
      try {
        await photos.add(r, blobs);
        kept = true;
      } catch (e) {
        toast(`Couldn't keep the photo on the phone (${(e as Error).message}). It is measured for this visit only.`);
      }
    }
    if (!kept) for (const [k, b] of Object.entries(blobs)) if (b) memBlobs.set(`${id}:${k}`, b);
    records.set(id, r);
    order.unshift(id);
    if (i === 0) void select(id);
    remeasure(id, i === 0);
  }
  renderList();
  tidySoon();
}

/** The engine's copies of a photo this browser cannot draw (HEIC outside Safari),
 * made when it opened the photo to measure it: kept as the browser's would be. */
async function takePreviews(id: string, p: EnginePreviews): Promise<void> {
  const r = records.get(id);
  if (!r || r.pixels || r.photoDeleted) return;       // the browser drew it, or the photo is gone
  r.pixels = p.pixels;
  r.preview = p.previewSize;
  for (const [kind, blob] of [["preview", p.preview], ["thumb", p.thumb]] as const) {
    if (photos && !memBlobs.has(`${id}:original`)) await photos.putBlob(id, kind, blob).catch(() => memBlobs.set(`${id}:${kind}`, blob));
    else memBlobs.set(`${id}:${kind}`, blob);
  }
  // The lens's focal length is remembered with the photo's size (D17), known only now.
  if (r.exif) learnFrom(r.exif, p.pixels, null);
  persist(r);
  if (selected === id) await select(id);              // the photo under the pins, and on the card
  else renderList();
}

/** D17: remember the lens's focal length from a photo that has it, and find
 * one for a photo that lacks it. Returns the focal length the photo is to be
 * measured with when its file does not say (PhotoRecord.focal). */
function learnFrom(exif: Exif, size: { width: number; height: number } | null, still: Still | null): PhotoFocal | null {
  const before = settings;
  if (size) settings = learn(settings, exif, size, { cameraLabel: still?.camera ?? null, facing: still?.facing ?? null });
  if (settings !== before) {
    saveSettings(store, settings);
    renderSettings();
  }
  const cam = still ? settings.cameras[still.camera] : undefined;
  if (still && cam?.ultraWide && !isFront(exif, still.facing, still.camera)) {
    // S3: a Galaxy S23 Ultra's facingMode opened its ultra-wide. Measured all
    // the same, but the main camera distorts less: open another next time.
    const f = cam.focal35 ? `${cam.focal35} mm` : `${cam.focalMm} mm lens`;
    setSettings({ ...settings, camera: null });
    toast(cameraView.rearCount > 1
      ? `That photo came from the ultra-wide camera (${f}). It is measured, but the camera will open on another back camera next time.`
      : `That photo came from an ultra-wide camera (${f}), which distorts more. It is measured all the same.`, 7000);
  }
  if (still?.kind === "frame") {
    const f = still.frameFocal;
    return f && f.from !== "override" ? { focal35: f.focal35, why: "frame", lens: f.lens } : null;
  }
  const known = lensFocal(settings, exif);
  return known ? { focal35: known.focal35, why: "lens", lens: known.lens } : null;
}

async function deletePhoto(id: string): Promise<void> {
  const r = records.get(id);
  const s = r && room ? syncOf(r, room) : null;
  if (!r || !confirm(s ? `Delete ${titleOf(r)} from this phone and the computer?` : `Delete ${titleOf(r)} from this phone?`)) return;
  closeExpanded();
  if (s) {
    // The computer deletes its copy when it hears this: now if it is open, else when next it is.
    deletedIds.add([s.id]);
    outbox?.forget(s.id);
    sendChange([], [s.id]);
  }
  await removePhoto(id);
}

/** A photo gone from this phone: deleted here or on the computer. */
async function removePhoto(id: string): Promise<void> {
  await photos?.delete(id).catch(() => {});
  drop(id);
  void core.call("close", { photo: id }).catch(() => {});
}

/** The photo whose name is being edited, and where: beside the card's star or the expanded view's. */
let renaming: { id: string; host: HTMLElement } | null = null;

function startRename(r: PhotoRecord, host: HTMLElement): void {
  if (renaming) endRename(true);
  renaming = { id: r.id, host };
  const input = document.createElement("input");
  Object.assign(input, { className: "rename", value: nameOf(r).text, maxLength: 100, placeholder: "Name this curve", enterKeyHint: "done" });
  input.setAttribute("aria-label", "Name of the curve");
  input.onkeydown = (e) => {
    if (e.key === "Enter" || e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();             // Escape ends the rename only, not the expanded view too
      endRename(e.key === "Enter");
    }
  };
  input.onblur = () => endRename(true);
  // The host holds the field now: no longer the name's button.
  for (const a of ["role", "tabindex", "aria-label"]) host.removeAttribute(a);
  host.classList.remove("named");
  host.replaceChildren(input);
  input.focus();
  input.select();
}

/** Enter or leaving the field keeps the name, Escape does not; empty clears it (the curve is
 * then called after its time). The names change in place: a tap that left the field must not lose its target. */
function endRename(keep: boolean): void {
  const r = renaming ? records.get(renaming.id) : undefined;
  const input = renaming?.host.querySelector("input");
  renaming = null;
  if (!r || !input) return;
  if (keep && setName(r, input.value)) {
    persist(r);
    const s = room ? syncOf(r, room) : null;
    if (s) sendChange([s.id]);
    renderList();
  }
  renderResult();
}

/** A name given is renamed by tapping it: the chip beside the star (card or expanded view).
 * Unnamed, they offer nothing (the star names a curve). */
for (const [id, host] of [["r-name", "r-name"], ["x-nm", "x-nm"]] as const) {
  const rename = () => {
    const r = selected ? records.get(selected) : undefined;
    if (r && nameOf(r).text && renaming?.host.id !== host) startRename(r, $(host));
  };
  $(id).onclick = rename;
  $(id).onkeydown = (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target === e.currentTarget) {
      e.preventDefault();
      rename();
    }
  };
}

/** The star (the card's or the expanded view's): starring opens the name field
 * beside it; un-starring clears the name. */
function toggleStar(nameHost: HTMLElement): void {
  const r = selected ? records.get(selected) : undefined;
  if (!r) return;
  const on = !starOf(r).on;
  setStar(r, on);
  if (!on) setName(r, "");
  persist(r);
  const s = room ? syncOf(r, room) : null;
  if (s) sendChange([s.id]);
  renderList();
  renderResult();
  if (on) startRename(r, nameHost);
}
$("star").onclick = () => toggleStar($("r-name"));
$("x-star").onclick = () => toggleStar($("x-nm"));

$("delete").onclick = () => {
  if (selected) void deletePhoto(selected);
};

/** When the photo was taken: the time today, else the day. */
function timeOf(r: PhotoRecord): string {
  const d = new Date(r.added);
  return d.toDateString() === new Date().toDateString()
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hourCycle: "h23" })
    : d.toLocaleDateString([], { month: "short", day: "numeric" });
}

/** What the page calls a curve: the person's name for it, else "Curve 18:42". */
const titleOf = (r: PhotoRecord): string => nameOf(r).text || `Curve ${timeOf(r)}`;

function drop(id: string): void {
  records.delete(id);
  order = order.filter((x) => x !== id);
  measurer.forget(id);
  histories.delete(id);
  for (const k of ["original", "preview", "thumb", "outputs"]) memBlobs.delete(`${id}:${k}`);
  if (selected === id) void select(order[0] ?? null);
  renderList();
}

// ---- keeping the phone's storage small ----

/** Delete a photo's own files and keep its curve (photos.ts): false if it has no curve or its files could not be made. */
async function deletePhotoFiles(id: string): Promise<boolean> {
  const r = records.get(id);
  if (!r || r.photoDeleted || !hasCurve(r)) return false;
  try {
    await outputsOf(id);
  } catch {
    return false;
  }
  if (records.get(id) !== r || measurer.busy(id) || !hasCurve(r)) return false;
  r.photoDeleted = Date.now();
  if (photos && !memBlobs.has(`${id}:original`)) {
    try {
      await photos.deleteFiles(r);
    } catch {
      delete r.photoDeleted;
      return false;
    }
  }
  for (const k of PHOTO_FILES) memBlobs.delete(`${id}:${k}`);
  histories.delete(id);
  void core.call("close", { photo: id }).catch(() => {});
  if (selected === id) void select(id);         // shown again, without its photo
  return true;
}

interface Tidied { photos: number; whole: number; failed: number }
let tidyChain: Promise<unknown> = Promise.resolve();

/** Past the newest `keepPhotos`, photos lose their files and keep their curves (photos.ts cleanupPlan). One at a time. */
function tidy(keepPhotos: number, spare: ReadonlySet<string>): Promise<Tidied> {
  const run = tidyChain.then(async (): Promise<Tidied> => {
    const plan = cleanupPlan(order.map((id) => records.get(id)!), keepPhotos, KEEP_CURVES, spare);
    for (const id of plan.whole) await removePhoto(id);
    let done = 0;
    for (const id of plan.photos) if (await deletePhotoFiles(id)) done++;
    renderList();
    return { photos: done, whole: plan.whole.length, failed: plan.photos.length - done };
  });
  tidyChain = run.catch(() => {});
  return run;
}

let tidyTimer: ReturnType<typeof setTimeout> | undefined;
/** After a photo is added or measured: what Settings, Photos kept says, sparing the photo on screen. */
function tidySoon(): void {
  if (!recordsLoaded) return;
  clearTimeout(tidyTimer);
  tidyTimer = setTimeout(() => void tidy(settings.keepPhotos, new Set(selected ? [selected] : [])).catch(() => {}), 1500);
}

$("free-space").onclick = async () => {
  const all = order.map((id) => records.get(id)!);
  const plan = cleanupPlan(all, 0, KEEP_CURVES);
  const n = plan.photos.length, failed = plan.whole.length;
  if (!n && !failed) {
    toast(all.some((r) => !r.photoDeleted) ? "Wait for the photos being measured." : "No photos left to delete: only curves.");
    return;
  }
  const what = [n ? `Delete ${n === 1 ? "1 photo" : `${n} photos`} and keep ${n === 1 ? "its curve" : "their curves"}?` : "",
    n ? "The curves can still be copied, downloaded and sent, but their pins can no longer be fixed." : "",
    failed ? `${failed === 1 ? "1 photo that gave no curve is" : `${failed} photos that gave no curve are`} deleted${n ? " too" : ""}.` : ""];
  if (!confirm(what.filter(Boolean).join(" "))) return;
  const t = await tidy(0, new Set());
  toast([t.photos ? `Deleted ${t.photos === 1 ? "1 photo" : `${t.photos} photos`}, kept the curves.` : "",
    t.whole ? `Deleted ${t.whole === 1 ? "1 photo" : `${t.whole} photos`} without a curve.` : "",
    t.failed ? `${t.failed === 1 ? "1 photo was" : `${t.failed} photos were`} kept: ${t.failed === 1 ? "its" : "their"} curve's files couldn't be made.` : ""]
    .filter(Boolean).join(" ") || "Nothing deleted.");
};

$("clear-all").onclick = async () => {
  const n = order.length;
  if (!n) return;
  const sent = room ? order.some((id) => syncOf(records.get(id)!, room!)) : false;
  if (!confirm(`Delete all ${n === 1 ? "1 photo and its curve" : `${n} photos and curves`} from this phone?${sent ? " Curves sent to the computer stay there." : ""}`)) return;
  await tidyChain;
  await photos?.clear().catch((e: Error) => toast(`Couldn't clear the phone's storage: ${e.message}`));
  for (const id of [...order]) {
    drop(id);
    void core.call("close", { photo: id }).catch(() => {});
  }
  toast(n === 1 ? "Deleted 1 photo." : `Deleted ${n} photos.`);
};

// ---- the engine (core worker) ----

let chainsLoaded: Promise<void>;

function engineProgress(p: Progress): void {
  $("engine-stage").textContent = ENGINE_STAGE[p.stage];
  $("engine-pct").textContent = `${Math.round(p.fraction * 100)}%`;
  $("engine-bar").style.width = `${p.fraction * 100}%`;
}

/** The engine's line under the header while it loads (a 2 px bar and a mono label); a failure shows where the card is, with Try again. */
async function startEngine(): Promise<void> {
  const box = $("engine");
  box.hidden = false;
  $("engine-fail").hidden = true;
  $("engine-note").textContent = "";
  try {
    const [loaded] = await Promise.all([core.preload(), chainsLoaded]);
    debug.engine = { readyAt: performance.now(), ms: loaded.ms };
    box.hidden = true;
    footer(`engine ${loaded.versions.splinewire ?? "?"} (TypeScript)`);
    measurer.coreReady();
  } catch (e) {
    box.hidden = true;
    $("engine-why").textContent = `${(e as Error).message}. Check the connection and try again.`;
    $("engine-fail").hidden = false;
  }
}

/** The worker died (most likely out of memory): the next call starts a new one, which loads again. */
function engineLost(): void {
  measurer.coreLoading();
  void startEngine();
}

function footer(extra = ""): void {
  $("build").textContent = [`Spline Wire ${import.meta.env.APP_VERSION}`, `build ${import.meta.env.BUILD_ID}`, extra].filter(Boolean).join(" · ");
}

/** The site no longer has a service worker (it kept Pyodide's 40 MB on the phone; D29).
 * Remove one an earlier visit installed, with its caches. sw.js does the same when the
 * browser updates it. Only this site's: other sites can share the origin (GitHub Pages). */
async function retireServiceWorker(): Promise<void> {
  // The site's root, the folder above assets/ where this script is (written so
  // that Vite does not take it for an asset of its own to bundle).
  const root = new URL(["..", ""].join("/"), import.meta.url).href;
  try {
    for (const reg of (await navigator.serviceWorker?.getRegistrations()) ?? []) {
      if (reg.scope.startsWith(root)) await reg.unregister();
    }
    if (typeof caches !== "undefined") {
      for (const name of await caches.keys()) if (name.startsWith("splinewire-")) await caches.delete(name);
    }
  } catch {
    /* not allowed here (private mode, http on a LAN address): nothing was installed */
  }
}

// ---- the selected photo: editor ----

let selected: string | null = null;
let pending: Pt | null = null;
let previewUrl: string | null = null;
const svg = $("photo") as unknown as SVGSVGElement;
// A double tap fits the chain to the view; a single tap edits once the double-tap window has passed (taps.ts).
const taps = new TapFilter(tap, () => editor.frame());
const editor = new PhotoEditor(svg, (p, slack, kind) => taps.tap(p, slack, kind), { wholePhoto: "width" });   // no chain: the width across
// The view changes size (the phone turned, the window resized): keep the same part in view.
new ResizeObserver(() => editor.refit()).observe(svg);

async function select(id: string | null): Promise<void> {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  editor.clear();
  selected = id;
  pending = null;
  taps.cancel();
  renderList();
  renderResult();
  const r = id ? records.get(id) : undefined;
  if (!r) return;
  if (r.status !== "queued" && r.status !== "loading" && r.status !== "measuring" && r.measuredWith !== focalKey()) {
    remeasure(r.id, true);             // the focal length settings changed since it was measured
  }
  const blob = await getBlob(r.id, "preview");
  // Shown grey while the live camera is (Settings, Live camera shows): only what is shown.
  const shown = blob && photosGrey() ? (await greyCopy(blob)) ?? blob : blob;
  if (selected !== r.id) return;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  if (shown && r.pixels) previewUrl = URL.createObjectURL(shown);
  editor.setImage(previewUrl, r.pixels);
  editor.frame();
  renderResult();                      // the card's picture, now that the photo is here
}

function tap(p: Pt, slack: number): void {
  const r = selected ? records.get(selected) : undefined;
  if (!r || r.photoDeleted || measurer.busy(r.id)) return;
  const v = r.doc?.view;
  if (!v) return;
  const { edits, mark } = editsAfterTap(v, p, slack);
  history(r.id).commit(v.edits);
  editing.add(r.id);
  r.edits = edits;
  pending = mark;
  remeasure(r.id, true);
  renderResult();
}

function history(id: string): EditHistory {
  let h = histories.get(id);
  if (!h) histories.set(id, (h = new EditHistory()));
  return h;
}

$("undo").onclick = () => {
  const r = selected ? records.get(selected) : undefined;
  if (!r || measurer.busy(r.id)) return;
  const e = history(r.id).undo();
  if (!e) return;
  r.edits = e;
  remeasure(r.id, true);
  renderResult();
};
$("reset").onclick = () => {
  const r = selected ? records.get(selected) : undefined;
  if (!r || measurer.busy(r.id) || !hasEdits(r.edits)) return;
  history(r.id).commit(r.edits);
  editing.add(r.id);
  r.edits = NO_EDITS;
  remeasure(r.id, true);
  renderResult();
};
addEventListener("resize", () => renderStrip());

const hasEdits = (e: Edits) => e.add.length > 0 || e.remove.length > 0;

/** Swap side: the curve to the other side of the chain from the one it was measured on.
 * New photos start with the side last chosen. */
function swapSide(): void {
  const r = selected ? records.get(selected) : undefined;
  if (!r || r.photoDeleted) return;
  const now = r.doc?.result?.object_side ?? (Array.isArray(r.side) ? "inside" : r.side);
  const side: Side = now === "inside" ? "outside" : "inside";
  r.side = side;
  settings = { ...settings, side };
  saveSettings(store, settings);
  remeasure(r.id, true);
  renderResult();
}
$("swap").onclick = swapSide;
$("x-swap").onclick = swapSide;

let pinsTimer: ReturnType<typeof setTimeout> | undefined;
const photoPins = stepper($("r-pins"), () => (selected ? records.get(selected)?.pins ?? 13 : 13), (n) => {
  const r = selected ? records.get(selected) : undefined;
  if (!r || r.photoDeleted) return;
  r.pins = n;
  settings = { ...settings, pins: n };           // as the current app does: the next photo has as many
  saveSettings(store, settings);
  renderSettings();
  clearTimeout(pinsTimer);                       // a few taps on − or + in a row measure once
  pinsTimer = setTimeout(() => remeasure(r.id, true), 500);
});

/** The chain style of this photo only (the default for new photos stays in Settings): a new chain brings its own pin count. */
$<HTMLSelectElement>("r-chain").onchange = (e) => {
  const r = selected ? records.get(selected) : undefined;
  const chain = (e.target as HTMLSelectElement).value;
  if (!r || r.photoDeleted || !(chain in chains) || chain === presetKey(chains, r.chain)) return;
  r.chain = chain;
  r.pins = chains[chain]!.n_pins;
  clearTimeout(pinsTimer);
  remeasure(r.id, true);
  renderList();
  renderResult();
};

function stepper(box: HTMLElement, get: () => number, set: (n: number) => void): HTMLInputElement {
  const input = box.querySelector("input")!;
  const commit = (n: number) => {
    const v = parsePins(String(n));
    if (v === "invalid") {
      input.value = String(get());
      return;
    }
    input.value = String(v);
    if (v !== get()) set(v);
  };
  for (const b of box.querySelectorAll<HTMLButtonElement>("button")) b.onclick = () => commit(get() + Number(b.dataset.step));
  input.onchange = () => commit(Number(input.value));
  return input;
}

// ---- the selected photo: what it says ----

const STATUS_WORDS: Record<PhotoRecord["status"], string> = {
  loading: "Waiting for the measuring engine…",
  queued: "Waiting for the photo before it…",
  measuring: "Measuring…",
  done: "",
  error: "",
};

function tagOf(r: PhotoRecord): [string, string] {
  if (r.status === "done") {
    const warned = (r.doc?.result?.warnings.length ?? 0) > 0;
    return warned ? ["chk", "CHECK"] : ["ok", "OK"];
  }
  if (r.status === "error") return ["bad", r.failure?.kind === "bug" ? "BUG" : "FAILED"];
  if (r.status === "measuring") return ["", "…"];
  return ["", "WAIT"];
}

function renderResult(): void {
  const r = selected ? records.get(selected) : undefined;
  $("result").hidden = !r;
  if (!r) {
    closeExpanded();
    return;
  }
  const doc = r.doc;
  const busy = measurer.busy(r.id);
  const gone = !!r.photoDeleted;
  const [tag, label] = tagOf(r);
  const failed = tag === "bad";

  // The card: the photo cropped to the chain and the curve, or the drawing once the photo is deleted.
  const card = $("result");
  for (const c of ["chk", "bad"]) card.classList.toggle(c, tag === c);
  card.classList.toggle("gone", gone);
  renderCardPicture(r);
  for (const star of [$("star"), $("x-star")]) {
    star.setAttribute("aria-pressed", String(starOf(r).on));
    star.setAttribute("aria-label", starOf(r).on ? `Unstar ${titleOf(r)}` : `Star ${titleOf(r)}`);
  }
  if (renaming?.host.id !== "r-name") $("r-name").textContent = nameOf(r).text;
  if (renaming?.host.id !== "x-nm") $("x-nm").textContent = nameOf(r).text;
  // The name, tapped, renames: a button to a screen reader while it is a name and not the field.
  for (const id of ["r-name", "x-nm"]) {
    const host = $(id), editing = renaming?.host.id === id, named = !!nameOf(r).text && !editing;
    host.classList.toggle("named", named);
    if (named) {
      host.setAttribute("role", "button");
      host.tabIndex = 0;
      host.setAttribute("aria-label", `Rename ${nameOf(r).text}`);
    } else for (const a of ["role", "tabindex", "aria-label"]) host.removeAttribute(a);
  }
  for (const t of [$("r-time"), $("x-time")]) t.textContent = timeOf(r);
  for (const t of [$("r-tag"), $("x-tag")]) {
    t.className = `tag ${tag}`;
    t.textContent = label;
  }
  $("r-status").textContent = STATUS_WORDS[r.status];
  $("swap").hidden = failed;
  for (const b of [$<HTMLButtonElement>("swap"), $<HTMLButtonElement>("x-swap")]) b.disabled = gone;
  $("x-swap").hidden = failed;

  // What went wrong: a person's error as is, under a plain heading; a bug with its type.
  const problem = $("r-problem");
  problem.replaceChildren();
  if (r.failure) {
    const f = failureText(r.failure);
    problem.append(el("b", { textContent: f.title }), el("div", { className: "raw", textContent: f.text }));
    if (f.bug) problem.append(el("div", { className: "advice", textContent: "Please report it with this text. Try the photo again or take another." }));
  } else if (doc && !doc.ok && doc.error) {
    const fixable = !!doc.view && (doc.view.pins.length + doc.view.unused.length) > 0;
    problem.append(el("b", { textContent: errorTitle(doc.error) }), el("div", { className: "raw", textContent: doc.error }));
    const advice = errorAdvice(doc.error, fixable);
    if (advice) problem.append(el("div", { className: "advice", textContent: advice }));
  }
  problem.hidden = !problem.childElementCount;

  // The expanded view: its title, the editor and the sheet.
  const xv = $("xv");
  for (const c of ["chk", "bad"]) xv.classList.toggle(c, tag === c);
  xv.setAttribute("aria-label", titleOf(r));
  $("x-details").hidden = !settings.debug;
  const info = doc?.info;
  const focal = info ? (info.focal_35mm ? `${info.focal_35mm} mm in the file` : "no focal length in the file") : "";
  const size = r.pixels ? `${r.pixels.width}×${r.pixels.height}` : "";
  const via = r.via === "camera" ? (r.camera?.still === "frame" ? `page camera, video frame${r.camera.why ? ` (takePhoto: ${r.camera.why})` : ""}` : "page camera")
    : r.via === "take" ? "system camera" : "library";
  const fov = r.camera?.fov != null ? `live view scale ${r.camera.fov.toFixed(3)}` : "";
  $("r-meta").textContent = [r.name, size, focal, `${chainLabel(r.chain)}`, via, fov].filter(Boolean).join(" · ");

  const v = doc?.view;
  svg.classList.toggle("wait", busy);
  svg.toggleAttribute("hidden", gone);
  editor.show(gone ? null : v ?? null, { pending });
  const drawing = $("x-drawing");
  drawing.hidden = !gone;
  const drawn = gone && doc?.curve ? curveDrawing(doc.curve, 1, { big: true, look: lookOf(r) }) : "";
  if (drawing.dataset.key !== `${r.id}:${drawn.length}`) {
    drawing.innerHTML = drawn;
    drawing.dataset.key = `${r.id}:${drawn.length}`;
  }
  const edited = !!v && hasEdits(v.edits);
  $<HTMLButtonElement>("undo").disabled = busy || gone || !history(r.id).canUndo;
  $<HTMLButtonElement>("reset").disabled = busy || gone || !edited;
  $("hint").textContent = gone ? ""
    : v ? "Tap a pin to remove it, a dot to add one. Drag to move, pinch to zoom, double-tap to fit."
    : r.pixels ? "" : "This browser can't show this photo, but it can still measure it.";
  photoPins.value = String(r.pins);
  const chainSelect = $<HTMLSelectElement>("r-chain");
  if (chainSelect.options.length !== Object.keys(chains).length) {
    chainSelect.replaceChildren(...Object.keys(chains).map((k) => new Option(chainLabel(k), k)));
  }
  chainSelect.value = presetKey(chains, r.chain);
  chainSelect.disabled = gone;
  for (const c of $("r-pins").querySelectorAll<HTMLInputElement | HTMLButtonElement>("input, button")) c.disabled = gone;

  const res = doc?.result;
  const focalDetail = res && r.focal && res.focal_source === "override" && settings.focalOverride35 === null ? photoFocalText(r.focal) : undefined;
  $("numbers").replaceChildren(...(res ? measurementRows(res, r.pins, focalDetail) : []).map(([k, val, sub, warn]) =>
    el("tr", { className: warn ? "chk" : "" }, el("th", { textContent: k, scope: "row" }), el("td", {}, val, ...(sub ? [el("small", { textContent: sub })] : [])))));
  const item = (x: { title: string; text: string }) => el("li", {}, el("b", { textContent: x.title }), x.text);
  $("warnings").replaceChildren(...(res?.warnings ?? []).map((w) => readWarning(w)).map(item));
  $("notes").replaceChildren(...(res?.notes ?? []).map((n) => readNote(n)).map(item));

  // Settings, This curve: copy, share, send, the files.
  const ok = r.status === "done" && !!doc?.ok;
  $<HTMLButtonElement>("copy").disabled = !ok || busy;
  $("share").hidden = !canShareFiles();
  $<HTMLButtonElement>("share").disabled = !ok || busy;
  renderSend(r, ok && !busy);
  $("downloads").replaceChildren(...(ok ? DOWNLOADS.map(([kind, label]) => {
    const b = el("button", { type: "button", className: "btn", textContent: label });
    b.setAttribute("aria-label", `Download the ${label} file`);
    b.disabled = busy;
    b.onclick = () => void download(r.id, kind);
    return b;
  }) : []));
}

/** The card's picture, drawn again only when it changes (the photo's <image> is not reloaded on every update). */
/** The part of the photo the card shows (photo px), or null when the card shows a drawing. */
function cardCropOf(r: PhotoRecord): Box | null {
  const v = r.doc?.view;
  if (r.photoDeleted || (!r.pixels && !v)) return null;
  const photo = v ? { width: v.width, height: v.height } : r.pixels!;
  const marks = v ? (v.pins.length ? v.pins : v.unused).map((m): [number, number] => [m.x, m.y]) : [];
  return marks.length >= 2 ? cardCrop(marks, v!.r, photo, CARD_ASPECT)
    : clampBox(fitBox([0, 0, photo.width, photo.height], CARD_ASPECT, 0), photo.width, photo.height);
}

function renderCardPicture(r: PhotoRecord): void {
  const pic = $("card-pic");
  const doc = r.doc, v = doc?.view;
  let html = "";
  if (r.photoDeleted || (!r.pixels && !v)) {
    html = doc?.curve ? curveDrawing(doc.curve, CARD_ASPECT, { pad: 10, big: true, look: lookOf(r) }) : "";
  } else {
    const photo = v ? { width: v.width, height: v.height } : r.pixels!;
    const crop = cardCropOf(r)!;
    const href = previewUrl && selected === r.id && r.pixels ? previewUrl : null;
    html = cardSvg({ href, photo, crop, curve: doc?.ok && v ? v.curve : null });
  }
  if (pic.dataset.html !== html) {
    pic.innerHTML = html;
    pic.dataset.html = html;
  }
}
const CARD_ASPECT = 1.8;

const DOWNLOADS: [OutputKind, string][] = [["dxf", "DXF"], ["svg", "SVG 1:1"], ["csv", "CSV"], ["json", "JSON"], ["fusion_csv", "Fusion CSV"]];

// ---- the list of photos ----

/** The chain a photo was measured with, in its colours, as the drawings show it; null until the presets are loaded. */
function lookOf(r: PhotoRecord): ChainSpec | null {
  return Object.keys(chains).length ? chainSpec(chains, r.chain, r.pins, r.shade ?? "light") : null;
}

/** The history's first rows; ••• shows the rest. */
const TILES = 12;
let fullHistory = false;

const STAR_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.8l2.7 5.9 6.4.7-4.8 4.3 1.4 6.3L12 16.8 6.3 20l1.4-6.3L2.9 9.4l6.4-.7z" stroke-linejoin="round"/></svg>';

function renderList(): void {
  $("photos-box").hidden = order.length === 0;
  const all = order.map((id) => records.get(id)!);
  const withPhoto = all.filter((r) => !r.photoDeleted);
  $("space").textContent = spaceText(withPhoto.length, withPhoto.reduce((n, r) => n + r.size, 0), all.length - withPhoto.length);
  $("free-space").hidden = withPhoto.length === 0;
  const shown = fullHistory ? order : order.slice(0, TILES);
  $("photos").replaceChildren(...shown.map((id) => tile(records.get(id)!)));
  const more = $("more");
  more.hidden = order.length <= TILES;
  more.setAttribute("aria-expanded", String(fullHistory));
  more.setAttribute("aria-label", fullHistory ? "Show less history" : "Show the full history");
  more.firstElementChild!.textContent = fullHistory ? "▲" : "•••";
}

/** A past curve: only the chain and the curve, no photo; its star and name, and a dot for Check or Failed. */
function tile(r: PhotoRecord): HTMLButtonElement {
  const [tag, label] = tagOf(r);
  const b = el("button", { type: "button", className: `tile${r.id === selected ? " on" : ""}` });
  const c = r.doc?.curve;
  b.innerHTML = c && c.pins.length > 1 ? curveDrawing(c, 1, { look: lookOf(r) }) : `<span class="wait">${tag === "bad" ? "—" : "…"}</span>`;
  if (starOf(r).on) {
    const name = el("span", { textContent: nameOf(r).text });
    const tl = el("span", { className: "tl" }, name);
    tl.insertAdjacentHTML("afterbegin", STAR_ICON);
    b.append(tl);
  }
  if (tag === "chk" || tag === "bad") b.append(el("span", { className: `dot ${tag}` }));
  b.setAttribute("aria-label", `Open ${titleOf(r)}, ${label}`);
  b.dataset.id = r.id;
  b.dataset.status = r.status;
  if (r.photoDeleted) b.dataset.photo = "deleted";
  b.onclick = () => {
    void select(r.id);
    scrollTo({ top: 0, behavior: reducedMotion() ? "auto" : "smooth" });
  };
  return b;
}

$("more").onclick = () => {
  fullHistory = !fullHistory;
  renderList();
};

const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---- the expanded view ----

/** The expanded view: a part of the page, in its flow, shown in place of the page's own content while open
 * (no overlay, no scroll of its own: the page scrolls as usual). The photo's box keeps its size while the page
 * scrolls (phone.css), so the editor never refits under a finger.
 *
 * Opening and closing animate with the browser's view transition where it has one (Safari 18, Chrome 111,
 * Firefox 144): it snapshots the page before and after and morphs the photo's box and its corner keys between
 * the card and the view (their view-transition-names, phone.css), crossfading the rest; the page itself stays
 * as it is. The pictures inside the box are not stretched to it (it changes shape, so they would grow past
 * both ends): `animatePictures` moves and scales them so the chain goes from where and how big the card shows it
 * to where and how big the view does, both snapshots in step. Elsewhere, and with reduced motion, it switches at once. */
let expanded = false;
/** Where the page was scrolled to when the view opened: back there when it closes. */
let scrolledTo = 0;

/** Show the page or the view, as `expanded` says now (a transition overtaken by another applies the newest). */
function showExpanded(): void {
  $("page").hidden = expanded;
  $("xv").hidden = !expanded;
  $("expand").setAttribute("aria-expanded", String(expanded));
  if (expanded) {
    renderResult();
    scrollTo(0, 0);
    editor.frame();
    $<HTMLButtonElement>("collapse").focus({ preventScroll: true });
  } else {
    scrollTo(0, scrolledTo);
    if (!$("result").hidden) $<HTMLButtonElement>("expand").focus({ preventScroll: true });
  }
}

/** The view transition's length and easing: as phone.css gives its groups. */
const VT_MS = 280, VT_EASE = "cubic-bezier(.3, 0, 0, 1)";

/** The picture in the card or in the view (the photo, or the drawing once the photo is deleted). */
function pictureOf(inView: boolean): SVGSVGElement | null {
  if (!inView) return $("card-pic").querySelector("svg");
  return $("x-drawing").hidden ? svg : $("x-drawing").querySelector("svg");
}

/** Where a picture's content is on the screen: its user units (photo px, or mm) to the screen, its centre and size. */
interface Placed { m: DOMMatrix; cx: number; cy: number; w: number; h: number }
function placeOf(el: SVGSVGElement | null): Placed | null {
  const m = el?.getScreenCTM(), b = el?.getBoundingClientRect();
  return m && b && b.width && b.height ? { m, cx: b.left + b.width / 2, cy: b.top + b.height / 2, w: b.width, h: b.height } : null;
}

/** The two snapshots of the picture in the morphing box (phone.css keeps each at its own size, clipped to the box).
 * They are moved and scaled so that both show the chain at the same place and size all the way: at the start where
 * the old picture has it, at the end where the new one has it. Its content point at the old picture's centre leads;
 * `r` is how much bigger the new picture draws it. (Both are transformed about their centres, which are the box's.)
 * As they show the same thing in the same place, they are not crossfaded (half see-through, the box's background
 * would show at the edge of the smaller): the one that covers more of the box stays opaque underneath, the other
 * fades over it (in when it is the new one, out when the old). Returns the animations, which the transition's end
 * must cancel: they hold their last frame, and the next transition's snapshots have the same names. */
function animatePictures(from: Placed | null, to: Placed | null): Animation[] {
  const root = document.documentElement, timing = { duration: VT_MS, easing: VT_EASE, fill: "both" as const };
  const OLD = "::view-transition-old(x-photo)", NEW = "::view-transition-new(x-photo)";
  const r = from && to ? to.m.a / from.m.a : NaN;
  if (!from || !to || !Number.isFinite(r) || r < 0.05 || r > 20) {
    return [root.animate({ opacity: [0, 1] }, { ...timing, pseudoElement: NEW })];      // just the new over the old
  }
  const p = new DOMPoint(from.cx, from.cy).matrixTransform(from.m.inverse()).matrixTransform(to.m);
  const dx = p.x - to.cx, dy = p.y - to.cy;
  const moves = [
    root.animate({ transform: ["none", `translate(${dx}px, ${dy}px) scale(${r})`] }, { ...timing, pseudoElement: OLD }),
    root.animate({ transform: [`translate(${-dx / r}px, ${-dy / r}px) scale(${1 / r})`, "none"] }, { ...timing, pseudoElement: NEW }),
  ];
  // How much of the box each covers where it covers least: the old at the end, the new at the start.
  const oldCover = Math.min((from.w * r) / to.w, (from.h * r) / to.h), newCover = Math.min(to.w / r / from.w, to.h / r / from.h);
  const oldUnder = oldCover > newCover;
  root.classList.toggle("vt-old-over", !oldUnder);
  return [...moves, oldUnder ? root.animate({ opacity: [0, 1] }, { ...timing, pseudoElement: NEW })
    : root.animate({ opacity: [1, 0] }, { ...timing, pseudoElement: OLD })];
}

/** The view's settings sliding out from under the photo (new: opening) or back under it (old: closing), fading
 * as they go (phone.css clips them to their place). Made afresh for each transition, like the pictures': as CSS
 * keyframes on the transition's pseudo-elements the slide ran on the first transition only, in Safari. */
function slideSheet(opening: boolean): Animation[] {
  const root = document.documentElement, timing = { duration: VT_MS, easing: VT_EASE, fill: "both" as const };
  const away = { transform: "translateY(-240px)", opacity: 0 }, home = { transform: "none", opacity: 1 };
  return [opening ? root.animate([away, home], { ...timing, pseudoElement: "::view-transition-new(x-sheet)" })
    : root.animate([home, away], { ...timing, pseudoElement: "::view-transition-old(x-sheet)" })];
}

/** showExpanded, animated where the browser can and the person has not asked for less motion. */
function switchView(animate: boolean): void {
  if (!animate || reducedMotion() || !document.startViewTransition) {
    showExpanded();
    return;
  }
  const root = document.documentElement;
  const r = selected ? records.get(selected) : undefined;
  root.classList.add("vt");                       // while it runs (the end-to-end tests wait for it)
  root.classList.toggle("vt-drawing", !!r?.photoDeleted);
  for (const c of ["chk", "bad"]) root.classList.toggle(`vt-${c}`, $("result").classList.contains(c));   // the frame's colour
  const opening = expanded, from = placeOf(pictureOf(!expanded));     // the picture shown before the switch
  let to: Placed | null = null;
  const t = document.startViewTransition(() => {
    showExpanded();
    to = placeOf(pictureOf(expanded));
  });
  let pictures: Animation[] = [];
  void t.ready.then(() => {
    // The page is not snapshotted (phone.css): the one now shown fades in around the moving pieces.
    const page = $(opening ? "xv" : "page").animate({ opacity: [0, 1] }, { duration: VT_MS, easing: VT_EASE });
    pictures = [...animatePictures(from, to), ...slideSheet(opening), page];
  }, () => {});
  void t.finished.finally(() => {
    for (const a of pictures) a.cancel();
    root.classList.remove("vt", "vt-drawing", "vt-old-over", "vt-chk", "vt-bad");
  });
}

function openExpanded(): void {
  if (expanded || !selected) return;
  expanded = true;
  scrolledTo = scrollY;
  history_.pushState({ ...(history_.state ?? {}), expanded: true }, "");
  switchView(true);
}

/** Back to the page, where it was. `fromHistory`: the back gesture or key did it; `animate`: false when the
 * browser has already animated it (Safari's swipe back). */
function closeExpanded(fromHistory = false, animate = true): void {
  if (!expanded) return;
  expanded = false;
  taps.cancel();
  if (renaming?.host.id === "x-nm") endRename(true);
  if (!fromHistory && history_.state?.expanded) history_.back();
  switchView(animate);
}

$("expand").onclick = openExpanded;
$("collapse").onclick = () => closeExpanded();
addEventListener("popstate", (e: PopStateEvent) => {
  if (expanded && !history_.state?.expanded) closeExpanded(true, !e.hasUAVisualTransition);
});
addEventListener("keydown", (e: KeyboardEvent) => {
  if (e.key === "Escape" && expanded && !renaming && !cameraView.isOpen) closeExpanded();
});

// ---- copy, share, download ----

type Outputs = Commands["outputs"]["result"];
/** A photo's output files as kept beside it (photos.ts "outputs"), with the measurement they are of. */
interface KeptOutputs extends Outputs { doc: string }
const outputsWork = new Map<string, { doc: string; done: Promise<Outputs> }>();

/** What identifies a photo's files: its document and the choices it was measured with
 * (a tapped side and "inside" give the same curve, but the JSON file says which). Not the
 * focal length settings: a change to them measures the photo again, and a new document. */
function docKey(r: PhotoRecord, doc: MeasurementDocument): string {
  return JSON.stringify([doc, r.side, r.edits, r.pins, r.chain, r.shade ?? "light"]);
}

/** The photo's output files: kept ones while they are of its measurement, else made by the core and kept. */
function outputsOf(id: string): Promise<Outputs> {
  const r = records.get(id);
  if (!r?.doc) return Promise.reject(new Error("the photo isn't measured"));
  const doc = r.doc, key = docKey(r, doc);
  const work = outputsWork.get(id);
  if (work?.doc === key) return work.done;
  const done = (async () => {
    const kept = await keptOutputs(id);
    if (kept?.doc === key) return { files: kept.files, names: kept.names };
    if (r.photoDeleted) throw new Error("its photo was deleted, and the curve's files with it");
    const out = await coreOutputs(id);
    // Measured again meanwhile: the files may be of either measurement, so not kept.
    if (records.get(id) === r && r.doc === doc && !measurer.busy(id)) {
      const blob = new Blob([JSON.stringify({ ...out, doc: key } satisfies KeptOutputs)], { type: "application/json" });
      if (photos && !memBlobs.has(`${id}:original`)) await photos.putBlob(id, "outputs", blob).catch(() => {});
      else memBlobs.set(`${id}:outputs`, blob);
    }
    return out;
  })();
  outputsWork.set(id, { doc: key, done });
  void done.finally(() => {
    if (outputsWork.get(id)?.done === done) outputsWork.delete(id);
  }).catch(() => {});
  return done;
}

async function keptOutputs(id: string): Promise<KeptOutputs | null> {
  const blob = await getBlob(id, "outputs");
  if (!blob) return null;
  try {
    return JSON.parse(await blob.text()) as KeptOutputs;
  } catch {
    return null;
  }
}

async function coreOutputs(id: string): Promise<Outputs> {
  try {
    return await core.call("outputs", { photo: id });
  } catch (err) {
    if (!(err instanceof CoreError) || err.failure.kind !== "closed") throw err;
  }
  // The worker closed the photo (memory) or the page was reloaded: open and
  // measure it again with the same options, which gives the same measurement.
  await core.measure(id, options(id)!, source(id));
  return core.call("outputs", { photo: id });
}

$("copy").onclick = async () => {
  const doc = selected ? records.get(selected)?.doc : null;
  if (!doc?.ok || !doc.curve) return;
  try {
    await copyText(pointsTsv(doc.curve.points));
    toast(`Copied ${doc.curve.points.length} points. In Fusion: Paste points.`);
  } catch (e) {
    toast(`Couldn't copy: ${(e as Error).message}`);
  }
};

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // No async clipboard (a plain-http address on the local network).
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.append(ta);
  ta.select();
  const done = document.execCommand("copy");
  ta.remove();
  if (!done) throw new Error("this browser doesn't allow it here");
}

function canShareFiles(): boolean {
  try {
    return typeof navigator.share === "function" && !!navigator.canShare?.({ files: [new File(["x"], "x.csv", { type: "text/csv" })] });
  } catch {
    return false;
  }
}

const TYPES: Partial<Record<OutputKind, string>> = {
  dxf: "application/dxf", svg: "image/svg+xml", csv: "text/csv", json: "application/json", fusion_csv: "text/csv",
};

/** A file's name: the core's (after the photo's), or after the person's name for the curve. */
function fileName(id: string, out: Outputs, kind: OutputKind): string {
  const r = records.get(id), given = out.names[kind] ?? `curve.${kind}`;
  return r ? renamedFile(given, r.name, nameOf(r).text) : given;
}

function fileOf(id: string, out: Outputs, kind: OutputKind): File {
  return new File([out.files[kind]], fileName(id, out, kind), { type: TYPES[kind] ?? "text/plain" });
}

$("share").onclick = async () => {
  const id = selected;
  if (!id) return;
  try {
    const out = await outputsOf(id);
    // Not every phone shares every file type (Chrome on Android refuses .dxf):
    // try the DXF with the CSV, then the CSV alone.
    const tries = [[fileOf(id, out, "csv"), fileOf(id, out, "dxf")], [fileOf(id, out, "csv")]];
    const files = tries.find((f) => navigator.canShare?.({ files: f }));
    if (!files) throw new Error("this phone can't share these files; use the downloads");
    const r = records.get(id);
    await navigator.share({ files, title: r ? displayName(r) : "Curve" });
  } catch (e) {
    if ((e as Error).name !== "AbortError") toast(`Couldn't share: ${(e as Error).message}`);
  }
};

async function download(id: string, kind: OutputKind): Promise<void> {
  try {
    const out = await outputsOf(id);
    const url = URL.createObjectURL(fileOf(id, out, kind));
    const a = el("a", { href: url, download: fileName(id, out, kind) });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
  } catch (e) {
    toast(`Couldn't make the file: ${(e as Error).message}`);
  }
}

// ---- pairing and sending ----

let link: RelayLink | null = null;
let outbox: Outbox | null = null;
let linkStatus: LinkStatus = { state: "off", peers: 0 };
const relayUrl = relayAddress(pageRelaySetting());
let secret: string | null = null;
/** The pairing's room: what a photo's curve is synced with (send.ts CurveSync). */
let room: string | null = null;
/** Curves deleted on this phone or on the computer (relay/sync.ts). */
const deletedIds = new DeletedIds(store, "deleted");
/** The photos are read from storage: until then a full state would say nothing. */
let recordsLoaded = false;

const paired = () => outbox !== null;

/** Bumped by each start, so a start overtaken by a newer one (two quick scans) leaves no link behind. */
let pairingRun = 0;

/** Pair from `hash` (#pair=<secret>): the page's address, or a code the camera saw (which is not in the address). */
async function startPairing(hash = location.hash): Promise<void> {
  const run = ++pairingRun;
  const fromAddress = hash === location.hash;
  secret = phoneSecret(store, hash, () => { if (fromAddress) history_.replaceState(null, "", location.pathname + location.search); });
  if (!secret || !relayUrl) {
    renderPairing();
    return;
  }
  const keys = await deriveKeys(secret);
  if (run !== pairingRun) return;
  room = keys.room;
  const hello = () => void link?.send({ type: "hello", role: "phone", device: deviceName(navigator.userAgent), app: import.meta.env.APP_VERSION });
  link = new RelayLink({
    relayUrl, keys, role: "phone",
    events: {
      status: (s) => {
        const before = linkStatus;
        linkStatus = s;
        debug.link.push({ ...s, t: Math.round(performance.now()) });
        outbox?.onStatus(s, before);
        if (s.state === "open" && (before.state !== "open" || s.peers > before.peers)) {
          hello();
          sendFullState();
        }
        renderPairing();
        renderResult();
      },
      message: (m) => {
        outbox?.onMessage(m);
        if (m.type === "hello" && m.role === "desktop") {
          sendFullState();
          if (m.device && m.device !== computerName()) {
            store.set("computer", m.device);
            renderPairing();
          }
        }
        if (m.type === "state") onState(m);
      },
    },
  });
  outbox = new Outbox(store, link, outboxChanged);
  link.start();
  renderPairing();
  renderResult();
}

/** Everything this phone holds of its curves: the computer answers with all it holds. */
function sendFullState(): void {
  if (!link || !room || !recordsLoaded) return;
  void link.send(phoneState(records.values(), deletedIds.recent(), room, true));
}

/** A change to some curves (a star, a deletion), at once; the next meeting repeats it if it is lost. */
function sendChange(ids: string[], deleted: string[] = []): void {
  if (!link || !room) return;
  void link.send(phoneState(records.values(), deleted, room, false, ids));
}

/** What the computer holds: revisions it has, stars set there, curves deleted there. */
function onState(m: Extract<Message, { type: "state" }>): void {
  if (!room) return;
  const { changed, deleted, behind } = applyState(records.values(), m, room);
  for (const r of changed) persist(r);
  if (behind.length) sendChange(behind);
  for (const r of deleted) {
    deletedIds.add([r.sync!.id]);
    outbox?.forget(r.sync!.id);
    void removePhoto(r.id);
    toast(`Deleted on the computer: ${displayName(r)}`);
  }
  if (changed.length || deleted.length) {
    renderList();
    if (selected) renderResult();
  }
}

/** After a measurement: the photo's curve to the computer, when it goes there (send.ts). */
async function syncCurve(r: PhotoRecord): Promise<void> {
  const doc = r.doc;
  if (!outbox || !room || !doc?.ok) return;
  const next = nextRevision(r, room);
  if (!next) return;
  try {
    const out = await outputsOf(r.id);
    // Measured again meanwhile (another tap): that measurement sends its own.
    if (r.doc !== doc || !records.has(r.id) || !outbox) return;
    const spec = chainSpec(chains, r.chain, r.pins, r.shade);
    const curve = curveFrom(doc, { chainName: chainLabel(r.chain), pitch_mm: spec.pitch_mm, half_width_mm: spec.half_width_mm, n_pins: r.pins },
      await pack(out.files));
    if (!curve || r.doc !== doc) return;
    curve.id = next.id;
    curve.rev = next.rev;
    r.sync = { ...next, remoteRev: syncOf(r, room)?.remoteRev ?? 0 };
    r.autoSend = true;
    persist(r);
    await outbox.put(curve);
  } catch (e) {
    toast(`Couldn't send: ${(e as Error).message}`);
  } finally {
    renderPairing();
    if (r.id === selected) renderResult();
  }
}

/** The computer's name, as its page says in its hello ("" until it has). */
const computerName = (): string => store.get("computer") ?? "";
/** When this visit last saw a curve delivered: the pill's "sent 18:42". */
let lastSent: number | null = null;
let deliveredSeen = 0;

function outboxChanged(): void {
  if (!outbox) return;
  if (outbox.delivered.length > deliveredSeen) lastSent = Date.now();
  deliveredSeen = outbox.delivered.length;
  renderPairing();
  if (selected) renderResult();
}

function renderSend(r: PhotoRecord, enabled: boolean): void {
  const offer = sendOffer(paired() ? room : null, r);
  const button = $<HTMLButtonElement>("send");
  const note = $("send-note");
  button.hidden = offer.kind !== "send";
  button.disabled = !enabled;
  button.textContent = "Send to computer";
  note.textContent = "";
  if (offer.kind === "synced") {
    const s = offer.sync;
    const now = r.doc && curveKey(r.doc);
    note.textContent = now && now !== s.key && (r.status === "done" || r.status === "error")
      ? "The computer has the last curve this photo gave."
      : deliveryText(delivery(s, outbox?.items ?? [], linkStatus), s.rev);
  } else if (offer.kind === "send") {
    note.textContent = "Taken before this phone was paired: send it, and the computer follows your changes from then on.";
  } else if (offer.kind === "unpaired" && r.status === "done") {
    note.textContent = "No computer paired";
  }
}

$("send").onclick = async () => {
  const r = selected ? records.get(selected) : undefined;
  if (!r?.doc?.ok || !outbox) return;
  $<HTMLButtonElement>("send").disabled = true;
  r.autoSend = true;
  await syncCurve(r);
  toast("Sent to the computer. Changes you make to it here follow.");
};

/** The computer pill: a lamp, the computer's name and how it is (linked; off, with curves queued; not paired). */
function renderPairing(): void {
  const lamp = $("lamp");
  const chip = $("link-text");
  const text = $("pair-text");
  $("unpair").hidden = !secret;
  $("link-who").textContent = secret ? computerName() : "";
  const queued = outbox?.items.length ?? 0;
  const waiting = queued ? ` · ${queued === 1 ? "1 curve" : `${queued} curves`} queued` : "";
  if (!secret) {
    lamp.className = "lamp";
    chip.textContent = "Not paired · tap to pair a computer";
    text.textContent = "Not paired. To send curves to a computer, open the Spline Wire site on the computer and point Scan (or the phone's Camera app) at the code it shows. Pairing is done once per phone. Copy points, Share and the downloads work without it.";
  } else if (!relayUrl) {
    lamp.className = "lamp bad";
    chip.textContent = "No relay";
    text.textContent = "Paired, but this build of the site has no relay address, so curves can't be sent. Copy points, Share and the downloads still work.";
  } else {
    const s = linkStatus;
    const at = lastSent ? ` · sent ${new Date(lastSent).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hourCycle: "h23" })}` : "";
    const [cls, short, long] =
      s.state === "open" && s.peers > 0 ? ["ok", `Linked${waiting || at}`, "Paired. The computer's page is open: each photo's curve goes there once measured, and your changes follow."]
      : s.state === "open" ? ["chk", `Off${waiting}`, "Paired. The computer's page is not open: curves wait here until it is."]
      : s.state === "waiting" ? ["bad", `Can't reach the relay${waiting}`, `Paired, but the relay can't be reached${s.problem ? ` (${s.problem})` : ""}. Curves wait here and go when it is back.`]
      : ["", "Connecting…", "Paired. Connecting to the relay…"];
    lamp.className = `lamp ${cls}`;
    chip.textContent = short;
    text.textContent = long;
  }
  // Curves still on their way, then those delivered during this visit.
  const items = outbox?.items ?? [];
  const rows = [
    ...items.map((i) => ({ id: i.curve.id, name: nameOfSent(i.curve.id, i.curve.photo), rev: i.curve.rev ?? 1,
      d: delivery({ id: i.curve.id, room: "", rev: 1, key: "", remoteRev: 0 }, items, linkStatus) })),
    ...[...new Set(outbox?.delivered ?? [])].slice(-10).reverse().filter((id) => !items.some((i) => i.curve.id === id))
      .map((id) => ({ id, name: nameOfSent(id), rev: 1, d: "delivered" as const })),
  ];
  $("outbox").replaceChildren(...rows.map((row) => {
    // One line a curve: its name, a word for where it is (the sentence is in the text above and in Settings), Don't send.
    const state = el("span", { className: `state${row.d === "delivered" ? " ok" : ""}`, textContent: OUTBOX_WORD[row.d] });
    state.title = deliveryText(row.d, row.rev);
    const li = el("li", {}, el("span", { className: "what", textContent: row.name }), state);
    li.dataset.curve = row.id;
    li.dataset.state = row.d;
    const r = recordOfCurve(row.id);
    // A curve the computer has never had can be held back; a change to one it has goes with the next.
    if (row.d !== "delivered" && (!r?.sync || r.sync.remoteRev === 0)) {
      const b = el("button", { type: "button", className: "btn plain", textContent: "Don't send" });
      b.setAttribute("aria-label", `Don't send the curve from ${row.name}`);
      b.onclick = () => {
        outbox?.forget(row.id);
        if (r) {
          r.sync = null;
          r.autoSend = false;
          persist(r);
        }
        renderPairing();
        if (selected) renderResult();
      };
      li.append(b);
    }
    return li;
  }));
}

function recordOfCurve(id: string): PhotoRecord | undefined {
  for (const r of records.values()) if (r.sync?.id === id || r.sent?.some((s) => s.id === id)) return r;
  return undefined;
}

/** What the outbox calls a curve: as the page does (its name, else "Curve 18:42"), or the photo's file name once it is gone from the phone. */
function nameOfSent(id: string, photo = "a curve"): string {
  const r = recordOfCurve(id);
  return r ? titleOf(r) : photo;
}

const OUTBOX_WORD: Record<ReturnType<typeof delivery>, string> = {
  "queued-no-computer": "Queued", "queued-offline": "Queued", sent: "Sent", delivered: "Delivered",
};

$("link-chip").onclick = () => {
  const box = $("pair-box");
  box.hidden = !box.hidden;
  $("link-chip").setAttribute("aria-expanded", String(!box.hidden));
};

$("unpair").onclick = () => {
  if (!confirm("Forget this computer? Curves not yet delivered are dropped. To pair again, scan the computer's code.")) return;
  for (const i of [...(outbox?.items ?? [])]) outbox?.forget(i.curve.id);
  link?.stop();
  forgetPairing(store);
  store.remove("computer");
  lastSent = null;
  link = null;
  outbox = null;
  secret = null;
  room = null;
  linkStatus = { state: "off", peers: 0 };
  renderPairing();
  renderResult();
};

// A code scanned while this page is already open in the browser: the address
// differs from the open one only in its #fragment, so the browser does not load
// the page again, it only changes location.hash. Pair here instead.
addEventListener("hashchange", () => {
  if (secretFromHash(location.hash)) pairAgain(location.hash);
});

/** Pair with the computer whose code this is, in place of the current pairing. */
function pairAgain(hash: string): void {
  const before = secret;
  if (secretFromHash(hash) === before && link) {
    toast("Already paired with this computer.");
    if (hash === location.hash) history_.replaceState(history_.state, "", location.pathname + location.search);
    return;
  }
  link?.stop();
  link = null;
  outbox = null;
  linkStatus = { state: "off", peers: 0 };
  void startPairing(hash)
    .then(() => { if (secret && secret !== before) toast("Paired with the computer."); })
    .catch((e: Error) => toast(`Pairing: ${e.message}`));
}

/** A QR code the live camera saw: the desktop page's pairing code pairs here,
 * without leaving the camera (the address is not changed: that would close it). */
function cameraCode(text: string): void {
  const m = codeMeaning(text, location.href);
  if (m.kind === "pair") pairAgain(`#pair=${m.secret}`);
  else if (m.kind === "elsewhere") toast(`That pairing code is for another copy of the site (${m.where}). Open the site at ${location.host} on the computer.`, 6000);
}

addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") link?.retryNow();
});
addEventListener("online", () => link?.retryNow());

// ---- settings ----

/** The chain strip under the header: the chosen chain in its colours, one pin per pin of the
 * chain, the new link popping in on +. Empty until the presets are loaded. */
let strip = { pins: 0, width: 0, look: "" };
function renderStrip(): void {
  const box = $("chain-strip");
  const pins = defaultPins(settings, chains), width = Math.round(box.clientWidth) || 358;
  const spec = Object.keys(chains).length ? chainSpec(chains, settings.chain, pins, settings.shade) : null;
  const look = spec ? `${presetKey(chains, settings.chain)} ${settings.shade}` : "";
  if (pins === strip.pins && width === strip.width && look === strip.look) return;
  const grew = strip.pins && pins === strip.pins + 1 && width === strip.width && look === strip.look ? pins - 1 : -1;
  box.innerHTML = stripSvg(spec, pins, width, grew);
  strip = { pins, width, look };
}

function renderSettings(): void {
  const select = $<HTMLSelectElement>("s-chain");
  if (!select.options.length || select.options.length !== Object.keys(chains).length) {
    select.replaceChildren(...Object.keys(chains).map((k) => new Option(chainLabel(k), k)));
  }
  select.value = presetKey(chains, settings.chain);
  $<HTMLSelectElement>("s-shade").value = settings.shade;
  $<HTMLSelectElement>("s-live").value = settings.liveSize;
  // The chain-only view is for debugging; one chosen there stays shown, so the menu tells the truth.
  $("s-view-chain").hidden = !settings.debug && settings.liveView !== "chain";
  $<HTMLSelectElement>("s-view").value = settings.liveView;
  $("debug-box").hidden = !settings.debug;
  const leftovers = settings.debug ? [] : debugLeftovers(settings, chains);
  $("debug-note").hidden = !leftovers.length;
  $("debug-note").textContent = leftovers.length ? `Debug settings still on: ${leftovers.join(", ")}. Open the page with ?debug to change them.` : "";
  document.documentElement.classList.toggle("photos-grey", photosGrey());
  $<HTMLSelectElement>("s-keep").value = String(settings.keepPhotos);
  settingsPins.value = String(defaultPins(settings, chains));
  renderStrip();
  // The focal length in use for photos without camera data: the one typed, else the detected camera's.
  const focal = $<HTMLInputElement>("s-focal");
  const saved = savedFocal(settings), learned = mainLens(settings);
  if (document.activeElement !== focal) focal.value = saved?.focal35.toString() ?? "";
  const override = $<HTMLInputElement>("s-override");
  if (document.activeElement !== override) override.value = settings.focalOverride35?.toString() ?? "";
  $("s-focal-status").textContent = focalStatus(saved, learned);
  const shown = learned ? cameraLines(learned) : null;
  $("s-camera").replaceChildren(...(shown
    ? [el("span", { textContent: shown.camera }), el("b", { textContent: shown.focal })]
    : [el("span", { textContent: "None yet" })]));
}

function setSettings(next: Settings, measureAgain = false): void {
  settings = next;
  saveSettings(store, settings);
  renderSettings();
  if (measureAgain && selected) {
    const r = records.get(selected);
    if (r && (r.status === "done" || r.status === "error")) remeasure(r.id, true);
  }
}

$<HTMLSelectElement>("s-chain").onchange = (e) => {
  const chain = (e.target as HTMLSelectElement).value;
  // A new chain brings its own pin count.
  setSettings({ ...settings, chain, pins: null });
  toast("New photos are measured with this chain.");
};
$<HTMLSelectElement>("s-shade").onchange = (e) => {
  const shade = (e.target as HTMLSelectElement).value as Shade;
  setSettings({ ...settings, shade });
  // Most often chosen after a photo of such a chain found nothing: measure it again too.
  const r = selected ? records.get(selected) : undefined;
  if (r && !r.photoDeleted && (r.shade ?? "light") !== shade) {
    r.shade = shade;
    remeasure(r.id, true);
    renderResult();
  }
  toast(shade === "dark" ? "Measuring dark dots or rims on light links." : "Measuring light dots or rims on dark links.");
};
$<HTMLSelectElement>("s-live").onchange = (e) => {
  const liveSize = (e.target as HTMLSelectElement).value === "full" ? "full" : "fast";
  // Asked for again: a camera whose still came back small is tried at the fast size again.
  setSettings({ ...settings, liveSize, fullStreamCameras: [] });
  toast(liveSize === "fast" ? "The live camera reads a smaller picture once the focal length is known: faster, for close-up shots."
    : "The live camera reads the full picture: slower, but reaches a chain farther away.");
};
$<HTMLSelectElement>("s-view").onchange = (e) => {
  const v = (e.target as HTMLSelectElement).value;
  setSettings({ ...settings, liveView: v === "chain" || v === "color" ? v : "grey" });
  if (selected) void select(selected);           // the photo shown again, grey or in colour
};

/** The photos are shown grey unless the live camera is in colour (the chain-only view counts as grey). */
function photosGrey(): boolean {
  return settings.liveView !== "color";
}
$<HTMLSelectElement>("s-keep").onchange = (e) => {
  const keepPhotos = Number((e.target as HTMLSelectElement).value);
  if (!(KEEP_PHOTO_CHOICES as readonly number[]).includes(keepPhotos)) return;
  setSettings({ ...settings, keepPhotos });
  renderList();
  void tidy(keepPhotos, new Set(selected ? [selected] : [])).then((t) => {
    if (t.photos) toast(`Deleted the ${t.photos === 1 ? "photo" : `${t.photos} photos`} past the newest ${keepPhotos}, kept the curves.`);
  }, () => {});
};
const settingsPins = stepper($("s-pins"), () => defaultPins(settings, chains), (n) => setSettings({ ...settings, pins: n }));

function focalInput(id: string, field: "savedFocal35" | "focalOverride35"): void {
  const input = $<HTMLInputElement>(id);
  input.onchange = () => {
    const v = parseFocal(input.value);
    if (v === "invalid") {
      $("s-error").textContent = "A focal length is a number of millimetres between 5 and 500 (35 mm equivalent), or empty.";
      return;
    }
    $("s-error").textContent = "";
    setSettings({ ...settings, [field]: v }, true);
  };
}
focalInput("s-focal", "savedFocal35");
focalInput("s-override", "focalOverride35");
$("debug-off").onclick = () => {
  setSettings({ ...settings, debug: false });
  // Off the address too, so a reload does not turn it on again.
  const url = new URL(location.href);
  url.searchParams.delete("debug");
  window.history.replaceState(window.history.state, "", url);
  toast("Debug mode is off. Open the page with ?debug to turn it on again.");
};

// Teaching the focal length from any photo of this phone's camera (D17): what
// Android's in-page stills leave out (S2: the Pixel's takePhoto has no 35 mm tag).
$<HTMLInputElement>("s-teach").onchange = async (e) => {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  const exif = await exifOfFile(file);
  if (isFront(exif)) {
    toast("That photo is from the front camera. Pick one the back camera took.", 6000);
    return;
  }
  if (exif.focal35 === null || exif.focalMm === null && !exif.lens) {
    toast("That photo has no focal length in it. Pick one the phone's own camera app took, not a screenshot or a shared copy.", 7000);
    return;
  }
  // The photo's size, upright: from the browser, or for a HEIC it cannot decode, from the engine.
  let pixels = (await previews(file))?.pixels ?? null;
  if (!pixels) {
    const id = `learn-${Date.now()}`;
    try {
      const data = await file.arrayBuffer();
      pixels = (await core.call("open", { photo: id, name: file.name || "photo", data, previews: true }, [data])).previews?.pixels ?? null;
    } catch {
      pixels = null;
    } finally {
      void core.call("close", { photo: id }).catch(() => {});
    }
  }
  if (!pixels) {
    toast("This browser can't read that photo's size, so its focal length can't be learned from it.", 7000);
    return;
  }
  const before = focalKey();
  // A new main camera changes the focal length for photos without camera data: the open one is measured again.
  setSettings(learn(settings, exif, pixels), focalKey() !== before);
  const shown = cameraLines({ name: lensName(exif), focal35: exif.focal35!, focalMm: exif.focalMm });
  toast(`Learned ${shown.focal} for ${shown.camera}.`, 5000);
  // Photos already taken with that lens, which lacked it, are measured with it now.
  for (const r of records.values()) {
    if (r.focal || !r.exif || r.exif.focal35 !== null) continue;
    const known = lensFocal(settings, r.exif);
    if (!known) continue;
    r.focal = { focal35: known.focal35, why: "lens", lens: known.lens };
    r.measuredWith = null;
    persist(r);
    if (r.id === selected && (r.status === "done" || r.status === "error")) remeasure(r.id, true);
  }
};

// ---- the rest ----

let toastTimer: ReturnType<typeof setTimeout> | undefined;
function toast(text: string, ms = 3500): void {
  const t = $("toast");
  t.textContent = text;
  t.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("on"), ms);
}

// ---- the camera ----

const cameraView = new CameraView({
  chain: async () => {
    await chainsLoaded;
    return chainSpec(chains, presetKey(chains, settings.chain), defaultPins(settings, chains), settings.shade);
  },
  settings: () => settings,
  saveCamera: (camera: CameraInfo) => {
    if (camera.deviceId !== settings.camera?.deviceId || camera.label !== settings.camera?.label) setSettings({ ...settings, camera });
  },
  // The overlay's analysis runs in TypeScript in its own small worker (src/pipeline/live.ts):
  // it needs no measuring engine, so coaching starts with the first frame.
  engineReady: () => true,
  analyse: (points) => live.analyse(points),
  still: (still) => void addFiles([still.file], "camera", still).catch((e: Error) => toast(`Couldn't add the photo: ${e.message}`)),
  systemCamera: () => $("take").click(),
  upload: () => $("choose").click(),
  teach: () => $("s-teach").click(),
  // Said, so that a camera view that closes unexpectedly tells the person (and us) why.
  closedBy: (why) => toast(`The camera closed: ${why}.`, 6000),
  code: cameraCode,
  fullStream: (label) => {
    if (!settings.fullStreamCameras.includes(label)) setSettings({ ...settings, fullStreamCameras: [...settings.fullStreamCameras, label] });
  },
});
debug.cameraView = cameraView;
debug.camera = cameraView.stats;
$("take-camera").onclick = () => void cameraView.start();

// Upload stays pressed while the phone's photo menu is open: from the tap until a photo is chosen, the menu is
// cancelled, or the page has the focus again (browsers without the cancel event), or failing all of those the
// next tap anywhere.
{
  const key = $("upload"), input = $<HTMLInputElement>("choose");
  const release = () => key.classList.remove("held");
  input.addEventListener("click", () => {
    key.classList.add("held");
    setTimeout(() => {
      addEventListener("focus", release, { once: true });
      document.addEventListener("pointerdown", release, { once: true });
    }, 0);
  });
  input.addEventListener("cancel", release);
  input.addEventListener("change", release);
}
for (const [id, via] of [["take", "take"], ["choose", "choose"]] as const) {
  const input = $<HTMLInputElement>(id);
  input.onchange = () => {
    const files = [...(input.files ?? [])];
    input.value = "";                             // the same photo can be picked again
    if (files.length) {
      void addFiles(files, via).catch((e: Error) => toast(`Couldn't add the photo: ${e.message}`));
    }
  };
}
$("engine-retry").onclick = () => void startEngine();

async function main(): Promise<void> {
  footer();
  renderStrip();
  core.onProgress = engineProgress;
  chainsLoaded = fetch(new URL("../chains.json", location.href)).then(async (r) => {
    if (!r.ok) throw new Error(`chains.json: HTTP ${r.status}`);
    chains = (await r.json()) as Record<string, ChainSpec>;
    renderSettings();
    renderList();                     // the drawings, now with their chains
    renderResult();
  });
  void startPairing().catch((e: Error) => toast(`Pairing: ${e.message}`));
  void retireServiceWorker();
  void startEngine();
  try {
    photos = await PhotoStore.open(indexedDB);
    for (const r of await photos.list()) {
      r.measuredWith ??= null;
      records.set(r.id, r);
      order.push(r.id);
    }
  } catch (e) {
    photos = null;
    toast(`Photos can't be kept on this phone (${(e as Error).message}); they last for this visit.`, 6000);
  }
  recordsLoaded = true;
  sendFullState();
  renderList();
  // Photos the last visit left half measured are measured again.
  for (const id of [...order].reverse()) {
    const r = records.get(id)!;
    if (r.status !== "done" && r.status !== "error") remeasure(id);
  }
  if (order[0]) await select(order[0]);
  renderSettings();
  tidySoon();
}

void main();
