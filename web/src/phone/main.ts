// The phone page (web-build.md, step 2): take or pick a photo, measure it in
// the core worker, fix the pins by touch, then copy, share, download or send
// the curve. The decisions live in the modules beside this one (photos,
// measurer, settings, send, wording), in src/editor/ and src/photo/ (the
// photo editor, shared with the desktop page); this file wires them to the page.

import "./phone.css";
import "../photo/photo.css";
import type { CameraInfo } from "../camera/choose";
import { fovRatio } from "../camera/focal";
import { CoreClient, type PhotoSource } from "../core/client";
import { engineKind, makeCoreWorker } from "../core/make-worker";
import { CoreError, type Commands, type Progress } from "../core/protocol";
import type { ChainSpec, Edits, MeasureOptions, MeasurementDocument, OutputKind, Side } from "../core/types";
import { LiveAnalyser } from "../pipeline/live-client";
import { EditHistory, editsAfterTap } from "../editor/edits";
import type { Pt } from "../editor/viewport";
import { previews } from "../photo/preview";
import { PhotoEditor } from "../photo/view";
import { pageRelaySetting, relayAddress } from "../relay/config";
import { deriveKeys } from "../relay/crypto";
import { deviceName } from "../relay/device";
import { pack } from "../relay/files";
import { RelayLink, type LinkStatus } from "../relay/link";
import { pointsTsv } from "../relay/messages";
import { Outbox } from "../relay/outbox";
import { codeMeaning } from "../camera/qr";
import { forgetPairing, phoneSecret, secretFromHash } from "../relay/pairing";
import { LocalStore } from "../relay/store";
import { CameraView, type Still } from "./camera-view";
import { curveFrom } from "./curve";
import { NO_EXIF, exifOfFile, type Exif } from "./exif";
import { focalOffer, isFront, learn, lensFocal, lensName, offerKey } from "./lenses";
import { Measurer, type Update } from "./measurer";
import { PhotoStore, newPhotoId, type BlobKind, type PhotoFocal, type PhotoRecord } from "./photos";
import { curveKey, delivery, deliveryText, sendOffer } from "./send";
import {
  chainLabel, chainSpec, defaultPins, loadSettings, parseFocal, parsePins, presetKey, saveSettings,
  type Settings, type Shade,
} from "./settings";
import { ENGINE_STAGE, SIDE_TEXT, errorAdvice, errorTitle, failureText, numbers, photoFocalText, readNote, readWarning } from "./wording";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const el = <K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, ...kids: (Node | string)[]) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids);
  return e;
};

const NO_EDITS: Edits = { add: [], remove: [] };

const store = new LocalStore("splinewire-phone:");
let settings: Settings = loadSettings(store);
let chains: Record<string, ChainSpec> = {};

const engine = engineKind();
const core = new CoreClient(() => makeCoreWorker(engine));
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
  return JSON.stringify([settings.savedFocal35, settings.focalOverride35]);
}

function options(id: string): MeasureOptions | null {
  const r = records.get(id);
  if (!r) return null;
  return {
    chain: chainSpec(chains, r.chain, r.pins, r.shade), side: r.side, edits: r.edits,
    saved_focal_35mm: settings.savedFocal35, focal_override_35mm: settings.focalOverride35 ?? r.focal?.focal35 ?? null,
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
      status: "queued", doc: null, failure: null, pixels: made?.pixels ?? null, preview: made?.previewSize ?? null, sent: [],
      measuredWith: null, exif, focal,
      camera: still ? { label: still.camera, still: still.kind, frameChain: still.frameChain, frameSize: still.frameSize, fov: null, ms: Math.round(still.ms) } : null,
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
    measurer.request(id, i === 0);
  }
  renderList();
  if (photos) {
    for (const id of await photos.prune().catch(() => [] as string[])) drop(id);
  }
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
  if (!r || !confirm(`Delete ${r.name} from this phone?`)) return;
  await photos?.delete(id).catch(() => {});
  drop(id);
  void core.call("close", { photo: id }).catch(() => {});
}

function drop(id: string): void {
  records.delete(id);
  order = order.filter((x) => x !== id);
  measurer.forget(id);
  histories.delete(id);
  for (const k of ["original", "preview", "thumb"]) memBlobs.delete(`${id}:${k}`);
  const u = thumbUrls.get(id);
  if (u) URL.revokeObjectURL(u);
  thumbUrls.delete(id);
  if (selected === id) void select(order[0] ?? null);
  renderList();
}

// ---- the engine (core worker) ----

let chainsLoaded: Promise<void>;

function engineProgress(p: Progress): void {
  $("engine-stage").textContent = ENGINE_STAGE[p.stage];
  $("engine-pct").textContent = `${Math.round(p.fraction * 100)}%`;
  $("engine-bar").style.width = `${p.fraction * 100}%`;
}

async function startEngine(): Promise<void> {
  const box = $("engine");
  box.hidden = false;
  box.classList.remove("bad");
  $("engine-retry").hidden = true;
  $("engine-note").textContent = engine === "typescript" ? ""
    : navigator.serviceWorker?.controller
      ? "Loading from this phone, about 10 s. You can take the photo meanwhile."
      : "The first visit downloads about 40 MB, once. You can take the photo meanwhile.";
  try {
    const [loaded] = await Promise.all([core.preload(), chainsLoaded]);
    debug.engine = { readyAt: performance.now(), ms: loaded.ms };
    box.hidden = true;
    const v = loaded.versions;
    footer(v.engine === "typescript"
      ? `engine ${v.splinewire ?? "?"} (TypeScript)`
      : `core ${v.splinewire ?? "?"} · Pyodide ${v.pyodide ?? "?"} · OpenCV ${v.opencv ?? "?"} · ready in ${(loaded.ms / 1000).toFixed(1)} s`);
    measurer.coreReady();
  } catch (e) {
    box.classList.add("bad");
    $("engine-stage").textContent = "Couldn't load the measuring engine.";
    $("engine-pct").textContent = "";
    $("engine-note").textContent = `${(e as Error).message}. Check the connection and try again.`;
    $("engine-retry").hidden = false;
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

/** Production builds only: the service worker keeps the engine's files on the phone. */
async function startServiceWorker(): Promise<void> {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) return;
  // sw.js sits at the site's root, the folder above assets/ where this script
  // is. (Written so that Vite does not take it for an asset of its own to bundle.)
  const root = new URL(["..", ""].join("/"), import.meta.url);
  try {
    const reg = await navigator.serviceWorker.register(new URL("sw.js", root), { scope: root.href });
    if (navigator.serviceWorker.controller) return;
    // First visit: wait a moment for it to take control, so the engine's
    // download goes through it and is kept for the next visit. Only when one
    // is on its way: a browser that blocks it must not delay the engine.
    if (!reg.installing && !reg.waiting && !reg.active) return;
    await Promise.race([
      new Promise((r) => navigator.serviceWorker.addEventListener("controllerchange", r, { once: true })),
      new Promise((r) => setTimeout(r, 3000)),
    ]);
  } catch {
    /* not allowed here (private mode, http on a LAN address): the page works without it */
  }
}

// ---- the selected photo: editor ----

let selected: string | null = null;
let pending: Pt | null = null;
let picking = false;                          // "tap its side" mode
let previewUrl: string | null = null;
const svg = $("photo") as unknown as SVGSVGElement;
const editor = new PhotoEditor(svg, tap);

async function select(id: string | null): Promise<void> {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  editor.clear();
  selected = id;
  pending = null;
  picking = false;
  renderList();
  renderResult();
  const r = id ? records.get(id) : undefined;
  if (!r) return;
  if (r.status !== "queued" && r.status !== "loading" && r.status !== "measuring" && r.measuredWith !== focalKey()) {
    measurer.request(r.id, true);             // the focal length settings changed since it was measured
  }
  const blob = await getBlob(r.id, "preview");
  if (selected !== r.id) return;
  if (blob && r.pixels) previewUrl = URL.createObjectURL(blob);
  editor.setImage(previewUrl, r.pixels);
  editor.frame();
}

function tap(p: Pt, slack: number): void {
  const r = selected ? records.get(selected) : undefined;
  if (!r || measurer.busy(r.id)) return;
  if (picking) {
    picking = false;
    r.side = [p.x, p.y];
    pending = p;
    measurer.request(r.id, true);
    renderResult();
    return;
  }
  const v = r.doc?.view;
  if (!v) return;
  const { edits, mark } = editsAfterTap(v, p, slack);
  history(r.id).commit(v.edits);
  editing.add(r.id);
  r.edits = edits;
  pending = mark;
  measurer.request(r.id, true);
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
  measurer.request(r.id, true);
  renderResult();
};
$("reset").onclick = () => {
  const r = selected ? records.get(selected) : undefined;
  if (!r || measurer.busy(r.id) || !hasEdits(r.edits)) return;
  history(r.id).commit(r.edits);
  editing.add(r.id);
  r.edits = NO_EDITS;
  measurer.request(r.id, true);
  renderResult();
};
$("fit").onclick = () => editor.frame();
addEventListener("resize", () => editor.refit());

const hasEdits = (e: Edits) => e.add.length > 0 || e.remove.length > 0;

// Object side and pins of the selected photo.
for (const b of document.querySelectorAll<HTMLButtonElement>(".sidebox .seg button")) {
  b.onclick = () => {
    const r = selected ? records.get(selected) : undefined;
    if (!r) return;
    const s = b.dataset.side!;
    if (s === "tap") {
      picking = !picking;
      renderResult();
      return;
    }
    picking = false;
    r.side = s as Side;
    settings = { ...settings, side: s as Side };     // new photos start with the side last chosen
    saveSettings(store, settings);
    measurer.request(r.id, true);
    renderResult();
  };
}

let pinsTimer: ReturnType<typeof setTimeout> | undefined;
const photoPins = stepper($("r-pins"), () => (selected ? records.get(selected)?.pins ?? 13 : 13), (n) => {
  const r = selected ? records.get(selected) : undefined;
  if (!r) return;
  r.pins = n;
  settings = { ...settings, pins: n };           // as the current app does: the next photo has as many
  saveSettings(store, settings);
  renderSettings();
  clearTimeout(pinsTimer);                       // a few taps on − or + in a row measure once
  pinsTimer = setTimeout(() => measurer.request(r.id, true), 500);
});

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
  if (!r) return;
  const doc = r.doc;
  const busy = measurer.busy(r.id);
  $("r-name").textContent = r.name;
  const info = doc?.info;
  const focal = info ? (info.focal_35mm ? `${info.focal_35mm} mm in the file` : "no focal length in the file") : "";
  const size = r.pixels ? `${r.pixels.width}×${r.pixels.height}` : "";
  const via = r.via === "camera" ? (r.camera?.still === "frame" ? "page camera, video frame" : "page camera")
    : r.via === "take" ? "system camera" : "library";
  const fov = r.camera?.fov != null ? `live view scale ${r.camera.fov.toFixed(3)}` : "";
  $("r-meta").textContent = [size, focal, `${chainLabel(r.chain)}`, via, fov].filter(Boolean).join(" · ");
  const [tag, label] = tagOf(r);
  const t = $("r-tag");
  t.className = `tag ${tag}`;
  t.textContent = label;
  void thumbInto($<HTMLImageElement>("r-thumb"), r.id);

  const ok = r.status === "done" && !!doc?.ok;
  const status = $("r-status");
  status.textContent = STATUS_WORDS[r.status] ||
    (ok ? `Measured: ${doc!.result!.curve_points} points. ${paired() ? "Copy them, or send them to the computer." : "Copy them, share them or download them."}` : "");

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

  const v = doc?.view;
  $("editor").hidden = !r.pixels && !v;
  svg.classList.toggle("wait", busy);
  svg.classList.toggle("picking", picking);
  editor.show(v ?? null, { pending });
  const edited = !!v && hasEdits(v.edits);
  $<HTMLButtonElement>("undo").disabled = busy || !history(r.id).canUndo;
  $<HTMLButtonElement>("reset").disabled = busy || !edited;
  $("hint").textContent = picking
    ? "Tap the photo on the object's side of the chain."
    : v ? "Tap a pin to remove it, a dot to add one. Drag to move, pinch to zoom." : r.pixels ? "" : "This browser can't show this photo, but it can still measure it.";

  const sideKey = Array.isArray(r.side) ? "tap" : r.side;
  for (const b of document.querySelectorAll<HTMLButtonElement>(".sidebox .seg button")) {
    b.setAttribute("aria-pressed", String(picking ? b.dataset.side === "tap" : b.dataset.side === sideKey));
  }
  const res = doc?.result;
  $("side-now").textContent = res ? `${SIDE_TEXT[res.object_side]}${Array.isArray(r.side) ? ", from your tap" : ""}` : "";
  photoPins.value = String(r.pins);

  const focalDetail = res && r.focal && res.focal_source === "override" && settings.focalOverride35 === null ? photoFocalText(r.focal) : undefined;
  $("numbers").replaceChildren(...(res ? numbers(res, r.pins, focalDetail) : []).map(([k, val, s, warn]) =>
    el("div", { className: warn ? "chk" : "" }, el("dt", { textContent: k }), el("dd", {}, val, el("span", { className: "s", textContent: s })))));
  const item = (x: { title: string; text: string }) => el("li", {}, el("b", { textContent: x.title }), x.text);
  $("warnings").replaceChildren(...(res?.warnings ?? []).map((w) => readWarning(w)).map(item));
  $("notes").replaceChildren(...(res?.notes ?? []).map((n) => readNote(n)).map(item));

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

const DOWNLOADS: [OutputKind, string][] = [["dxf", "DXF"], ["svg", "SVG 1:1"], ["csv", "CSV"], ["json", "JSON"], ["fusion_csv", "Fusion CSV"]];

// ---- the list of photos ----

const thumbUrls = new Map<string, string>();

async function thumbInto(image: HTMLImageElement, id: string): Promise<void> {
  let url = thumbUrls.get(id);
  if (!url) {
    const blob = await getBlob(id, "thumb");
    if (!blob || !records.has(id)) {
      image.removeAttribute("src");
      return;
    }
    url = thumbUrls.get(id) ?? URL.createObjectURL(blob);
    thumbUrls.set(id, url);
  }
  if (image.getAttribute("src") !== url) image.src = url;
}

function renderList(): void {
  $("photos-box").hidden = order.length === 0;
  const list = $("photos");
  list.replaceChildren(...order.map((id) => {
    const r = records.get(id)!;
    const [tag, label] = tagOf(r);
    const th = el("img", { className: "th", alt: "" });
    void thumbInto(th, id);
    const when = new Date(r.added).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    const open = el("button", { type: "button", className: "open" }, th,
      el("span", { className: "who" }, el("span", { className: "name", textContent: r.name }),
        el("span", { className: "meta mono", textContent: `${when} · ${r.pins} pins` })),
      el("span", { className: `tag ${tag}`, textContent: label }));
    open.setAttribute("aria-label", `Open ${r.name}, ${label}`);
    open.onclick = () => {
      void select(id);
      $("result").scrollIntoView({ behavior: "smooth", block: "start" });
    };
    const del = el("button", { type: "button", className: "del", textContent: "×" });
    del.setAttribute("aria-label", `Delete ${r.name}`);
    del.onclick = () => void deletePhoto(id);
    const li = el("li", { className: id === selected ? "on" : "" }, open, del);
    li.dataset.id = id;
    li.dataset.status = r.status;
    return li;
  }));
}

// ---- copy, share, download ----

async function outputsOf(id: string): Promise<Commands["outputs"]["result"]> {
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

function fileOf(out: Commands["outputs"]["result"], kind: OutputKind): File {
  return new File([out.files[kind]], out.names[kind] ?? `curve.${kind}`, { type: TYPES[kind] ?? "text/plain" });
}

$("share").onclick = async () => {
  const id = selected;
  if (!id) return;
  try {
    const out = await outputsOf(id);
    // Not every phone shares every file type (Chrome on Android refuses .dxf):
    // try the DXF with the CSV, then the CSV alone.
    const tries = [[fileOf(out, "csv"), fileOf(out, "dxf")], [fileOf(out, "csv")]];
    const files = tries.find((f) => navigator.canShare?.({ files: f }));
    if (!files) throw new Error("this phone can't share these files; use the downloads");
    await navigator.share({ files, title: records.get(id)?.name ?? "Curve" });
  } catch (e) {
    if ((e as Error).name !== "AbortError") toast(`Couldn't share: ${(e as Error).message}`);
  }
};

async function download(id: string, kind: OutputKind): Promise<void> {
  try {
    const out = await outputsOf(id);
    const url = URL.createObjectURL(fileOf(out, kind));
    const a = el("a", { href: url, download: out.names[kind] ?? `curve.${kind}` });
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
/** Names of curves delivered this visit, for the list (the outbox forgets them once acknowledged). */
const sentNames = new Map<string, string>();

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
  const hello = () => void link?.send({ type: "hello", role: "phone", device: deviceName(navigator.userAgent), app: import.meta.env.APP_VERSION });
  link = new RelayLink({
    relayUrl, keys, role: "phone",
    events: {
      status: (s) => {
        const before = linkStatus;
        linkStatus = s;
        debug.link.push({ ...s, t: Math.round(performance.now()) });
        outbox?.onStatus(s, before);
        if (s.state === "open" && (before.state !== "open" || s.peers > before.peers)) hello();
        renderPairing();
        renderResult();
      },
      message: (m) => outbox?.onMessage(m),
    },
  });
  outbox = new Outbox(store, link, outboxChanged);
  link.start();
  renderPairing();
  renderResult();
}
const history_ = window.history;

function outboxChanged(): void {
  if (!outbox) return;
  const delivered = new Set(outbox.delivered);
  for (const r of records.values()) {
    let changed = false;
    for (const s of r.sent) {
      if (!s.delivered && delivered.has(s.id)) {
        s.delivered = true;
        changed = true;
      }
    }
    if (changed) persist(r);
  }
  renderPairing();
  if (selected) renderResult();
}

function renderSend(r: PhotoRecord, enabled: boolean): void {
  const offer = sendOffer(paired(), r.doc, r.sent);
  const button = $<HTMLButtonElement>("send");
  const note = $("send-note");
  button.hidden = offer.kind === "unpaired" || offer.kind === "nothing";
  button.disabled = !enabled;
  note.textContent = "";
  if (offer.kind === "send") button.textContent = "Send to computer";
  else if (offer.kind === "resend") {
    button.textContent = "Send again, as a new curve";
    note.textContent = "You changed this photo after sending it. The computer has the earlier curve; sending adds this one beside it.";
  } else if (offer.kind === "sent") {
    button.hidden = true;
    note.textContent = deliveryText(delivery(offer.curve, outbox?.items ?? [], linkStatus));
  } else if (offer.kind === "unpaired" && r.status === "done") {
    note.textContent = "To send curves to a computer, pair this phone with it (see Computer below).";
  }
}

$("send").onclick = async () => {
  const r = selected ? records.get(selected) : undefined;
  const doc = r?.doc;
  if (!r || !doc?.ok || !outbox) return;
  const offer = sendOffer(true, doc, r.sent);
  if (offer.kind !== "send" && offer.kind !== "resend") return;
  const button = $<HTMLButtonElement>("send");
  button.disabled = true;
  try {
    const out = await outputsOf(r.id);
    const spec = chainSpec(chains, r.chain, r.pins, r.shade);
    const curve = curveFrom(doc, { chainName: chainLabel(r.chain), pitch_mm: spec.pitch_mm, half_width_mm: spec.half_width_mm, n_pins: r.pins },
      await pack(out.files));
    if (!curve) return;
    r.sent.push({ id: curve.id, key: curveKey(doc)!, at: curve.measured_at, delivered: false });
    sentNames.set(curve.id, r.name);
    persist(r);
    await outbox.send(curve);
    toast(offer.kind === "resend" ? "Sent as a new curve. The computer keeps the earlier one too." : "Sent to the computer.");
  } catch (e) {
    toast(`Couldn't send: ${(e as Error).message}`);
  } finally {
    renderResult();
    renderPairing();
  }
};

function renderPairing(): void {
  const lamp = $("lamp");
  const chip = $("link-text");
  const text = $("pair-text");
  $("unpair").hidden = !secret;
  if (!secret) {
    lamp.className = "lamp";
    chip.textContent = "Not paired";
    text.textContent = "Not paired. To send curves to a computer, open the Spline Wire site on the computer and point Take photo's live camera (or the phone's Camera app) at the code it shows. Copy points, Share and the downloads work without it.";
  } else if (!relayUrl) {
    lamp.className = "lamp bad";
    chip.textContent = "No relay";
    text.textContent = "Paired, but this build of the site has no relay address, so curves can't be sent. Copy points, Share and the downloads still work.";
  } else {
    const s = linkStatus;
    const [cls, short, long] =
      s.state === "open" && s.peers > 0 ? ["ok", "Computer connected", "Paired. The computer's page is open: curves you send arrive at once."]
      : s.state === "open" ? ["chk", "Computer not open", "Paired. The computer's page is not open: curves you send wait here until it is."]
      : s.state === "waiting" ? ["bad", "Can't reach relay", `Paired, but the relay can't be reached${s.problem ? ` (${s.problem})` : ""}. Curves you send wait here and go when it is back.`]
      : ["", "Connecting…", "Paired. Connecting to the relay…"];
    lamp.className = `lamp ${cls}`;
    chip.textContent = short;
    text.textContent = long;
  }
  // Curves still on their way, then those delivered during this visit.
  const items = outbox?.items ?? [];
  const rows = [
    ...items.map((i) => ({ id: i.curve.id, name: i.curve.photo, at: i.curve.measured_at, d: delivery({ id: i.curve.id, key: "", at: "", delivered: false }, items, linkStatus) })),
    ...(outbox?.delivered ?? []).slice(-10).reverse().map((id) => ({ id, name: sentNames.get(id) ?? nameOfSent(id), at: "", d: "delivered" as const })),
  ];
  $("outbox").replaceChildren(...rows.map((row) => {
    const state = el("div", { className: `state${row.d === "delivered" ? " ok" : ""}`, textContent: deliveryText(row.d) });
    const li = el("li", {}, el("div", { className: "what" }, el("b", { textContent: row.name }), state));
    li.dataset.curve = row.id;
    li.dataset.state = row.d;
    if (row.d !== "delivered") {
      const b = el("button", { type: "button", className: "btn plain", textContent: "Don't send" });
      b.setAttribute("aria-label", `Don't send the curve from ${row.name}`);
      b.onclick = () => {
        outbox?.forget(row.id);
        renderPairing();
        if (selected) renderResult();
      };
      li.append(b);
    }
    return li;
  }));
}

function nameOfSent(id: string): string {
  for (const r of records.values()) if (r.sent.some((s) => s.id === id)) return r.name;
  return "a curve";
}

$("unpair").onclick = () => {
  if (!confirm("Forget this computer? Curves not yet delivered are dropped. To pair again, scan the computer's code.")) return;
  for (const i of [...(outbox?.items ?? [])]) outbox?.forget(i.curve.id);
  link?.stop();
  forgetPairing(store);
  link = null;
  outbox = null;
  secret = null;
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

function renderSettings(): void {
  const select = $<HTMLSelectElement>("s-chain");
  if (!select.options.length || select.options.length !== Object.keys(chains).length) {
    select.replaceChildren(...Object.keys(chains).map((k) => new Option(chainLabel(k), k)));
  }
  select.value = presetKey(chains, settings.chain);
  $<HTMLSelectElement>("s-shade").value = settings.shade;
  settingsPins.value = String(defaultPins(settings, chains));
  const focal = $<HTMLInputElement>("s-focal");
  if (document.activeElement !== focal) focal.value = settings.savedFocal35?.toString() ?? "";
  const override = $<HTMLInputElement>("s-override");
  if (document.activeElement !== override) override.value = settings.focalOverride35?.toString() ?? "";
  const offer = focalOffer(settings);
  $("s-offer").hidden = !offer;
  if (offer) {
    $("s-offer-text").textContent = `Use ${offer.focal35} mm from your last photo (${offer.name}) for photos without camera data?`;
  }
  $("s-lenses").replaceChildren(...settings.lenses.map((l) => el("li", {},
    el("span", { textContent: `${l.name}${l.ultraWide ? " (ultra-wide)" : ""}` }), el("span", { textContent: `${l.focal35} mm` }))));
  $("s-teach-note").hidden = settings.lenses.length > 0;
}

function setSettings(next: Settings, remeasure = false): void {
  settings = next;
  saveSettings(store, settings);
  renderSettings();
  if (remeasure && selected) {
    const r = records.get(selected);
    if (r && (r.status === "done" || r.status === "error")) measurer.request(r.id, true);
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
  if (r && (r.shade ?? "light") !== shade) {
    r.shade = shade;
    measurer.request(r.id, true);
    renderResult();
  }
  toast(shade === "dark" ? "Measuring dark dots or rims on light links." : "Measuring light dots or rims on dark links.");
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
$("s-offer-yes").onclick = () => {
  const offer = focalOffer(settings);
  if (offer) setSettings({ ...settings, savedFocal35: offer.focal35 }, true);
};
$("s-offer-no").onclick = () => {
  const offer = focalOffer(settings);
  if (offer) setSettings({ ...settings, declinedOffer: offerKey(offer) });
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
  const made = await previews(file);
  setSettings(learn(settings, exif, made?.pixels ?? { width: 0, height: 0 }));
  toast(`Learned ${exif.focal35} mm for ${lensName(exif)}.`, 5000);
  // Photos already taken with that lens, which lacked it, are measured with it now.
  for (const r of records.values()) {
    if (r.focal || !r.exif || r.exif.focal35 !== null) continue;
    const known = lensFocal(settings, r.exif);
    if (!known) continue;
    r.focal = { focal35: known.focal35, why: "lens", lens: known.lens };
    r.measuredWith = null;
    persist(r);
    if (r.id === selected && (r.status === "done" || r.status === "error")) measurer.request(r.id, true);
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
  systemCamera: () => {
    setMode("upload");
    $("take").click();
  },
  upload: () => setMode("upload"),
  teach: () => {
    ($("settings") as HTMLDetailsElement).open = true;     // a closed <details> may not pass the click on
    $("s-teach").click();
  },
  // Said, so that a camera view that closes unexpectedly tells the person (and us) why.
  closedBy: (why) => toast(`The camera closed: ${why}.`, 6000),
  code: cameraCode,
});
debug.cameraView = cameraView;
debug.camera = cameraView.stats;
$("take-camera").onclick = () => void cameraView.start();

/** Live camera or Upload a photo, remembered. Upload releases the camera
 * entirely; choosing Live opens it (a tap is the right moment to ask for it). */
function setMode(mode: Settings["photoMode"], open = false): void {
  if (mode === "upload") cameraView.close();
  if (mode !== settings.photoMode) setSettings({ ...settings, photoMode: mode });
  renderMode();
  if (mode === "live" && open) void cameraView.start();
}

function renderMode(): void {
  const live = settings.photoMode === "live";
  $("live-box").hidden = !live;
  $("upload-box").hidden = live;
  $("mode-live").setAttribute("aria-pressed", String(live));
  $("mode-upload").setAttribute("aria-pressed", String(!live));
}
$("mode-live").onclick = () => setMode("live", true);
$("mode-upload").onclick = () => setMode("upload");

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
  renderMode();
  core.onProgress = engineProgress;
  chainsLoaded = fetch(new URL("../chains.json", location.href)).then(async (r) => {
    if (!r.ok) throw new Error(`chains.json: HTTP ${r.status}`);
    chains = (await r.json()) as Record<string, ChainSpec>;
    renderSettings();
  });
  void startPairing().catch((e: Error) => toast(`Pairing: ${e.message}`));
  await startServiceWorker();
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
  renderList();
  // Photos the last visit left half measured are measured again.
  for (const id of [...order].reverse()) {
    const r = records.get(id)!;
    if (r.status !== "done" && r.status !== "error") measurer.request(id);
  }
  if (order[0]) await select(order[0]);
  renderSettings();
}

void main();
