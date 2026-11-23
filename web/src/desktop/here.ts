// Measuring a photo on this computer (decision D9): a photo dropped on the
// desktop page, or picked with its button, measured by the same core worker
// as on the phone, its pins fixed with the mouse, then added to the page's
// list of curves as one "measured here".
//
// This module and everything it pulls in (the core client and its worker,
// the photo editor) load only when a photo is dropped: most visits only
// receive curves, and must not pay for the engine. main.ts imports it with
// import(), never statically.
//
// The logic is the phone page's: its Measurer (one measurement at a time,
// coalesced), its settings helpers, its wording (in mouse words), curveFrom
// and pack(), the shared photo editor (src/photo/). This file only lays it
// out for a desktop window.

import "../photo/photo.css";
import "./here.css";
import { CoreClient } from "../core/client";
import { makeCoreWorker } from "../core/make-worker";
import { CoreError, type Commands, type Failure, type Progress } from "../core/protocol";
import type { ChainSpec, Edits, MeasureOptions, MeasurementDocument, Side } from "../core/types";
import { EditHistory, editsAfterTap } from "../editor/edits";
import type { Pt } from "../editor/viewport";
import { curveFrom } from "../phone/curve";
import { Measurer, type Update } from "../phone/measurer";
import type { Status } from "../phone/photos";
import { SHADE_TEXT, chainLabel, chainSpec, defaultPins, loadSettings, parseFocal, parsePins, presetKey, saveSettings, type Settings,
  type Shade } from "../phone/settings";
import { ENGINE_STAGE, SIDE_TEXT, errorAdvice, errorTitle, failureText, numbers, readNote, readWarning } from "../phone/wording";
import { greyCopy, previews } from "../photo/preview";
import { PhotoEditor } from "../photo/view";
import { pack } from "../relay/files";
import type { Curve } from "../relay/messages";
import type { KeyValueStore } from "../relay/store";
import { esc } from "./wording";

// ---- the engine: one per page, loaded on the first photo ----------------

export interface Engine {
  core: CoreClient;
  /** Every progress event since the engine was made, with its time (ms since the page opened). */
  progress: (Progress & { t: number })[];
  chains: Promise<Record<string, ChainSpec>>;
  /** Load the engine and the presets; the same promise until a load fails or the worker dies. */
  load(): Promise<Commands["load"]["result"]>;
  /** The worker died: the next load() starts a new one. */
  lost(): void;
  listen(fn: (p: Progress) => void): () => void;
}

let shared: Engine | null = null;

export function engine(siteRoot: string): Engine {
  if (shared) return shared;
  const core = new CoreClient(() => makeCoreWorker());
  const progress: Engine["progress"] = [];
  const listeners = new Set<(p: Progress) => void>();
  core.onProgress = (p) => {
    progress.push({ ...p, t: Math.round(performance.now()) });
    for (const fn of listeners) fn(p);
  };
  let chains: Promise<Record<string, ChainSpec>> | null = null;
  const presets = () => (chains ??= fetch(new URL("chains.json", siteRoot)).then(async (r) => {
    if (!r.ok) throw new Error(`chains.json: HTTP ${r.status}`);
    return (await r.json()) as Record<string, ChainSpec>;
  }).catch((e: Error) => {
    chains = null;
    throw e;
  }));
  let loading: Promise<Commands["load"]["result"]> | null = null;
  shared = {
    core,
    progress,
    get chains() {
      return presets();
    },
    load() {
      loading ??= Promise.all([core.preload(), presets()]).then(([r]) => r).catch((e: Error) => {
        loading = null;
        throw e;
      });
      return loading;
    },
    lost() {
      loading = null;
    },
    listen(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
  return shared;
}

// ---- one photo being measured ---------------------------------------------

export interface DraftHost {
  siteRoot: string;
  /** The desktop page's store: the chain, pins, side and focal length chosen here are kept in it. */
  store: KeyValueStore;
  toast(message: string): void;
  /** The person added the measured curve to the list. */
  add(curve: Curve): void;
  /** The person discarded the photo. */
  discard(): void;
  /** The draft's name or state changed: redraw its entry in the list. */
  changed(): void;
}

/** What the end-to-end test and the console can read. */
export const timings: { name: string; status: Status; ms: number; sinceDrop: number }[] = [];

const NO_EDITS: Edits = { add: [], remove: [] };
let serial = 0;

export function openDraft(file: File, host: DraftHost): Draft {
  return new Draft(file, host);
}

export class Draft {
  /** Shown in #main; kept while another curve is on show, so the view stays as the person left it. */
  readonly root: HTMLElement;
  readonly name: string;
  status: Status = "loading";
  private readonly id = `here-${Date.now().toString(36)}-${++serial}`;
  private readonly dropped = performance.now();
  private readonly engine: Engine;
  private readonly measurer: Measurer;
  private readonly editor: PhotoEditor;
  private readonly history = new EditHistory();
  private settings: Settings;
  private chains: Record<string, ChainSpec> = {};
  private chain: string;
  private pins: number;
  private side: Side | [number, number];
  private edits: Edits = NO_EDITS;
  private doc: MeasurementDocument | null = null;
  private failure: Failure | null = null;
  private engineError: string | null = null;
  private editing = false;
  private picking = false;
  private pending: Pt | null = null;
  private adding = false;
  private closed = false;
  private previewUrl: string | null = null;
  private readonly unlisten: () => void;

  constructor(private readonly file: File, private readonly host: DraftHost) {
    this.name = file.name || "photo.jpg";
    this.settings = loadSettings(host.store);
    this.chain = this.settings.chain;
    this.pins = this.settings.pins ?? 13;
    this.side = this.settings.side;
    this.root = document.createElement("div");
    this.root.className = "here";
    this.root.innerHTML = layout(this.name);
    this.engine = engine(host.siteRoot);
    this.editor = new PhotoEditor(this.q<SVGSVGElement>(".photo-svg"), (p, slack) => this.tap(p, slack));
    this.measurer = new Measurer({
      core: this.engine.core,
      options: () => (this.closed ? null : this.options()),
      source: () => ({ name: this.name, bytes: () => this.file.arrayBuffer() }),
      update: (_id, u) => this.onUpdate(u),
    });
    this.unlisten = this.engine.listen((p) => this.progress(p));
    this.wire();
    this.render();
    void this.showPhoto();
    this.measurer.request(this.id);
    void this.startEngine();
  }

  /** The person sees it again (after looking at another curve): sizes may have changed. */
  shown(): void {
    this.editor.refit();
  }

  refit(): void {
    this.editor.refit();
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.unlisten();
    this.measurer.forget(this.id);
    if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
    void this.engine.core.call("close", { photo: this.id }).catch(() => {});
  }

  /** One line for the list. */
  get state(): string {
    if (this.engineError) return "the engine didn't load";
    return { loading: "loading the engine…", queued: "waiting…", measuring: "measuring…", done: "measured, not added yet",
      error: "couldn't measure it" }[this.status];
  }

  // ---- the engine and the measurement ----

  private async startEngine(): Promise<void> {
    this.engineError = null;
    this.render();
    try {
      await this.engine.load();
      this.chains = await this.engine.chains;
      this.chain = presetKey(this.chains, this.chain);
      this.pins = this.settings.pins ?? defaultPins(this.settings, this.chains);
      if (this.closed) return;
      this.measurer.coreReady();
    } catch (e) {
      this.engineError = (e as Error).message;
    }
    this.render();
    this.host.changed();
  }

  private options(): MeasureOptions {
    return {
      chain: chainSpec(this.chains, this.chain, this.pins, this.settings.shade), side: this.side, edits: this.edits,
      saved_focal_35mm: this.settings.savedFocal35,
    };
  }

  private onUpdate(u: Update): void {
    if (this.closed) return;
    this.status = u.status;
    if (u.status === "done" || u.status === "error") {
      timings.push({ name: this.name, status: u.status, ms: Math.round(u.ms ?? 0), sinceDrop: Math.round(performance.now() - this.dropped) });
      if (u.doc) {
        this.doc = u.doc;
        this.failure = null;
      } else if (u.failure) {
        this.failure = u.failure;
        // An edit the core refused: back to what was last measured.
        if (this.editing) {
          this.history.rollback();
          this.edits = this.doc?.view?.edits ?? NO_EDITS;
        }
        if (u.failure.type === "WorkerError") {
          this.engine.lost();
          this.measurer.coreLoading();
          void this.startEngine();
        }
      }
      this.editing = false;
      this.pending = null;
    }
    this.render();
    this.host.changed();
  }

  private remeasure(): void {
    this.measurer.request(this.id, true);
    this.render();
  }

  private tap(p: Pt, slack: number): void {
    if (this.measurer.busy(this.id)) return;
    if (this.picking) {
      this.picking = false;
      this.side = [p.x, p.y];
      this.pending = p;
      this.remeasure();
      return;
    }
    const v = this.doc?.view;
    if (!v) return;
    const { edits, mark } = editsAfterTap(v, p, slack);
    this.history.commit(v.edits);
    this.editing = true;
    this.edits = edits;
    this.pending = mark;
    this.remeasure();
  }

  private async showPhoto(): Promise<void> {
    const made = await previews(this.file);
    // The desktop page shows photos grey: the pins and the curve stand out (the core measures the original).
    const grey = made && (await greyCopy(made.preview));
    if (this.closed) return;
    if (made) this.previewUrl = URL.createObjectURL(grey ?? made.preview);
    this.editor.setImage(this.previewUrl, made?.pixels ?? null);
    this.editor.frame();
    this.render();
  }

  private async add(): Promise<void> {
    const doc = this.doc;
    if (!doc?.ok || this.adding || this.measurer.busy(this.id)) return;
    this.adding = true;
    this.render();
    try {
      const out = await this.outputs();
      const spec = chainSpec(this.chains, this.chain, this.pins, this.settings.shade);
      const curve = curveFrom(doc, { chainName: chainLabel(this.chain), pitch_mm: spec.pitch_mm, half_width_mm: spec.half_width_mm,
        n_pins: this.pins }, await pack(out.files));
      if (!curve) return;
      this.close();
      this.host.add(curve);
    } catch (e) {
      this.host.toast(`Couldn't add it: ${(e as Error).message}`);
    } finally {
      this.adding = false;
      if (!this.closed) this.render();
    }
  }

  /** The files of the last measurement; the photo is opened and measured again if the worker let it go. */
  private async outputs(): Promise<Commands["outputs"]["result"]> {
    const core = this.engine.core;
    try {
      return await core.call("outputs", { photo: this.id });
    } catch (err) {
      if (!(err instanceof CoreError) || err.failure.kind !== "closed") throw err;
    }
    await core.measure(this.id, this.options(), { name: this.name, bytes: () => this.file.arrayBuffer() });
    return core.call("outputs", { photo: this.id });
  }

  private saveSettings(change: Partial<Settings>): void {
    this.settings = { ...this.settings, ...change };
    saveSettings(this.host.store, this.settings);
  }

  // ---- the page ----

  private q<T extends Element = HTMLElement>(sel: string): T {
    return this.root.querySelector(sel) as T;
  }

  private wire(): void {
    this.root.addEventListener("click", (ev) => {
      const b = (ev.target as Element).closest<HTMLElement>("[data-here]");
      if (!b || (b as HTMLButtonElement).disabled) return;
      const act = b.dataset.here!;
      if (act === "discard") {
        this.close();
        this.host.discard();
      } else if (act === "add") {
        void this.add();
      } else if (act === "retry") {
        void this.startEngine();
      } else if (act === "fit") {
        this.editor.frame();
      } else if (act === "undo") {
        if (this.measurer.busy(this.id)) return;
        const e = this.history.undo();
        if (!e) return;
        this.edits = e;
        this.remeasure();
      } else if (act === "reset") {
        if (this.measurer.busy(this.id) || !hasEdits(this.edits)) return;
        this.history.commit(this.edits);
        this.editing = true;
        this.edits = NO_EDITS;
        this.remeasure();
      } else if (act === "inside" || act === "outside") {
        this.picking = false;
        this.side = act;
        this.saveSettings({ side: act });
        this.remeasure();
      } else if (act === "tap") {
        this.picking = !this.picking;
        this.render();
      } else if (act === "fewer" || act === "more") {
        this.setPins(this.pins + (act === "more" ? 1 : -1));
      }
    });
    this.q<HTMLSelectElement>("[data-field=chain]").addEventListener("change", (ev) => {
      this.chain = (ev.target as HTMLSelectElement).value;
      this.pins = this.chains[this.chain]?.n_pins ?? this.pins;           // a new chain brings its own pin count
      this.saveSettings({ chain: this.chain, pins: null });
      this.remeasure();
    });
    this.q<HTMLSelectElement>("[data-field=shade]").addEventListener("change", (ev) => {
      this.saveSettings({ shade: (ev.target as HTMLSelectElement).value as Shade });
      this.remeasure();
    });
    this.q<HTMLInputElement>("[data-field=pins]").addEventListener("change", (ev) => this.setPins(Number((ev.target as HTMLInputElement).value)));
    this.q<HTMLInputElement>("[data-field=focal]").addEventListener("change", (ev) => {
      const input = ev.target as HTMLInputElement;
      const v = parseFocal(input.value);
      this.q("[data-out=focal-error]").textContent = v === "invalid"
        ? "A focal length is a number of millimetres between 5 and 500 (35 mm equivalent), or empty." : "";
      if (v === "invalid") return;
      this.saveSettings({ savedFocal35: v });
      this.remeasure();
    });
  }

  private setPins(n: number): void {
    const v = parsePins(String(n));
    if (v === "invalid" || v === this.pins) {
      this.render();
      return;
    }
    this.pins = v;
    this.saveSettings({ pins: v });
    this.remeasure();
  }

  private progress(p: Progress): void {
    this.q("[data-out=stage]").textContent = ENGINE_STAGE[p.stage];
    this.q("[data-out=pct]").textContent = `${Math.round(p.fraction * 100)}%`;
    this.q<HTMLElement>("[data-out=bar]").style.width = `${p.fraction * 100}%`;
  }

  private render(): void {
    if (this.closed) return;
    const doc = this.doc;
    const res = doc?.result ?? null;
    const busy = this.measurer.busy(this.id);
    const ok = this.status === "done" && !!doc?.ok;
    const loadingEngine = this.status === "loading" || !!this.engineError;

    const engineBox = this.q("[data-out=engine]");
    engineBox.hidden = !loadingEngine;
    engineBox.classList.toggle("bad", !!this.engineError);
    this.q("[data-here=retry]").hidden = !this.engineError;
    if (this.engineError) {
      this.q("[data-out=stage]").textContent = "Couldn't load the measuring engine.";
      this.q("[data-out=pct]").textContent = "";
      this.q("[data-out=engine-note]").textContent = `${this.engineError}. Check the connection and try again.`;
    } else {
      this.q("[data-out=engine-note]").textContent =
        "About 10 s, once per visit. The photo is measured in this browser: it is not uploaded anywhere.";
    }

    this.q("[data-out=status]").textContent = {
      loading: "Waiting for the measuring engine…", queued: "Waiting…", measuring: "Measuring…",
      done: ok ? `Measured: ${res!.curve_points} points.` : "", error: "",
    }[this.status];

    // What went wrong: a person's error as is, under a plain heading; a bug with its type.
    const problem = this.q("[data-out=problem]");
    if (this.failure) {
      const f = failureText(this.failure);
      problem.innerHTML = `<b>${esc(f.title)}</b><div class="raw">${esc(f.text)}</div>` +
        (f.bug ? `<div class="advice">Please report it with this text. Try the photo again.</div>` : "");
    } else if (doc && !doc.ok && doc.error) {
      const fixable = !!doc.view && (doc.view.pins.length + doc.view.unused.length) > 0;
      const advice = errorAdvice(doc.error, fixable, "mouse");
      problem.innerHTML = `<b>${esc(errorTitle(doc.error))}</b><div class="raw">${esc(doc.error)}</div>` +
        (advice ? `<div class="advice">${esc(advice)}</div>` : "");
    } else {
      problem.innerHTML = "";
    }
    problem.hidden = !problem.innerHTML;

    const v = doc?.view ?? null;
    const svg = this.q<SVGSVGElement>(".photo-svg");
    svg.classList.toggle("wait", busy);
    svg.classList.toggle("picking", this.picking);
    this.editor.show(v, { pending: this.pending });
    this.q<HTMLButtonElement>("[data-here=undo]").disabled = busy || !this.history.canUndo;
    this.q<HTMLButtonElement>("[data-here=reset]").disabled = busy || !v || !hasEdits(v.edits);
    this.q("[data-out=hint]").textContent = this.picking
      ? "Click the photo on the object's side of the chain."
      : v ? "Click a pin to remove it, a dot to add one. Drag to move, wheel to zoom." : "";

    const sideKey = Array.isArray(this.side) ? "tap" : this.side;
    for (const b of this.root.querySelectorAll<HTMLButtonElement>(".seg button")) {
      b.setAttribute("aria-pressed", String(this.picking ? b.dataset.here === "tap" : b.dataset.here === sideKey));
    }
    this.q("[data-out=side-now]").textContent = res ? `${SIDE_TEXT[res.object_side]}${Array.isArray(this.side) ? ", from your click" : ""}` : "";

    const chainSel = this.q<HTMLSelectElement>("[data-field=chain]");
    if (chainSel.options.length !== Object.keys(this.chains).length) {
      chainSel.replaceChildren(...Object.keys(this.chains).map((k) => new Option(chainLabel(k), k)));
    }
    chainSel.value = this.chain;
    this.q<HTMLSelectElement>("[data-field=shade]").value = this.settings.shade;
    const pinsInput = this.q<HTMLInputElement>("[data-field=pins]");
    if (document.activeElement !== pinsInput) pinsInput.value = String(this.pins);
    const focal = this.q<HTMLInputElement>("[data-field=focal]");
    if (document.activeElement !== focal) focal.value = this.settings.savedFocal35?.toString() ?? "";

    this.q("[data-out=nums]").innerHTML = (res ? numbers(res, this.pins) : []).map(([k, val, s, warn]) =>
      `<div class="cell${warn ? " warn" : ""}"><div class="k">${esc(k)}</div><div class="v">${esc(val)}</div><div class="s">${esc(s)}</div></div>`).join("");
    const notes = [...(res?.warnings ?? []).map((w) => ({ level: "warn", ...readWarning(w, "mouse") })),
      ...(res?.notes ?? []).map((n) => ({ level: "info", ...readNote(n, "mouse") }))];
    this.q("[data-out=notes]").innerHTML = notes.map((r) => `<div class="note ${r.level}"><b>${esc(r.title)}</b>${esc(r.text)}</div>`).join("");

    const add = this.q<HTMLButtonElement>("[data-here=add]");
    add.disabled = !ok || busy || this.adding;
  }
}

const hasEdits = (e: Edits) => e.add.length > 0 || e.remove.length > 0;

function layout(name: string): string {
  return `<section class="view" aria-labelledby="here-title">
    <div class="head"><h2 id="here-title">${esc(name)}</h2>
      <span class="meta">measuring on this computer · not in the list yet</span>
      <span class="spacer"></span>
      <button type="button" class="btn plain small" data-here="discard">Discard</button></div>
    <div class="engine" data-out="engine" role="status" aria-live="polite" hidden>
      <div class="engine-text"><span data-out="stage">Getting the measuring engine ready…</span><span class="mono" data-out="pct"></span></div>
      <div class="bar"><div data-out="bar"></div></div>
      <p data-out="engine-note"></p>
      <button type="button" class="btn small" data-here="retry" hidden>Try again</button>
    </div>
    <div class="problem" data-out="problem" role="alert" hidden></div>
    <div class="stage">
      <svg class="photo-svg" role="img" aria-label="The photo with the measured chain"><image x="0" y="0" preserveAspectRatio="none"/><g class="ov-layer"></g></svg>
    </div>
    <div class="tools">
      <button type="button" class="btn small" data-here="undo" aria-label="Undo the last pin change">Undo</button>
      <button type="button" class="btn small" data-here="reset" aria-label="Remove all pin changes">Reset</button>
      <span class="hint" data-out="hint"></span>
      <span class="spacer"></span>
      <button type="button" class="btn small" data-here="fit" aria-label="Fit the chain to the view">Fit</button>
    </div>
  </section>
  <aside class="read" aria-label="Measurement">
    <div class="sec"><span>Measure here</span><span class="mono" data-out="status" role="status" aria-live="polite"></span></div>
    <div class="scroll">
      <div class="field-block">
        <div class="label">Object side<span class="muted" data-out="side-now"></span></div>
        <div class="seg" role="group" aria-label="Which side of the chain touched the object">
          <button type="button" data-here="inside" aria-pressed="false">Inside</button>
          <button type="button" data-here="outside" aria-pressed="false">Outside</button>
          <button type="button" data-here="tap" aria-pressed="false">Click its side</button>
        </div>
      </div>
      <label class="field"><span>Chain</span><select data-field="chain"></select></label>
      <label class="field"><span>Colours</span><select data-field="shade">
        <option value="light">${SHADE_TEXT.light}</option><option value="dark">${SHADE_TEXT.dark}</option></select></label>
      <div class="field"><span>Pins in the chain</span>
        <span class="stepper"><button type="button" data-here="fewer" aria-label="One pin fewer">−</button><input type="number"
          data-field="pins" min="3" max="999" aria-label="Pins in the chain"><button type="button" data-here="more" aria-label="One pin more">+</button></span></div>
      <label class="field"><span>Focal length if the photo has none, mm (35 mm equiv.)</span>
        <input class="num" data-field="focal" inputmode="decimal" placeholder="none"></label>
      <p class="bad-text" data-out="focal-error" role="alert"></p>
      <div class="nums" data-out="nums"></div>
      <div class="notes" data-out="notes"></div>
    </div>
    <div class="out">
      <button type="button" class="copy" data-here="add"><span>Add to the list</span></button>
      <div class="then">then <b>Copy points</b>, and <b>Paste points</b> in a Fusion sketch</div>
    </div>
  </aside>`;
}
