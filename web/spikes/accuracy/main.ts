// The accuracy test page (README.md beside it; docs/real-world-tests.md).
// Logic in src/testbench/bench.ts; this file wires it to the page: the
// settings, the camera, the photo library, storage and the report.

import { ultraWides, lensFocal } from "../../src/phone/lenses";
import { exifOfFile, type Exif } from "../../src/phone/exif";
import { loadSettings } from "../../src/phone/settings";
import { specFromDict } from "../../src/pipeline/chain";
import { decodeToGray } from "../../src/pipeline/decode";
import { GAUGES, PLAQUES } from "../../src/pipeline/targets";
import type { ChainSpec } from "../../src/pipeline/types";
import { LocalStore } from "../../src/relay/store";
import {
  chooseFocal, COLOURWAYS, measureAndScore, photoName, report, reportText, testId, verdict,
  type Colourway, type TestRecord,
} from "../../src/testbench/bench";
import { Camera } from "./camera";
import { allTests, clearAll, deleteTest, photoOf, saveTest } from "./store";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const version = `${import.meta.env.APP_VERSION} (build ${import.meta.env.BUILD_ID})`;
$("version").textContent = `Version ${version}.`;

// ---- settings: this page's own, and the phone page's lenses (read only) ----

const store = new LocalStore("splinewire-bench:");
const phoneSettings = () => loadSettings(new LocalStore("splinewire-phone:"));
const fields = ["phone", "shade", "links", "focal", "target", "tilt", "frame", "surface", "light"] as const;
type Field = (typeof fields)[number];
const input = (f: Field) => $<HTMLInputElement | HTMLSelectElement>(f);

for (const [key, label] of Object.entries(COLOURWAYS)) $("shade").append(new Option(label, key));
const groups: [string, [string, string][]][] = [
  ["Fixed-chain plaques", PLAQUES.map((p) => [p.name, `Plaque: ${p.label}`])],
  ["Contact gauges", GAUGES.map((g) => [g.name, g.label])],
  ["No known shape", [["object", "An object (measured, not scored)"]]],
];
for (const [label, options] of groups) {
  const g = document.createElement("optgroup");
  g.label = label;
  for (const [value, text] of options) g.append(new Option(text, value));
  $("target").append(g);
}
for (const f of fields) {
  const saved = store.get(f);
  if (saved !== null) input(f).value = saved;
  input(f).addEventListener("change", () => {
    store.set(f, input(f).value);
    if (f === "focal") showFocalNote();
  });
}

function links(): number {
  const n = Math.round(Number(input("links").value));
  return n >= 6 && n <= 40 ? n : 13;
}

function savedFocal(): number | null {
  const f = Number(input("focal").value);
  if (f >= 5 && f <= 500) return f;
  return phoneSettings().savedFocal35;
}

function showFocalNote(): void {
  const s = phoneSettings();
  const parts = [];
  if (s.lenses.length) parts.push(`the phone page remembers ${s.lenses.map((l) => `${l.name} (${l.focal35} mm)`).join(", ")}`);
  if (s.savedFocal35) parts.push(`its saved focal length is ${s.savedFocal35} mm`);
  $("focal-note").textContent = "The focal length is for photos without one of their own: a photo's EXIF is used first, "
    + "then a lens the phone page remembers, then this number" + (parts.length ? `; ${parts.join("; ")}.` : ".");
}
showFocalNote();

let chainBase: Promise<Record<string, unknown>> | null = null;
async function chain(): Promise<ChainSpec> {
  chainBase ??= fetch(new URL("../../chains.json", location.href)).then(async (r) => {
    if (!r.ok) throw new Error(`chains.json: HTTP ${r.status}`);
    return ((await r.json()) as Record<string, Record<string, unknown>>)["snap"]!;
  });
  const shade = input("shade").value as Colourway;
  return specFromDict({ ...(await chainBase), n_pins: links(), fiducial_shade: shade });
}

// ---- taking and measuring ----

const camera = new Camera($<HTMLVideoElement>("video"));
const paint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const now = (text: string) => { $("now").textContent = text; };

function showCamera(open: boolean): void {
  $("video").hidden = !open;
  $("shoot").hidden = !open;
  $("close").hidden = !open;
  $("open").hidden = open;
}

$("open").onclick = async () => {
  now("Starting the camera…");
  try {
    const s = phoneSettings();
    const label = await camera.start(s.camera, ultraWides(s));
    showCamera(true);
    now(`Camera: ${label || "rear camera"}. Frame the whole chain, then take the photo.`);
  } catch (e) {
    camera.stop();
    showCamera(false);
    now(`The camera did not open: ${(e as Error).message}. Choose a photo from the library instead.`);
  }
};
$("close").onclick = () => {
  camera.stop();
  showCamera(false);
  now("");
};
$("shoot").onclick = async () => {
  ($("shoot") as HTMLButtonElement).disabled = true;
  try {
    const shot = await camera.shoot();
    showCamera(false);
    await run(shot.blob, shot.kind === "photo" ? "camera" : "frame");
  } catch (e) {
    now(`The photo was not taken: ${(e as Error).message}`);
  } finally {
    ($("shoot") as HTMLButtonElement).disabled = false;
  }
};
$<HTMLInputElement>("pick").onchange = async (e) => {
  const el = e.target as HTMLInputElement;
  const file = el.files?.[0];
  el.value = "";
  if (file) await run(file, "library");
};

async function run(photo: Blob, route: TestRecord["route"]): Promise<void> {
  const phone = input("phone").value.trim() || "phone";
  const target = input("target").value;
  const n = Number(store.get("count") ?? "0") + 1;
  store.set("count", String(n));
  const id = testId(phone, n);
  now(`${id}: reading the photo…`);
  const exif: Exif = await exifOfFile(photo).catch(() => ({ make: null, model: null, lens: null, focalMm: null, focal35: null }));
  const record: TestRecord = {
    id, photo: photoName(id, target, photo.type), created: new Date().toISOString(), phone, chain: "snap",
    shade: input("shade").value as Colourway, pins: links(), target, route,
    conditions: { tilt: input("tilt").value, surface: input("surface").value, light: input("light").value, frame: input("frame").value },
    exif: { make: exif.make, model: exif.model, focal_mm: exif.focalMm, focal_35mm: exif.focal35 },
    saved_focal_35mm: savedFocal(), size: [0, 0],
  };
  try {
    const gray = await decodeToGray(photo);
    record.size = [gray.width, gray.height];
    now(`${id}: measuring ${gray.width}×${gray.height}… (a few seconds; the page waits)`);
    await paint();
    const remembered = lensFocal(phoneSettings(), exif)?.focal35 ?? null;
    record.results = measureAndScore(gray, await chain(), target, chooseFocal(exif.focal35, remembered, record.saved_focal_35mm));
    const v = verdict(record.results.focal.main, record.results.pins_found, record.pins);
    now(`${id}: ${v.ok ? "PASS" : "CHECK"} — ${v.text}`);
  } catch (e) {
    record.error = (e as Error).message;
    now(`${id}: ${record.error}`);
  }
  await saveTest(record, photo);
  await list();
}

// ---- the list, saving and sending ----

async function list(): Promise<void> {
  const tests = (await allTests()).reverse();
  const box = $("tests");
  box.replaceChildren();
  if (!tests.length) box.innerHTML = '<p class="muted">None yet.</p>';
  for (const t of tests) {
    const row = document.createElement("div");
    row.className = "test";
    const r = t.results;
    const head = document.createElement("div");
    head.innerHTML = `<b>${t.id}</b> ${t.target} · ${COLOURWAYS[t.shade]} · ${t.pins} links · ${t.route}`;
    row.append(head);
    if (!r) {
      row.append(line(t.error ?? "not measured", false));
    } else {
      for (const run of [r.focal.main, r.focal.estimated]) {
        if (!run) continue;
        const v = verdict(run, r.pins_found, t.pins);
        row.append(line(`${run.source} focal ${run.focal_35mm ?? "?"} mm: ${v.text}`, v.ok));
      }
      const info = document.createElement("div");
      info.className = "muted";
      info.textContent = `tilt ${r.tilt_deg}° (aimed ${t.conditions.tilt}), ${t.conditions.surface}, ${t.conditions.light}, `
        + `${t.size[0]}×${t.size[1]}, ${(r.ms / 1000).toFixed(1)} s` + (r.warnings.length ? ` · ${r.warnings.join(" · ")}` : "");
      row.append(info);
    }
    const actions = document.createElement("div");
    actions.className = "actions";
    actions.append(button("Save photo", async () => {
      const blob = await photoOf(t.id);
      if (blob) await saveFiles([new File([blob], t.photo, { type: blob.type || "image/jpeg" })]);
    }), button("Delete", async () => {
      if (confirm(`Delete ${t.id} and its photo?`)) {
        await deleteTest(t.id);
        await list();
      }
    }));
    row.append(actions);
    box.append(row);
  }
}

function line(text: string, ok: boolean): HTMLElement {
  const d = document.createElement("div");
  d.className = ok ? "ok" : "bad";
  d.textContent = text;
  return d;
}

function button(text: string, onclick: () => Promise<void>): HTMLButtonElement {
  const b = document.createElement("button");
  b.className = "quiet";
  b.textContent = text;
  b.onclick = () => void onclick().catch((e) => alert((e as Error).message));
  return b;
}

/** Share the files where the phone can (iOS: Save to Files keeps the bytes), else download them. */
async function saveFiles(files: File[]): Promise<void> {
  const nav = navigator as Navigator & { canShare?: (d: { files: File[] }) => boolean };
  if (nav.canShare?.({ files }) && navigator.share) {
    try {
      await navigator.share({ files });
      return;
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
    }
  }
  for (const f of files) download(f, f.name);
}

function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

async function current() {
  return report(await allTests(), version, navigator.userAgent, new Date().toISOString());
}

$("copy").onclick = async () => {
  const r = await current();
  try {
    await navigator.clipboard.writeText(reportText(r));
    $("sent").textContent = `Copied ${r.tests.length} test(s): paste them into a message.`;
  } catch {
    $("sent").textContent = "Could not copy: use Download the report instead.";
  }
};
$("download").onclick = async () => {
  const r = await current();
  const phone = (input("phone").value.trim() || "phone").replace(/\W+/g, "-");
  download(new Blob([JSON.stringify(r, null, 1)], { type: "application/json" }), `spline-wire-tests-${phone}.json`);
  $("sent").textContent = `Downloaded the report of ${r.tests.length} test(s). Keep it with the photos: `
    + "splinewire score --report scores them again.";
};
$("save-all").onclick = async () => {
  const tests = await allTests();
  const files: File[] = [];
  for (const t of tests) {
    const blob = await photoOf(t.id);
    if (blob) files.push(new File([blob], t.photo, { type: blob.type || "image/jpeg" }));
  }
  if (!files.length) {
    $("sent").textContent = "No photos to save.";
    return;
  }
  await saveFiles(files);
};
$("clear").onclick = async () => {
  if (!confirm("Delete every test and photo on this phone? Save the photos and the report first.")) return;
  await clearAll();
  await list();
};

void list();
