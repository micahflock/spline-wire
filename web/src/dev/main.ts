// The throwaway test page (web-build step 1's exit criterion): drop a photo,
// see it measured by the core worker. Exposes `core` and `chains` on window
// for the Playwright test (e2e/core.spec.ts).

import { CoreClient, type PhotoSource } from "../core/client";
import type { ChainSpec, MeasurementDocument, OutputKind, Side } from "../core/types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const core = new CoreClient(() => new Worker(new URL("../core/worker.ts", import.meta.url), { type: "module" }));
const chains: Promise<Record<string, ChainSpec>> = fetch("chains.json").then((r) => r.json());
Object.assign(window, { core, chains });

core.onProgress = (p) => {
  $<HTMLProgressElement>("progress").value = p.fraction;
  $("stage").textContent = p.stage;
};
core.preload().then((r) => { $("stage").textContent = `ready in ${(r.ms / 1000).toFixed(1)} s`; },
                    (e: Error) => { $("error").textContent = e.message; });

void chains.then((all) => {
  for (const name of Object.keys(all)) $("chain").append(new Option(name, name));
});

let current: { id: string; file: File; bitmap: ImageBitmap } | null = null;

async function measure(): Promise<void> {
  if (!current) return;
  const { id, file } = current;
  const isLatest = core.latest(id);
  const chain = (await chains)[$<HTMLSelectElement>("chain").value]!;
  const source: PhotoSource = { name: file.name, bytes: () => file.arrayBuffer() };
  $("error").textContent = "";
  try {
    const t0 = performance.now();
    const doc = await core.measure(id, { chain, side: $<HTMLSelectElement>("side").value as Side }, source);
    if (isLatest()) show(doc, performance.now() - t0);
  } catch (e) {
    $("error").textContent = (e as Error).message;
  }
}

function show(doc: MeasurementDocument, ms: number): void {
  $("error").textContent = doc.error ?? "";
  const r = doc.result;
  const rows: [string, unknown][] = r ? [
    ["pins", `${r.pins} (${r.missing_pins} missing, ${r.rejected} left out)`], ["curve points", r.curve_points],
    ["tilt", `${r.tilt_deg}°`], ["focal (35 mm)", `${r.focal_35mm} mm, ${r.focal_source}`],
    ["link residual rms / max", `${r.link_residual_rms_mm} / ${r.link_residual_max_mm} mm`],
    ["object side", r.object_side],
  ] : [];
  rows.push(["photo", `${doc.info.width}×${doc.info.height} ${doc.info.format ?? ""} ${doc.info.model ?? ""}`],
            ["measured in", `${(ms / 1000).toFixed(2)} s`]);
  $("numbers").replaceChildren(...rows.map(([k, v]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td></td><td></td>";
    tr.cells[0]!.textContent = k;
    tr.cells[1]!.textContent = String(v);
    return tr;
  }));
  const items = (xs: string[]) => xs.map((x) => Object.assign(document.createElement("li"), { textContent: x }));
  $("warnings").replaceChildren(...items(r?.warnings ?? []));
  $("notes").replaceChildren(...items(r?.notes ?? []));
  draw(doc);
  void downloads(doc.ok);
}

function draw(doc: MeasurementDocument): void {
  const canvas = $<HTMLCanvasElement>("photo");
  if (!current) return;
  const { bitmap } = current;
  const scale = Math.min(1, 1600 / bitmap.width);
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const g = canvas.getContext("2d")!;
  g.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const v = doc.view;
  if (!v) return;
  g.scale(canvas.width / v.width, canvas.height / v.height);
  const lw = v.width / canvas.width;
  g.lineWidth = 2 * lw;
  const circle = (x: number, y: number, r: number, color: string) => {
    g.strokeStyle = color;
    g.beginPath();
    g.arc(x, y, r, 0, 2 * Math.PI);
    g.stroke();
  };
  for (const m of v.unused) circle(m.x, m.y, m.r, "#888");
  for (const m of v.pins) circle(m.x, m.y, m.r, m.edit !== undefined ? "#d70" : "#2a7");
  if (v.curve.length) {
    g.strokeStyle = "#e3c";
    g.beginPath();
    v.curve.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.stroke();
  }
}

async function downloads(ok: boolean): Promise<void> {
  const box = $("downloads");
  box.replaceChildren();
  if (!ok || !current) return;
  const { files, names } = await core.call("outputs", { photo: current.id });
  for (const [kind, name] of Object.entries(names)) {
    const a = document.createElement("a");
    a.textContent = name;
    a.download = name;
    a.href = URL.createObjectURL(new Blob([files[kind as OutputKind]], { type: "application/octet-stream" }));
    box.append(a);
  }
  const copy = document.createElement("button");
  copy.textContent = "Copy points";
  copy.onclick = () => void navigator.clipboard.writeText(files.tsv).then(() => (copy.textContent = "Copied"));
  box.append(copy);
}

async function take(file: File): Promise<void> {
  if (current) await core.call("close", { photo: current.id }).catch(() => {});
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    $("error").textContent = `${file.name} is not a photo this browser can show`;
    return;
  }
  current = { id: `${file.name}-${file.size}-${file.lastModified}`, file, bitmap };
  await measure();
}

$<HTMLInputElement>("file").onchange = (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (f) void take(f);
};
$("chain").onchange = $("side").onchange = () => void measure();
const drop = $("drop");
drop.ondragover = (ev) => { ev.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (ev) => {
  ev.preventDefault();
  drop.classList.remove("over");
  const f = ev.dataTransfer?.files[0];
  if (f) void take(f);
};
