// The decoding check (spikes/decode-check/README.md): decode the test photo in
// this browser, compare the gray pixels with Pillow's and the pins with
// Python's, and time the TypeScript engine on a photo of the person's own.

import { exifOfFile } from "../../src/phone/exif";
import { specFromDict } from "../../src/pipeline/chain";
import type { Gray } from "../../src/pipeline/cv";
import { decodeToGray } from "../../src/pipeline/decode";
import { detectFiducials } from "../../src/pipeline/detect";
import { measureDetections } from "../../src/pipeline/measure";
import { focalPxFrom35mm } from "../../src/pipeline/session";
import photoUrl from "./photo.jpg?url";
import grayUrl from "./pillow-gray.png?url";
import python from "./python.json";

const $ = (id: string) => document.getElementById(id)!;
const report: string[] = [`page: decode-check, ${navigator.userAgent}`];
const spec = specFromDict(python.chain as unknown as Record<string, unknown>);
const chainName = spec.contact === "round" ? "the snap-fit chain" : "the O-ring chain";
report.push(`chain: ${chainName} (${spec.pitch_mm} mm pitch, ${spec.n_pins} pins)`);

function show(el: HTMLElement, line: string): void {
  el.textContent += line + "\n";
  report.push(line);
}

/** Let the page paint before a long synchronous step. */
const paint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

async function test(): Promise<void> {
  const out = $("test-out");
  out.textContent = "";
  ($("run") as HTMLButtonElement).disabled = true;
  try {
    const t0 = performance.now();
    const gray = await decodeToGray(await (await fetch(photoUrl)).blob());
    const decodeMs = performance.now() - t0;
    const ref = await decodeToGray(await (await fetch(grayUrl)).blob());     // lossless: Pillow's pixels exactly
    show(out, `decoded ${gray.width}x${gray.height} in ${decodeMs.toFixed(0)} ms (expected ${python.width}x${python.height}: `
      + `${gray.width === python.width && gray.height === python.height ? "upright, as Pillow" : "NOT as Pillow: the EXIF rotation differs"})`);
    if (gray.width !== ref.width || gray.height !== ref.height) return;
    let differ = 0, worst = 0;
    for (let i = 0; i < ref.data.length; i++) {
      const d = Math.abs(ref.data[i]! - gray.data[i]!);
      if (d) differ++;
      worst = Math.max(worst, d);
    }
    show(out, `gray pixels differing from Pillow's: ${differ} of ${ref.data.length} (${(100 * differ / ref.data.length).toFixed(3)}%), at most ${worst} levels`);
    await paint();
    const r = measure(gray, python.focal_px);
    show(out, `detected ${r.fiducials} fiducials (Python ${python.fiducials}) in ${r.detectMs.toFixed(0)} ms; measured ${r.pinsPx.length} pins in ${r.measureMs.toFixed(0)} ms`);
    if (r.pinsPx.length === python.pins_px.length) {
      const px = Math.max(...r.pinsPx.map((p, i) => Math.hypot(p[0] - python.pins_px[i]![0]!, p[1] - python.pins_px[i]![1]!)));
      const mm = Math.max(...r.pinsMm.map((p, i) => Math.hypot(p[0] - python.pins_mm[i]![0]!, p[1] - python.pins_mm[i]![1]!)));
      show(out, `pins against Python's: worst ${px.toFixed(4)} px, ${mm.toFixed(5)} mm`);
    }
    memory(out);
  } catch (err) {
    show(out, `failed: ${err instanceof Error ? `${err.name}: ${err.message}` : String(err)}`);
  } finally {
    ($("run") as HTMLButtonElement).disabled = false;
  }
}

function measure(gray: Gray, focal: number | null) {
  const t0 = performance.now();
  const fids = detectFiducials(gray, spec);
  const t1 = performance.now();
  const m = measureDetections(fids, [gray.width, gray.height], spec, focal);
  return {
    fiducials: fids.length, detectMs: t1 - t0, measureMs: performance.now() - t1,
    pinsPx: m.pins_px, pinsMm: m.rectification.pins_mm, warnings: m.warnings,
  };
}

async function own(file: File): Promise<void> {
  const out = $("own-out");
  out.textContent = "";
  try {
    const exif = await exifOfFile(file);
    show(out, `${file.type || "unknown type"}, ${(file.size / 1e6).toFixed(1)} MB; EXIF: ${exif.make ?? "?"} ${exif.model ?? ""}, `
      + `focal ${exif.focalMm ?? "?"} mm, 35 mm equivalent ${exif.focal35 ?? "none"}`);
    const t0 = performance.now();
    const gray = await decodeToGray(file);
    show(out, `decoded ${gray.width}x${gray.height} (${(gray.width * gray.height / 1e6).toFixed(1)} MP) in ${(performance.now() - t0).toFixed(0)} ms`);
    await paint();
    if (!exif.make && !exif.focal35) {
      show(out, "this photo carries no camera data (EXIF): the focal length is estimated from the chain, which is less "
        + "accurate. How was it picked? (a photo from the camera app keeps it; Take Photo in the picker, a screenshot "
        + "or a photo sent through a messaging app does not)");
    }
    show(out, `measuring with ${chainName}…`);
    const focal = exif.focal35 ? focalPxFrom35mm(exif.focal35, [gray.width, gray.height]) : null;
    const r = measure(gray, focal);
    show(out, `detected ${r.fiducials} fiducials in ${r.detectMs.toFixed(0)} ms; measured ${r.pinsPx.length} pins in ${r.measureMs.toFixed(0)} ms`
      + (r.warnings.length ? `\nwarnings: ${r.warnings.join(" / ")}` : ""));
    memory(out);
  } catch (err) {
    show(out, `failed: ${err instanceof Error ? `${err.name}: ${err.message}` : String(err)}`);
  }
}

function memory(out: HTMLElement): void {
  const m = (performance as { memory?: { usedJSHeapSize: number } }).memory;
  if (m) show(out, `JS heap in use: ${(m.usedJSHeapSize / 1e6).toFixed(0)} MB`);
}

$("run").onclick = () => void test();
($("pick") as HTMLInputElement).onchange = (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) void own(f);
};
$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText(report.join("\n"));
    $("copied").textContent = "Copied. Paste it into a message to the developer.";
  } catch {
    $("copied").textContent = "Could not copy; select the text above instead.";
  }
};
