// Web worker: loads Pyodide, its packages and the splinewire wheel from this
// origin, then measures photos with measure_dump.run (the CLI's code path).
//
// Messages in:  {id, cmd: "load", opts} | {id, cmd: "measure", photo: ArrayBuffer|url, name, truthUrl?, repeats?}
//               | {id, cmd: "heap"}
// Messages out: {id, ok: true, result} | {id, ok: false, error, type}

const BASE = new URL("../../", self.location).href;           // repo root on the static server
const PYO = BASE + "out/pyodide_spike/pyodide/";
const WHEELS = BASE + "out/pyodide_spike/wheels/";
const PACKAGES = ["numpy", "scipy", "opencv-python", "pillow", "pyyaml", "micropip"];

importScripts(PYO + "pyodide.js");

let py = null;

function heapBytes() {
  const m = py && (py._module || py._api?._module);
  const buf = m?.HEAPU8?.buffer || m?.HEAP8?.buffer || m?.wasmMemory?.buffer;
  return buf ? buf.byteLength : null;
}

async function text(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.text();
}

async function load(opts = {}) {
  const t = {};
  const t0 = performance.now();
  py = await loadPyodide({ indexURL: PYO });
  t.pyodide_core_ms = performance.now() - t0;

  let a = performance.now();
  await py.loadPackage(PACKAGES, { messageCallback: () => {} });
  t.packages_ms = performance.now() - a;

  const manifest = JSON.parse(await text(BASE + "out/pyodide_spike/manifest.json"));
  const micropip = py.pyimport("micropip");
  a = performance.now();
  const notes = [];
  if (opts.ezdxfFromPyPI) {
    // Let micropip resolve ezdxf on PyPI (needs network to pypi.org and files.pythonhosted.org).
    await micropip.install("ezdxf");
    notes.push("ezdxf installed by micropip from PyPI");
  } else {
    // ezdxf's deps (pyparsing, typing-extensions, fonttools, numpy) come from the Pyodide lock.
    await micropip.install(WHEELS + manifest.ezdxf_wheel);
    notes.push("ezdxf installed by micropip from this origin");
  }
  t.ezdxf_ms = performance.now() - a;

  a = performance.now();
  const swUrl = WHEELS + manifest.splinewire_wheel;
  if (opts.tryDeps) {
    // What a plain install does: the wheel lists pillow-heif and segno.
    try {
      await micropip.install(swUrl);
      notes.push("splinewire installed with its dependencies");
    } catch (e) {
      notes.push("splinewire with dependencies failed: " + String(e.message || e).split("\n").slice(-3).join(" | "));
    }
  }
  await micropip.install(swUrl, { deps: false });
  t.splinewire_wheel_ms = performance.now() - a;

  py.FS.mkdirTree("/spike/data");
  py.FS.mkdirTree("/spike/out");
  py.FS.writeFile("/spike/measure_dump.py", await text(BASE + "experiments/pyodide_spike/measure_dump.py"));
  py.FS.writeFile("/spike/data/chain.yaml", await text(BASE + "data/chain.yaml"));

  a = performance.now();
  await py.runPythonAsync(`
import sys, time
sys.path.insert(0, "/spike")
import measure_dump
import splinewire.process, splinewire.output, ezdxf
`);
  t.import_ms = performance.now() - a;

  const probe = py.runPython(`
import sys, json
json.dumps({"pillow_heif_loaded": "pillow_heif" in sys.modules,
            "segno_loaded": "segno" in sys.modules,
            "versions": measure_dump.versions()})`);
  t.total_ms = performance.now() - t0;
  return { timings: t, notes, heap_bytes: heapBytes(), ...JSON.parse(probe) };
}

async function measure(msg) {
  let bytes = msg.photo;
  if (typeof bytes === "string") bytes = await (await fetch(bytes)).arrayBuffer();
  const path = "/spike/" + msg.name;
  py.FS.writeFile(path, new Uint8Array(bytes));
  let truth = null;
  if (msg.truthUrl) {
    truth = "/spike/truth.json";
    py.FS.writeFile(truth, await text(msg.truthUrl));
  }
  const a = performance.now();
  const run = py.globals.get("measure_dump").run;
  const res = run(path, "/spike/data/chain.yaml", "/spike/out", truth, msg.repeats || 1);
  const json = py.pyimport("json").dumps(res);
  run.destroy();
  res.destroy();
  const wall = performance.now() - a;
  return { wall_ms: wall, heap_bytes: heapBytes(), ...JSON.parse(json) };
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    let result;
    if (msg.cmd === "load") result = await load(msg.opts);
    else if (msg.cmd === "measure") result = await measure(msg);
    else if (msg.cmd === "heap") result = { heap_bytes: heapBytes() };
    else throw new Error("unknown command " + msg.cmd);
    self.postMessage({ id: msg.id, ok: true, result });
  } catch (e) {
    // A Python ValueError is a message for the person; anything else is a bug.
    const s = String(e && e.message || e);
    const m = s.match(/(\w+Error): ([^\n]*)\n*$/);
    self.postMessage({ id: msg.id, ok: false, type: m ? m[1] : (e && e.name), error: m ? m[2] : s, trace: s });
  }
};
