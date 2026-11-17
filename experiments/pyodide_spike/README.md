# Pyodide spike (S1, desktop Chromium)

Does the `splinewire` pipeline run under Pyodide, as fast and as exactly as the CLI? See `docs/web-build.md`, S1.

| File | What it does |
|---|---|
| `fetch.py` | Downloads Pyodide core (npm `pyodide` package) and the needed packages (cdn.jsdelivr.net/pyodide, sha256-checked against the lock), ezdxf's wheel (PyPI), and builds the splinewire wheel, all into `out/pyodide_spike/` (git-ignored). Records sizes, including brotli transfer sizes from jsdelivr, in `sizes.json`. |
| `server.py` | Static server rooted at the repo, with year-long caching (like a versioned CDN path), COOP/COEP, and `/__stats` / `/__reset` counting the bytes it served. |
| `index.html`, `worker.js` | The page and its module worker: loads Pyodide and the packages from the same origin, installs ezdxf and the splinewire wheel with micropip (`deps=False`), then measures photos. `index.html?manual` lets a person pick a photo. |
| `measure_dump.py` | Measures a photo as `splinewire measure` does (`load_photo`, `detect_fiducials`, `process_photo`) and dumps full-precision pins and curve points. The same file runs natively and in the worker. |
| `run.mjs` | Playwright driver: cold load (fresh profile), warm load (new tab, HTTP cache warm), a dependency check tab, each measuring the given photos twice; samples Chromium's RSS. Writes `out/pyodide_spike/results-browser.json`. |
| `compare.py` | Pin-by-pin difference between a native `measure_dump.py --json` file and the browser results. |

## Rerun

```bash
uv run python experiments/pyodide_spike/fetch.py                 # ~45 MB into out/pyodide_spike/
uv run splinewire synth --env shadow                             # 4032x3024, out/synth/s-curve-shadow.jpg
uv run splinewire synth --env steep --shape pipe --out out/synth/steep
uv run python experiments/pyodide_spike/measure_dump.py out/synth/s-curve-shadow.jpg \
    --truth out/synth/s-curve-truth.json --repeats 2 --json out/pyodide_spike/native-shadow.json
node experiments/pyodide_spike/run.mjs out/synth/s-curve-shadow.jpg:out/synth/s-curve-truth.json \
    out/synth/steep/pipe-steep.jpg:out/synth/steep/pipe-truth.json
uv run python experiments/pyodide_spike/compare.py out/pyodide_spike/native-shadow.json \
    out/pyodide_spike/results-browser.json --session cold
```

To separate OpenCV-version differences from browser differences, run `measure_dump.py` with `python -I` in a
venv holding the Pyodide versions (`numpy==2.4.6 scipy==1.18.0 opencv-python-headless==4.11.0.86 pillow==12.2.0
pyyaml==6.0.3 ezdxf==1.4.5`, plus the splinewire wheel with `--no-deps`) and compare that file instead.

Manual: `python experiments/pyodide_spike/server.py`, then open
`http://127.0.0.1:8770/experiments/pyodide_spike/index.html?manual`.

## Notes

- Pyodide 314.x refuses classic workers ("Classic web workers are not supported"): the worker is a module worker.
- A plain `micropip.install` of the splinewire wheel fails: the wheel requires `opencv-python-headless`, and Pyodide
  ships OpenCV as `opencv-python`. Install with `deps=False` after `loadPackage`. Pyodide 314 does ship
  `pillow-heif` 1.3.0; `segno` is pure Python. Neither is imported at module load by the pipeline.
- `performance.measureUserAgentSpecificMemory()` is not available in headless Chromium here; memory comes from the
  wasm heap size reported by the worker and the RSS of Chromium's processes.
- Results of the 2026-11-17 run are in the Log of `docs/web-build.md` (or the report that produced it).
