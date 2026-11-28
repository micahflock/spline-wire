# Live view: cost and reach of the camera view's detector

Measures the phone page's live detector (`web/src/overlay/blobs.ts`), the regions it reads (`web/src/overlay/region.ts`) and the analysis (`web/src/pipeline/live.ts`) on synthetic frames of the default snap-fit chain (the S-curve pose, 13 pins), at several work resolutions and chain spans. Span is the chain's extent (its pins' greatest distance plus a link's width) as a share of the frame's short side. Findings, before and after (2026-11-19): **Findings** below; the phone readings and later changes are in [docs/web-build.md](../../docs/web-build.md)'s Log (2026-11-19 to 2026-11-23); why this is CPU work: [docs/future-work.md](../../docs/future-work.md) (WebGPU detection).

- `make_frames.py`: the 12 MP renders, and each scaled by area to 480–2016 px on the long side.
- `bench.ts`: per work size and span, the threshold (one pass against the two it replaced), the detector, pins found, the analysis; `toGray`; the still's detector.
- `regions.ts`: (1) pins found against the rim's size in work pixels (sets `REGIONS.target`); (2) the tracker frame by frame from a cold start, still and moving; (3) the scan's cost; (4) sharpness read by the old 800 px frame and by a tracked region on blurred renders.
- `still_by_span.py`: the still's accuracy (worst pin, full pipeline) by span, for the coach's `minSpan`.
- `tilt_direction.py`: whether the analysis can tell which way to move the phone to look straight down (`tilt_dir`, the coach's arrow): pins projected through a known tilted camera with noise on their centres, one reading at a time and through the arrow's limits (`coach.ts` `nextArrow`). Findings in `coach.ts` (`COACH.arrowDeg`).
- `real_frame.ts`: the detector and the analysis on a real iPhone frame (`data/wood-table-iphone.png`, the video area of the owner's camera-view screenshot, 2026-11-21: the snap-fit chain on a grained wooden table), the whole frame drawn at 640–2016 px. It found the chain only up to 1008 px until the largest candidates were tested first (`BLOBS.maxCandidates`).
- `motion.ts`: the tracker against a moving picture: the snap-fit chain sliding back and forth at 0.25–3 links per processed frame, and jumping 1–8 links in one frame, with the analysis every frame (a known focal length) or one in 15 (none); frames with all 13 pins, and frames that fell back to scanning.
- `fused_threshold.ts`: the first prototype of the one-pass threshold (now `thresholds()` in `blobs.ts`).

## Findings (2026-11-19, live view resolution)

The live detector read the whole frame scaled to 800 px on the long side, and that, not the still, limited how far the phone could be: the 13-pin snap-fit chain was lost below a span of ~0.40 (span: the pins' greatest distance plus a link's width, over the frame's short side), exactly the coach's old `minSpan`. Now (these scripts, synthetic frames, Node on a machine about 2× the docs' reference; Chromium headless, no GPU):

- **One-pass threshold** (`blobs.ts`, `thresholds()`): both masks from running window sums in one pass, buffers kept between frames, padded masks so the components need no bounds checks, the components' stack preallocated. The masks are bit-identical to `localThreshold`'s (`blobs.test.ts`), and the detections identical on all 105 frame/fiducial combinations of the benchmark. `detectBlobs` per frame, before → after: 800 px 10–13 → 3–4 ms, 1200 px 23–29 → 8 ms, 1600 px 40–54 → 11–15 ms (the thresholds alone at 800 px: 7.9 → 2.7 ms).
- **Regions** (`overlay/region.ts`): while the chain is tracked, only the box round it (1.5 links of margin, plus a link per pin not seen) is drawn, at the scale that makes its rims 18 work px across (all rims are found from ~14.5 px: 13/13 at 14.3–14.9, 9/13 at 13.1–13.7), never finer than 2016 px equivalent. With no chain: the whole frame at 1200 px, then each quarter (0.6 of each side) at 2016 px equivalent, in turn. A chain is followed from frame to frame (as trace.ts follows it) and dropped after 4 frames without 3 of its pins, or 2 analyses that find no chain.

  | Span | Pins found, 800 px frame (before) | After: found, from a cold start | Tracked region, detector per frame |
  |---|---|---|---|
  | 0.55 | 13 | 13 from the first frame | 0.17 MP, 2 ms |
  | 0.40 | 12 | 13 from the first frame | 0.09 MP, 1.4 ms |
  | 0.30 | 0 | 13 from the first frame | 0.07 MP, 1.2 ms |
  | 0.22 | 0 | 13 from the second frame (a quarter) | 0.07 MP, 1.3 ms |
  | 0.16 | 0 | 12 (rims 14 px at 2016 px) | 0.06 MP, 1.0 ms |

  Scanning costs 7–9 ms a frame (1.1 MP), against 10–13 ms for the old 800 px frame. Still or moving at a fifth of a link a frame (twice the coach's motion limit), the tracker keeps all 13 pins. Sharpness is read at the old full frame's scale (`Window.refScale`), so the coach's limits keep their meaning: on blurred renders the old frame and a tracked region read 0.60 / 0.62 (σ 4 px) and 0.41 / 0.41 (σ 7); only mild blur reads sharper (0.77 / 0.89 at σ 2).
- **The grab in a worker** (`camera/grab.ts`, `grab-worker.ts`): `createImageBitmap(video, …)` costs the page as long as drawing it (37 of 37 ms for a 12 MP frame in headless Chromium), but taking a `VideoFrame` of the video costs ~0 ms, so the page hands the frame to a worker that draws the region, reads it back and finds the blobs. A first check compares the worker's copy of a region with the page's own (orientation and pixels); without VideoFrame (Firefox on Android may lack it), after a failed check, an error or no answer in 2 s, or with `?grab=main`, the page grabs as before. In Chromium (`e2e/camera.spec.ts`, the O-ring chain): the page's own time per frame ~39 ms (grab 34 + detector 4.8) → 0.1–0.2 ms, and a 16 ms timer late by median 26 / p95 45 / max 91 ms → 0.2 / 2.7 / 7 ms; the worker's grab is still ~35–40 ms per 12 MP frame there (no GPU), the detector 1.4–1.8 ms.
- **Coach** (`coach.ts`): `minSpan` 0.40 / 0.33 → 0.26 / 0.22 (enter / keep, of the pins' bounding box, at least 0.23 in the span above). The still measures its worst pin to 0.003–0.004 mm from span 0.40 to 0.26 and 0.007 mm at 0.22, and loses a pin at 0.18 (`still_by_span.py`, four poses, clean renders). Nothing else in the coach depends on the work size.
- **Tests**: `blobs.test.ts` (masks bit for bit; windows; spans 0.30 and 0.22 on three tables, ordered by the real analysis), `region.test.ts`, `session.test.ts` (the trace lands on the pins in video pixels), `grab.test.ts`; `e2e/camera-snap.spec.ts` at spans 0.55 (page grab), 0.30 (tracked, the still taken and measured) and 0.22 (tracked, all 13 pins, "Move closer", no still).

Before and after a change: run the same script in a worktree of the old revision (`git worktree add --detach ../before <rev>`, with `web/node_modules` linked in).

```bash
mkdir -p out/live_view
uv run python experiments/live_view/make_frames.py out/live_view          # 12 MP renders, scaled by area to 480–2016 px
cd web
npx esbuild ../experiments/live_view/bench.ts --bundle --platform=node --format=esm --outfile=../out/live_view/bench.mjs
node ../out/live_view/bench.mjs ../out/live_view                          # per-stage times, pins found, analysis time, still detection
npx esbuild ../experiments/live_view/fused_threshold.ts --bundle --platform=node --format=esm --outfile=../out/live_view/fused.mjs
node ../out/live_view/fused.mjs ../out/live_view                          # both thresholds in one pass vs today's two
npx esbuild ../experiments/live_view/regions.ts --bundle --platform=node --format=esm --outfile=../out/live_view/regions.mjs
node ../out/live_view/regions.mjs ../out/live_view                        # reach by scale, the tracker, the scan, sharpness
npx esbuild ../experiments/live_view/motion.ts --bundle --platform=node --format=esm --outfile=../out/live_view/motion.mjs
node ../out/live_view/motion.mjs                                          # the tracker at speed (~4 min)
npx esbuild ../experiments/live_view/real_frame.ts --bundle --platform=node --format=esm --outfile=../out/live_view/real_frame.mjs
node ../out/live_view/real_frame.mjs ../experiments/live_view/data/wood-table-iphone.png   # a real frame on a grained table, by work size
cd .. && uv run python experiments/live_view/still_by_span.py             # the still's accuracy by span (~1 min)
uv run python experiments/live_view/tilt_direction.py arrow               # the arrow through the coach's limits (~10 min; no argument: one reading too, ~25 min)
```

`fused_threshold.ts` checks that its masks equal `localThreshold`'s; `blobs.test.ts` checks `thresholds()`'s.
