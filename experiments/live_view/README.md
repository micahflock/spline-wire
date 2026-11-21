# Live view: cost and reach of the camera view's detector

Measures the phone page's live detector (`web/src/overlay/blobs.ts`), the regions it reads (`web/src/overlay/region.ts`) and the analysis (`web/src/pipeline/live.ts`) on synthetic frames of the default snap-fit chain (the S-curve pose, 13 pins), at several work resolutions and chain spans. Span is the chain's extent (its pins' greatest distance plus a link's width) as a share of the frame's short side. Findings, before and after (2026-11-19): [docs/open-items.md](../../docs/open-items.md) item 4 (live view resolution); why this is CPU work: [docs/future-work.md](../../docs/future-work.md) (WebGPU detection).

- `make_frames.py`: the 12 MP renders, and each scaled by area to 480–2016 px on the long side.
- `bench.ts`: per work size and span, the threshold (one pass against the two it replaced), the detector, pins found, the analysis; `toGray`; the still's detector.
- `regions.ts`: (1) pins found against the rim's size in work pixels (sets `REGIONS.target`); (2) the tracker frame by frame from a cold start, still and moving; (3) the scan's cost; (4) sharpness read by the old 800 px frame and by a tracked region on blurred renders.
- `still_by_span.py`: the still's accuracy (worst pin, full pipeline) by span, for the coach's `minSpan`.
- `tilt_direction.py`: whether the analysis can tell which way to move the phone to look straight down (`tilt_dir`, the coach's arrow): pins projected through a known tilted camera with noise on their centres, one reading at a time and through the arrow's limits (`coach.ts` `nextArrow`). Findings in `coach.ts` (`COACH.arrowDeg`).
- `real_frame.ts`: the detector and the analysis on a real iPhone frame (`data/wood-table-iphone.png`, the video area of the owner's camera-view screenshot, 2026-11-21: the snap-fit chain on a grained wooden table), the whole frame drawn at 640–2016 px. It found the chain only up to 1008 px until the largest candidates were tested first (`BLOBS.maxCandidates`).
- `motion.ts`: the tracker against a moving picture: the snap-fit chain sliding back and forth at 0.25–3 links per processed frame, and jumping 1–8 links in one frame, with the analysis every frame (a known focal length) or one in 15 (none); frames with all 13 pins, and frames that fell back to scanning.
- `fused_threshold.ts`: the first prototype of the one-pass threshold (now `thresholds()` in `blobs.ts`).

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
