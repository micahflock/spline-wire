# Live view: cost and reach of the camera view's detector

Measures the phone page's live detector (`web/src/overlay/blobs.ts`) and analysis (`web/src/pipeline/live.ts`) on synthetic frames of the default snap-fit chain (the S-curve pose, 13 pins), at several work resolutions and chain spans. Span is the chain's extent as a share of the frame's short side. Findings and the plan that follows from them: [docs/future-work.md](../../docs/future-work.md) (WebGPU detection) and [docs/open-items.md](../../docs/open-items.md) item 4 (live view resolution).

```bash
mkdir -p out/live_view
uv run python experiments/live_view/make_frames.py out/live_view          # 12 MP renders, scaled by area to 480–2016 px
cd web
npx esbuild ../experiments/live_view/bench.ts --bundle --platform=node --format=esm --outfile=../out/live_view/bench.mjs
node ../out/live_view/bench.mjs ../out/live_view                          # per-stage times, pins found, analysis time, still detection
npx esbuild ../experiments/live_view/fused_threshold.ts --bundle --platform=node --format=esm --outfile=../out/live_view/fused.mjs
node ../out/live_view/fused.mjs ../out/live_view                          # both thresholds in one pass vs today's two
```

`bench.ts` keeps a copy of `blobs.ts`'s private `components` to time it on its own. `fused_threshold.ts` is a prototype of the one-pass threshold and checks that its masks equal `localThreshold`'s.
