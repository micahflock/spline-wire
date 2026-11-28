# Colour channel for detection: orange rims on off-white links

Work queue item 13 in [docs/web-build.md](../../docs/web-build.md): detection reads one channel of the photo, chosen as the chain spec's `gray_channel` (`luma`, the default, or `r`, `g`, `b`), instead of always the luma. Is blue the right channel for orange rims on off-white links, the case in [docs/future-work.md](../../docs/future-work.md) (*Fiducial colours picked from a photo*)? Measured 2026-11-28.

- `render_still.py`: renders the snap-fit chain (`data/chain-snap.yaml`, with its relief: `synthetic.render_snap_photo`, which now draws in colour) with orange rims (240, 122, 35; gray 147) on off-white links (235, 230, 214; gray 230) on six tables, as a phone saves them (per-channel noise, sigma 4, 3, 6 for R, G, B, as white balance under indoor light leaves it; 4:2:0 chroma through JPEG). Measures each still (3024×4032, span 0.40, 12° tilt, the true focal length) through every channel with Colours light and dark (`PhotoSession`, scored against the true pins), and writes the live view's frames (a 1440×1920 stream, the phone's default, at span 0.30). With `dim`, every channel's noise is three times as large.
- `live.ts`: the live view on those frames (`overlay/blobs.ts`, `region.ts`, `pipeline/live.ts`, each channel by `toGray`): from a cold start (16 frames, the analysis every third, as `experiments/live_view/regions.ts`), and seeded with the true pins (already tracked).

```bash
uv run python experiments/colour_channel/render_still.py out/colour_channel           # ~3 min
uv run python experiments/colour_channel/render_still.py out/colour_channel_dim dim
cd web && npx esbuild ../experiments/colour_channel/live.ts --bundle --platform=node --format=esm --outfile=../out/colour_channel/live.mjs
node ../out/colour_channel/live.mjs ../out/colour_channel/live                         # ~20 s
node ../out/colour_channel/live.mjs ../out/colour_channel_dim/live
```

Span: the pins' extent plus a link's width over the frame's short side.

## Findings

**With Colours light, no channel finds the chain on any table** (0 of 13 everywhere): orange is darker than the off-white links in every channel but red, where the two are the same (236 against 235). So the channel goes with Colours: dark.

**The still, Colours dark** (pins in chain order of 13; worst pin against the truth, mm):

| table (gray) | Gray | Red | Green | Blue |
|---|---|---|---|---|
| dark (60) | 0 | 0 | 12 | **13** (0.005) |
| mid-gray (120) | 13 (0.003) | 0 | 13 (0.004) | **13** (0.006) |
| light gray (200) | 13 (0.003) | 0 | 13 (0.003) | **13** (0.006) |
| white paper (235) | 13 (0.004) | 0 | 13 (0.005) | **13** (0.005) |
| wood (150, 100, 60) | 12 | 0 | 12 | **13** (0.004) |
| green mat (40, 90, 60) | 0 | 0 | 13 (0.003) | **13** (0.006) |

Median levels in the photo, rim / head inside the rim / table, the dark table: gray 148 / 230 / 60, red 236 / 235 / 60, green 124 / 230 / 60, blue 41 / 214 / 60. In blue the rim is darker than both the link and every table (wood and the green mat are 60 in blue), which is what Colours dark needs; in gray and green it is lighter than a dark table.

**The live view, Colours dark** (cold start: the frame where all 13 rims were first found, and the pins the last analysis ordered; tracked: rims found, pins ordered):

| table | Gray | Green | Blue |
|---|---|---|---|
| dark | never, 4 · 0, 0 | never, 0 · 7, 4 | **frame 1, 13 · 13, 13** |
| mid-gray | never, 8 · 11, 11 | frame 2, 12 · 13, 13 | **frame 0, 13 · 13, 13** |
| light gray | frame 0, 13 · 13, 13 | frame 0, 13 · 13, 13 | **frame 0, 13 · 13, 13** |
| white paper | frame 0, 13 · 13, 13 | frame 0, 13 · 13, 13 | **frame 0, 13 · 13, 12** |
| wood | never, 2 · 9, 3 | never, 11 · 11, 11 | **frame 1, 13 · 13, 13** |
| green mat | never, 4 · 0, 0 | never, 3 · 10, 4 | **frame 1, 13 · 13, 13** |

Red finds nothing on any table. Gray's figures match the earlier gray-only model (future-work.md: 0 of 13 on a dark table, 9 of 13 tracked on mid-gray).

**Blue is the right channel** for orange on off-white: the only one that finds and measures the whole chain on all six tables, still and live, at a cost of 0.001–0.003 mm in the still's worst pin against gray (blue is the noisiest channel).

**In dim light** (noise ×3): the still through blue still measures 13 of 13 on all six tables (worst pin 0.005–0.010 mm; gray none on the dark table and the green mat, green 11 on the dark table). The live view through blue follows the chain once tracked (13, 13 on every table), but its scan no longer finds it from a cold start on the four darker tables (dark, mid-gray, wood, green mat; best 0–2 rims in 16 frames); on light gray and white paper it does (frame 1). No channel's scan finds it on the dark table or the green mat. So in a dim room on a dark surface the live view may not lock on; a photo (Upload, or the system camera) still measures. Not a reason to prefer another channel; whether a phone's blue is as noisy as modelled is for a real phone (open-items.md item 4).
