# Detection robustness and fiducial design

[← Back to the README](../README.md)

How the detector holds up in realistic phone photos of a 3D-printed chain, what was changed to make it hold up, and which fiducial works best for a 0.4 mm nozzle. Everything here is simulated; real photos (next-steps item 1) still have to confirm it.

## Summary

- **The whole chain is now recovered in 155 of 167 simulated photos, up from 77**, over 29 named environments plus 80 random ones. When the chain is recovered, the worst pin is off by 0.023 mm (median), 0.056 mm (95th percentile), 0.13 mm at most: far inside the 1 mm target.
- **The one failure left is glare on shiny filament.** A lamp reflected in the black layer washes it out until it is as light as the white rings. With **matte** black filament every simulated photo worked (32/32 random setups, including 5 with the lamp reflected on the chain). Standard PLA: 32/34. Glossy or silk: 10/14, and 0/3 with a reflection.
- **The chain now uses a 5 mm dot** (easiest to make); the plain ring stays supported (`fiducial: ring` in `data/chain.yaml`) and is the most robust design. Of seven fiducial designs printed through the same 0.4 mm-nozzle model, the plain ring (5.0/2.0 mm) is the most robust. A 20% larger ring (6.0/2.4 mm) is equally robust and ~20% more accurate, and works from further away, but leaves only 1 mm of black around it on the 8 mm link. A solid 5 mm **dot** comes close in the fiducial study (86/92 vs 87/92, same median accuracy) once its detector checks for the dark link around each dot; on the full benchmark it recovers the chain in 143/167 photos against the ring's 155/167 (142 before two ordering fixes, see §5.1), losing mostly where lamp sheen on standard PLA lifts the black link toward the dot. Designs with 0.8 mm features (bullseye, checker-corner centre, ArUco tags) print, but fall apart with blur, distance or a bad print; ArUco tags don't fit an 8 mm round-ended link at all.
- **Print defects, not the camera, set the remaining error.** Seam blobs and edge wobble move the printed ring off its pin by a few hundredths of a millimetre. Lens distortion, perspective bias of ellipse centres and the relief of the black layer each matter less.

![Simulated photos](img/environments.jpg)

## 1. The simulator — `splinewire/scene.py`

`render_scene(pins, spec, env, design)` renders a phone photo of a printed plaque or chain, returning the image and where each pin centre really is in it. `PRESETS` holds 29 named environments (one hard factor at a time, then combinations); `random_environment` draws every factor at once. `uv run splinewire synth --env glare` writes one as a JPEG with EXIF.

| factor | model | range (random) |
|---|---|---|
| **Print** (0.4 mm nozzle) | Convex black corners rounded to ~half the line width (0.2 mm), black features narrower than a line vanish, white gaps under ~0.2 mm close, all edges move by the over-extrusion, smooth wobble (~1 mm correlation), a seam blob on every circular edge, extrusion lines on the top surface that modulate brightness and gloss. The black layer stands proud of the white; its walls are ray-traced. | corners 0.15–0.25 mm, over-extrusion −0.08…+0.15 mm, wobble ≤0.04 mm, seams ≤0.15 mm, relief 0.2/0.4/0.6 mm |
| **Mount** | test plaque (white plate, rim), bare chain on the table, laser print on paper | |
| **Filament** | black and white PLA albedo; specular reflectance and lobe width by type: matte, standard ("basic"), glossy/silk | 35% / 45% / 20% |
| **Table** | paper, graph paper, printed text (ring-like letters), wood, black, grey, green cutting mat, woven fabric, terrazzo; glossy tables reflect the lamp | |
| **Clutter** | washers (6–10 mm; one always in line with a chain end, 1–2 pitches out), nuts, coins | 0–8 objects |
| **Light** | ambient + a lamp (1/r², Blinn-Phong highlight), optionally placed so its reflection lands on the chain; gradient; soft shadow edge or band across the chain (phone, hand, pencil); vignetting | ambient 20–90%, shadow penumbra 2–20 mm |
| **Camera** | 12 MP (or 3 MP, as after a messaging app), 24–28 mm equivalent; tilt, roll, chain anywhere in the frame; residual radial distortion; depth-of-field defocus from the real aperture; hand shake; lens blur | 3.5–17 px/mm, tilt 0–40°, k1 ±0.015 |
| **Sensor, ISP** | centre-weighted auto-exposure with a highlight shoulder, shot + read noise, edge-preserving denoise, sharpening, sRGB, JPEG | gain 1–16×, JPEG 70–95 |

The truth is checked in `tests/test_scene.py`: on a perfect print the detected rings land within 0.15 px of the true pins; with lens distortion, within 0.2 px of where the distortion puts them. The black layer's walls hide the same share of every window edge, so the rings look as if they lay at half the relief height: still one plane, so deskewing is unaffected (as `experiments/relief_bias.py` found). Pixel truth is taken on that plane.

Not modelled: colour (the detector works on gray), chromatic aberration, rolling shutter, multi-frame HDR artefacts, a warped print, fingers or objects over the chain, and the chain hardware's own joint play.

## 2. Benchmark — `experiments/cv_benchmark.py`

Five chain shapes (S-curve, pipe, cove, tight wrap, wave), three photos per preset with different view directions, and 80 random environments. "Chain" is the whole chain recovered in order, with no stray ring taken for a pin.

| environment | before: chain | after: chain | after: worst pin, mm |
|---|---|---|---|
| ideal (the old synthetic photo) | 3/3 | 3/3 | 0.002 |
| daylight (typical: plaque, paper, lamp) | 3/3 | 3/3 | 0.021 |
| far / very far (5, 3.2 px/mm) | 3/3, 3/3 | 3/3, 3/3 | 0.022, 0.061 |
| close (18 px/mm) | 1/3 | 3/3 | 0.023 |
| steep (45°), misfocus, shake, dim | 3, 2, 3, 3 /3 | 3/3 each | ≤0.028 |
| shadow edge / shadow band | 1/3, 0/3 | 3/3, 3/3 | 0.078, 0.040 |
| black, grey table | 1/3, 0/3 | 3/3, 3/3 | 0.022, 0.050 |
| wood, cutting mat, fabric, terrazzo | 1, 1, 1, 2 /3 | 3/3 each | ≤0.047 |
| printed text (~140 ring-like letters) | 0/3 | 3/3 | 0.030 |
| clutter (washers, nuts, coins) | 1/3 | 3/3 | 0.022 |
| bad print, bare chain | 0/3, 1/3 | 3/3, 3/3 | 0.052, 0.027 |
| lens (k1 = 0.03, chain in a corner), JPEG 45 at 4 px/mm | 3/3, 3/3 | 3/3, 3/3 | 0.062, 0.097 |
| worst (several at once) | 2/3 | 3/3 | 0.059 |
| lamp reflected in matte PLA | 3/3 | 3/3 | 0.021 |
| lamp reflected in standard / glossy PLA | 0/3, 0/3 | 0/3, 0/3 | – |
| **random environments** | **27/80** | **74/80** | 0.130 |
| **all** | **77/167** | **155/167** | median 0.023, p95 0.056 |

Random environments by filament, after: matte 32/32, standard 32/34, glossy 10/14 (0/3 with the lamp reflected on the chain). All six random failures are a lamp reflected in standard or glossy filament (placed there, or close enough by chance).

Runtime: 0.4–1 s per 12 MP photo, the printed-text page (~140 ring-like letters to refine) included.

## 3. What broke, and the fixes — `detect.py`, `order.py`, `pipeline.py`

Each fix came from a failure the benchmark found.

1. **Extrusion lines bridge rings to their surroundings.** Under a lamp the top surface's lines show as bright stripes 0.42 mm apart, which cross the 1.5 mm black margin and join a ring to the white rim: no longer a blob with one hole. Candidates are now found on an image pyramid (full, ½, ¼, ⅛ size), so each ring is also seen where it is a few dozen pixels across and the stripes have averaged away.
2. **Threshold windows a quarter of the image wide** mixed the table, rim, shadows and glare into each ring's threshold. Windows are now 25 px at every pyramid level, one or two rings wide. Candidate recall over the benchmark photos (true pins with a candidate): 90.3% with 81 px windows, 96.3% at 31 px, 96.6% at 25 px.
3. **Sheen makes a black link lighter than the table beside it** (bare chain on wood): the link then sits above the local mean and the ring merges into it. A second mask, mean + 0.8 local standard deviations where there is contrast, still isolates the ring, the brightest thing around. It also covers what the old midrange mask did (black chain on a black table), which was dropped.
4. **Seam blobs fail a strict ellipse test at close range** (8/13 rings found at 18 px/mm): a blob 0.1 mm high is 2 px there. The contour test now allows a few stray points (90th percentile), and the precise fit rejects outliers.
5. **Relief at steep angles** makes the raised centre disc look rounder than the window around it; the inner/outer shape check was loosened accordingly.
6. **Sub-pixel accuracy from gray levels, not a binary image.** Each candidate is refined on the full-resolution photo along 32–180 rays: each edge is where the profile crosses halfway between *that ray's* dark and bright levels, so a shadow edge or glare gradient across a ring does not shift it. Ellipses are fitted to the edge points with outlier rejection.
7. **Weighting the two ellipse centres by area as well as fit quality**: a print defect pulls a small circle's fitted centre further than a large one's (∝ 1/radius). Median worst-pin error −20%, 95th percentile −21%.
8. **Printed letters broke ordering.** With ~100 ring-like "O", "0", "@" detections, the median nearest-neighbour distance (the old pitch estimate) was the letter spacing. Each ring's own ellipse now predicts the step to the next pin (pitch = 2 ring diameters, foreshortened at most by the ellipse's axis ratio), neighbours must match in size, and light-on-dark rings are never chained with dark-on-light ones (printed letters).
9. **A washer one pitch past the chain end** joins the chain by geometry. After deskewing, a pin whose ring is >12% off its neighbours' size in mm, or an end pin whose link is far from one pitch, is dropped and the chain re-solved. A washer of exactly a ring's size (an M2 washer is 5.0/2.2 mm) passes both; then the chain has one pin too many, and the end that looks least like its neighbours (what surrounds it: a link, or the table; and size) is dropped.
10. **Speed.** Ring-scale thresholds turn woven fabric into ~300,000 specks; tracing them took 20 s per photo. Connected components are filtered by size and fill first, and only plausible ones traced: 0.5 s.

### 3.1 Busy backgrounds seen close up

At 15–25 px/mm (a 12 MP still or a 4K frame filled by the chain) woven fabric, terrazzo and print resolve into tens of thousands of dot-like spots, and most pass the dot test: they are light, round and have a darker margin. The video frame study (web-build S5) found the whole pipeline ran out of memory on them. Measured on its renders, on a shared 4-core machine with a 4 GB cap per process:

| photo | before: detections, detect time | before: result | now: detections, detect time | now: result |
|---|---|---|---|---|
| 12 MP random-063 | 27,328, 1,286 s | ordering needs an 11.1 GiB array | 1,481, 8.7 s | 3 pins not detected (as before): `ValueError` "…the background has 1491 dot-like spots…: lay the chain on a plain surface" |
| 12 MP random-031 | 11,188, 324 s | out of memory (1.9 GiB array) | 544, 5.9 s | chain, 0.055 mm |
| 12 MP fabric-0 | 11,230, 141 s | chain, 0.016 mm | 1,426, 4.2 s | chain, 0.016 mm |
| 12 MP fabric-2 | – | process killed | 933, 7.3 s | chain, 0.101 mm |
| 4K fabric-0 / fabric-2 | 12,661 / 15,014, 155 / 301 s | out of memory | 1,134 / 948, 3.7 / 5.3 s | chain, 0.016 / 0.020 mm |
| 4K random-031 / random-063 | 11,760 / 11,672, 268 / 426 s | out of memory | 394 / 1,373, 4.3 / 5.7 s | chain, 0.050 / 0.030 mm |
| 3 MP fabric at 18 px/mm (the regression test) | 5,725, 40 s | chain, but `measure` took 883 s and 1.7 GB | 1,194, 2.6 s | chain, 0.025 mm; `measure` 1.8 s, process peak 409 MiB |

Three changes, the first two with no effect on any result:

1. **Detection's deduplication** compared every candidate with every one kept, in Python: 221 of 244 s on the 4K fabric-0 frame, 548 of 574 s on the 12 MP random-031 still (profiled). A grid now lists, for each cell, the kept detections whose reach covers it, so each candidate is compared only with those. Same detections, bit for bit, on all 334 photos checked (the benchmark and the S5 1080p frames).
2. **Ordering's all-pairs distance tables** (n × n, several of them) are replaced by a k-d tree (`scipy.spatial.cKDTree`, in Pyodide's SciPy too): each step looks only at detections within the 2.4 pitches a step can reach. Distances are computed exactly as the tables computed them, so every comparison, sum and tie comes out the same: the same order, links and gaps on all 334 photos × both polarities × four argument sets, and on 16 random cluttered chains in `tests/test_order.py` against the old code kept in `tests/dense_order.py`. The walk's size, length and turn tests are also done for all neighbours at once: 130 µs a step instead of 460 (profiled, 2,000 fabric spots).
3. **Bounds for what is left.** Refinement costs about 1 ms a candidate, and on a woven texture almost every walk is long (ordering 1,336 fabric spots, all starts, took 5.6 s at an 80° turn limit and 23 s at 125°). So:
   - detection refines only the **1,500 largest candidates** (`detect._MAX_CANDIDATES`). Over the benchmark and the S5 stills and sharp frames (668 photos), no pin ranked below 115th by size among its photo's candidates; 19 photos have more than 1,500;
   - ordering starts walks from **at most 600 detections** (`order._MAX_STARTS`; the benchmark's most is 510, so its orders are unchanged), the most end-like: fewest look-alike neighbours a pitch away, then highest contrast. Beyond 600, walks also stop at twice the chain's pin count. With limits of 600 the order was identical on all 815 photos tried (benchmark, S5 stills and sharp frames, both turn limits); with 300 the chain was found wherever it was before; with 150 it was lost on printed text. Ordering 11,000–13,000 detections now takes 1.3–1.5 s and 5 MiB at the peak (traced), the capped 900–1,500 of a 12 MP fabric photo 0.6–1.5 s and 1 MiB;
   - with more than 600 detections, `measure` adds "the background has N dot-like spots (fabric, terrazzo, print?) that can pass for pins: lay the chain on a plain surface" to its error when the chain is not found, and as a warning when the chain found looks wrong (pin count, gaps or link lengths off).

**Benchmark** (`cv_benchmark.py --baseline HEAD`, chain.yaml as is, 125°): 166 of 167 photos identical in every recorded field; random-049 has 45 look-alike detections instead of 220 (its candidates were capped), with the same chain and pin errors. Totals unchanged: 143/167, worst pin median 0.025, p95 0.056, max 0.151 mm. Detection time median 0.56 s (0.55 before), slowest photo 3.2 s (12.3 s before, random-049).

**S5 subset** (fabric and the 80 random environments, the still-doc, 12 MP still and sharp 4K, 1080p and 720p conditions, 415 photos, 80°): 403 identical; 6 that ran out of memory now recover the chain (12 MP 68 → 70 of 83, 4K 69 → 73); random-063's 12 MP still fails as before, in 8.7 s with the message above; 5 have fewer look-alikes and the same result. Detection over the 83 stills took 105 s instead of 1,872 s, over the 83 4K frames 70 s instead of 1,229 s.

Not measured in the browser. At the 2–2.5× slowdown S1 found, the worst 12 MP fabric photo should take 10–25 s there, mostly in finding candidates (connected components, contours and ellipse fits for up to 63,000 specks, unchanged); ordering's memory is now a few MB against the hundreds of MB the image and its masks take.

## 4. Where the remaining error comes from

- **Print defects dominate.** On the worst benchmark photo (0.16 mm), rendering without print defects gives 0.06 mm; without lens distortion, relief, defocus or noise, it hardly changes. Seams and wobble genuinely move a printed ring off its pin, and a small circle most: hence the area weighting (fix 7) and the case for larger rings (§5).
- **Lens distortion is minor** (`experiments/lens_distortion.py`, 12 MP, 300 mm away, 0.1 px noise):

  | k1 | chain centred | chain in a corner |
  |---|---|---|
  | 0 | 0.03 mm | 0.02–0.04 mm |
  | 0.01 | 0.03 mm | 0.03–0.04 mm |
  | 0.02 | 0.03–0.04 mm | 0.05–0.07 mm |
  | 0.04 | 0.03–0.05 mm | 0.09–0.14 mm |

  Adding k1 to the deskew fixes straight and S-shaped chains, but on arcs, where distortion and curvature trade off, it makes things worse: 0.15–0.26 mm even with no distortion at all. `rectify_chain(estimate_distortion=True)` exists for the experiment; the pipeline leaves it off.
- **Perspective bias of ellipse centres** (the image of a circle's centre is not the centre of its image ellipse) is ~f·(r/Z)²·sin(2·tilt)/2: about 0.07 px for a 12 MP photo from 300 mm at 25°, 4× that from half the distance. It is nearly the same shift for every pin, so the measured shape moves by ~0.002 mm (the ideal photos).
- **Relief** makes the rings look as if they lay 0.2 mm up: one plane, no effect.
- **Glare** is the limit on *detection*: at the mirror angle, satin black PLA's highlight is about twice as bright as diffuse white PLA, so the contrast between ring and link is gone, not merely low.

## 5. Fiducial options for a 0.4 mm nozzle — `experiments/fiducial_study.py`

All designs are windows cut into the black top layer over white, centred on the pin, inside the 8 mm link (`splinewire/fiducials.py`). A 0.4 mm nozzle lays ~0.42 mm lines: two lines (0.8 mm) is the narrowest band that prints dependably, convex black corners round off by ~0.2 mm, and white gaps under ~0.2 mm fill in.

![Designs as drawn, as a typical print and as a bad print](img/fiducials-printed.png)

| design | narrowest feature | black margin to link edge | notes |
|---|---|---|---|
| **ring** 5.0/2.0 mm | 1.5 mm band (3.5 lines), 2.0 mm centre | 1.5 mm | supported: `fiducial: ring` |
| ring-6 6.0/2.4 mm | 1.8 mm band, 2.4 mm centre | 1.0 mm | |
| **dot**, 5.0 mm (the chain's) | 5.0 mm disc | 1.5 mm | no hole: told from other bright blobs by the dark link around it |
| bullseye 6.0 mm | 0.8 mm bands, 1.2 mm centre dot | 1.0 mm | four edges to fit |
| ring-x 6.2 mm | 0.8 mm band and gap, checker corner in a 3.0 mm disc | 0.9 mm | centre from the checker's saddle point (`cv2.cornerSubPix`), exact under perspective |
| ArUco 4×4, 5.6 mm | 0.93 mm cells | 0.04 mm at a round chain end | printed inverted (white border); ids give the order |
| ArUco 4×4, 4.8 mm | 0.8 mm cells | 0.4 mm | |

Each design went through the same renderer and a detector suited to it (the production ring detector; for the dot, blobs checked for a uniformly bright inside and a steadily darker margin all round (§5.1); all four edges for the bullseye; ring then saddle point for ring-x; OpenCV ArUco with id ordering), then the same ordering and deskewing.

**Environments** (26 presets × 2 photos + 40 random, 92 photos): chain recovered / worst-pin error of recovered photos, median · 95th percentile.

| ring | ring-6 | dot | bullseye | ring-x | ArUco 5.6 | ArUco 4.8 |
|---|---|---|---|---|---|---|
| **87/92**<br>0.026 · 0.069 mm | **87/92**<br>0.020 · 0.062 mm | 86/92<br>0.025 · 0.076 mm | 77/92<br>0.023 · 0.053 mm | 76/92<br>0.024 · 0.126 mm | 2/92<br>0.087 mm | 16/92<br>0.049 · 0.172 mm |

**Resolution** (typical conditions, 3 photos each): the lowest px/mm at which every photo worked.

| ring | ring-6 | dot | bullseye | ring-x | ArUco 5.6 | ArUco 4.8 |
|---|---|---|---|---|---|---|
| 2.5 | **2.0** | 3.5 (one dot missed at 3.0) | 4.0 | 4.0 | never (≤6.5) | none of 3 at ≤5, 2/3 at 6.5 |

**Hand shake** at 8 px/mm: the rings and the dot work up to a 16 px (2 mm) streak; bullseye and ring-x fail from 12 px.

What it shows:

- **Plain rings are the most robust**, and their accuracy is far better than needed. Their 1.5–1.8 mm bands survive blur, distance and bad prints (see the figure), and the hole makes them hard to mistake for clutter.
- **ring-6 vs ring**: equally robust here, ~20% more accurate and usable from further away, because print defects move a larger circle's centre less and it has more edge pixels. The cost is the black margin: 1.0 mm instead of 1.5 mm. With less margin, sheen and blur bridge the ring to its surroundings more easily, and the reading of what surrounds each ring (used to reject a ring-sized washer) picks up the rim: on the plaque it read 185 instead of ~95 at two pins.
- **The dot** is nearly as good as the ring once its detector is as careful (§5.1): 86/92 photos against 87/92, and the same median accuracy. What it gives up: a slightly longer error tail (95th percentile 0.076 vs 0.069 mm, worst 0.16 vs 0.08 mm, with one edge instead of two to average), a little range (3.5 vs 2.5 px/mm), and ~40 false detections per photo on busy tables that ordering then has to sort out.
- **Bullseye and ring-x** have the most precise centres when they are found (ring-x's saddle point is good to ~0.01 mm on a clean print), but their 0.8 mm features need 4 px/mm, fail with shake, and distort visibly in a bad print. The saddle point also follows print defects near the pin, so its worst case (0.34 mm) is the worst of any ring-type design.
- **ArUco** cannot work here. Square markers big enough for 0.4 mm-nozzle cells don't leave a dark quiet zone inside a round-ended 8 mm link (the 5.6 mm marker loses both end pins every time), corner localisation is 2–5× worse than an ellipse fit, and the cells need ≥6.5 px/mm. Ids would solve ordering and mirror detection, but ordering is already reliable.

### 5.1 The dot, looked at more closely

A first-pass dot detector recovered only 71/92 photos. The dot's accuracy was never the problem: on photos both designs recovered, the worst pin was 0.0235 mm for the dot against 0.0238 mm for the ring. What failed was telling dots from everything else. A ring has to be a bright band with exactly one dark hole, which almost nothing on a table is; a dot only has to be a bright ellipse, which terrazzo chips, wood-grain highlights, specks, sheen and the counters of printed letters all are (up to 1,800 false detections in one photo). The fixes, first made in the study and now the production detector (`detect_dots` in `splinewire/detect.py`):

- **The dark link around it.** Along each of up to 180 rays, the margin just outside the dot (1.15–1.45 radii, inside the link's edge) must be darker than the dot's inside by a steady fraction: the median ratio under 0.92, and 60% of rays within ±0.15 of it. The ratio rather than the difference, because a shadow edge darkens dot and margin alike. Terrazzo went from ~300 false detections per photo to ~100, and the chain was recovered in every terrazzo and bare-chain photo.
- **Outline only.** A dot wider than the threshold window comes out hollow at full resolution; candidates use the outer outline and ignore holes.
- **Shape in ordering** (production `order.py`, helps the ring too): neighbouring pins must have about the same ellipse shape, since all lie on one plane. The white counter inside a printed "O" is a narrower oval; two had joined a chain.

| | ring | dot, first pass | dot |
|---|---|---|---|
| chain recovered | 87/92 | 71/92 | 86/92 |
| worst pin: median · 95th pct · max | 0.026 · 0.069 · 0.081 mm | 0.025 · 0.071 · 0.090 mm | 0.025 · 0.076 · 0.159 mm |
| false detections per photo | 14 | 104 | 43 |
| lowest px/mm, all photos | 2.5 | 2.5 | 3.5 |
| hand shake (8 px/mm) | 16 px | 16 px | 16 px |

The dot loses six photos the ring recovers and wins three the ring loses. Its losses are one to three dots missed where the lamp's sheen lifts the black margin to within ~10% of the dot (the ring still has its hole's second edge there), a defocused or shaken dot whose outline starts out distorted, and one terrazzo photo where a chip joined the chain. Its largest error (0.16 mm, a shaken photo of a printed page) is twice the ring's worst.

**The dot on the full benchmark** (§2's 167 photos, rendered with dots; `cv_benchmark.py`, `--fiducial ring` for the ring):

| | ring | dot |
|---|---|---|
| chain recovered | 155/167 | 142/167 (143 since, see below) |
| worst pin: median · 95th pct · max | 0.023 · 0.056 · 0.130 mm | 0.025 · 0.056 · 0.151 mm |
| random setups, matte / standard / glossy | 32/32 · 32/34 · 10/14 | 30/32 · 28/34 · 9/14 |

Of the dot's extra losses, most are dots missed under sheen: the default lamp sits near the reflection angle for one of the three view directions, and standard PLA's link then reads within ~10% of the dot (four of the six presets the dot fails, all on that view). The rest are look-alikes on busy tables (terrazzo, fabric, printed pages) joining the chain. Two changes for the dot also went into ordering for both fiducials: no walk counts as longer than the chain's pin count, with ties going to the walk whose detections stand out most (a woven fabric had strung 58 highlights into a "chain"); and each pin must stay within 1.6× of its walk's median size.

**The bend limit, and two later fixes (2026-11).** Everything above was measured with `max_bend_deg: 80`. `chain.yaml` now sets 125 for the O-ring chain, and a string of look-alikes may then turn more sharply too: at 125 the dot recovered 140/167 (two printed-page photos lost) and the ring 154/167. Reading the ordering again found two flaws in how walks of equal length were compared. A walk longer than the chain counts as no longer than it, but its contrast was still added up over all of it, so 14 letters in a row beat the 13 dots; and walks over the same pins in another order, which should tie on contrast and be told apart by their steps off the pitch, were told apart by the last bit of a floating-point sum (23 of 3000 simulated zigzags at the O-ring chain's stop came out in the wrong order). With both fixed the dot recovers 143/167 at either limit: the two printed-page photos and one terrazzo photo are back, none is lost and the error figures are the same. The ring is unchanged, 155 at 80 and 154 at 125. `uv run python experiments/cv_benchmark.py --baseline <rev>` shows such a change photo by photo with `--json`.

**Using the dot:** 5 mm on the 8 mm link (1.5 mm of black margin, which the detector relies on); matte black, which matters more for the dot than for the ring; keep lamps out of the reflection.

**Recommendation.** The plain ring, 5.0/2.0 mm, is the most robust choice, printed as windows in a two-layer black top over white, in **matte** black. The chain uses the 5 mm dot because it is easier to make; it is as accurate, and with matte filament nearly as robust (30/32 vs 32/32 random setups). If the real chain's links can be ~9 mm wide, a 6 mm ring (keeping a 1.5 mm black margin) buys the ring-6 gains without its cost. Rules of thumb for any size: band ≥1.5 mm (≥3.5 lines), centre ≥2 mm, inner/outer ≈ 0.4, ≥1.5 mm of black around it.

For the real chain, print the ring in one piece with the link so its centre is the pin hole's centre to within print accuracy, and keep the pin out of the ring's centre: an edge formed by a pin head would shift the centre by the hole's clearance.

## 6. Photo and print guidance

- **Matte black filament**; no ironing (it makes the top glossy).
- **Keep lamps out of the reflection.** If the black links look grey or shiny on the phone screen, move the light or tilt the phone a little. Diffuse light (a window, an overcast day) is best.
- **Distance barely matters with a full-resolution photo**: rings need ~2.5 px/mm, which a 12 MP phone at 26 mm-equivalent still gives from ~1.2 m. Photos shrunk by a messaging app to ~1600 px wide reach that limit at ~0.5 m.
- **Tilt up to ~40°, any table**: shadows, printed pages, wood, fabric and clutter are all handled. A plain matte surface is still the safest.

## Reproduce

```bash
uv run python experiments/cv_benchmark.py --baseline 752b3a9   # ~15 min first time (renders cached in out/)
uv run python experiments/video_frame_accuracy.py --only fabric random --random 80 --blur 0   # §3.1 (results are cached per photo: move out/video-frame-accuracy/dot/records aside first)
uv run python experiments/fiducial_study.py                     # ~40 min: environments, resolution, blur
uv run python experiments/fiducial_study.py --figure docs/img/fiducials-printed.png
uv run python experiments/contact_sheet.py docs/img/environments.jpg
uv run python experiments/lens_distortion.py
uv run splinewire synth --env glare --shape pipe                 # look at one
```
