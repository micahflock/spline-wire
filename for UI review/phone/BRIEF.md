# Spline Wire phone page: sketch v1 redesign (McMaster + Teenage Engineering)

**Status: approved by m on 2026-11-24 (mockup v2).** Implement this. Ask m before changing anything the mockup or this brief doesn't cover.

Visual reference: `mockup.html` in this folder (open it in a browser; it's self-contained, with the photos embedded). The left phone is interactive. The strip at the top switches theme, the working curve's state (OK / Check / Failed), the engine download and the computer link. m's sketch is `sketch-v1.png`.
The mockup is a **reference for look, layout and behaviour, not code to copy**. Rebuild it in the existing page.

## Scope
- Files: `web/phone/index.html` (markup), `web/src/phone/phone.css` (look), `web/src/phone/main.ts` (behaviour), `web/src/phone/camera-view.ts` (camera view; some overlay colours are hard-coded there), `web/src/photo/photo.css` (photo markers).
- Don't change the pipeline, the measuring engine, the relay protocol, storage or EXIF handling. This is the page's layout and look.
- The photo-overlay change from `claude/kind-darwin-2ao3ta` (pins drawn as outlines, no numbers) is merged into main (#61). Build on it. The mockup already shows its markers. Before starting, check `git ls-remote --heads origin` for newer unmerged work on these files.
- Keep every existing control. Most of them move into Settings (see the list below). Only "Tap its side" goes away.
- Update `web/e2e/phone.spec.ts` selectors as needed. Don't drop any test coverage.

## Page order (top to bottom)
1. **Header:** the wordmark `spline·wire` (mono, lowercase), then `PINS [13] [+] [−]`. This is the **chain's** pin count (today's Settings → Pins in the chain), so it applies to new photos. Keep the sketch's order, + before −.
2. **Chain strip:** a drawing of the chain with one pin per pin count. Links alternate a thick and a thin capsule, like the logo. When the row is full it **wraps serpentine**, so the next row runs back right to left. The new link pops in on +.
3. **Engine progress:** a 2 px line under the header with a tiny mono label ("Measuring engine · first visit, 40 MB once · 62%"). It's hidden once the engine is ready. Engine errors show in the working curve card, with Try again.
4. **Two capture keys:** **Scan** (live video, the orange key) and **Upload** (grey key, the library picker). The Live/Upload mode switch is gone. The system-camera file input moves to Settings → Live camera ("Use the system camera instead") so it stays reachable.
5. **Computer pill:** a lamp, the computer's name and the status. Tapping it opens the pairing details in place: the pairing text, the outbox (sent/queued per curve) and Forget this computer. There are three states:
   - Linked: green lamp, "Linked · sent 18:42".
   - Computer off: amber blinking lamp, "Off · 2 curves queued".
   - Not paired: grey lamp, "Not paired · tap to pair a computer".
6. **Working curve card:** the latest curve, or the tile picked below. It shows the **greyscale photo cropped to the chain**, with the curve drawn over it in blue (white halo, no pin markers).
   - Top left: a faint star. Tapping it makes it a favourite (yellow) and opens a name field in place. Enter or blur saves the name, and the name then sits next to the star. Un-starring clears the name.
   - Top right: the time and the OK / CHECK / FAILED tag.
   - Bottom left, floating: **Swap side** flips the curve to the other side of the chain (today's Inside/Outside).
   - Bottom right: the **expand** key, which opens the expanded view (below). The collapse key sits in the same bottom-right corner of the expanded photo, so the key never jumps corners.
   - Check: 2 px amber border. Failed: 2 px red border, no curve, and a problem box over the photo (the ValueError text plus advice). There's no Swap side on a failed curve.
   - If the photo was deleted ("photos kept" limit), the card shows the chain drawing on the card colour, with no note.
7. **History grid:** 4 columns, newest first, 12 tiles. Each tile shows **only the chain and the curve, no photo**. The star and name float on top when set. A small amber or red dot marks a Check or Failed curve. Tapping a tile makes it the working curve and scrolls to the top.
8. **`•••` over a dashed rule:** expands to the full history (the same grid, all rows). It shows ▲ to collapse.
9. **Settings** (below the fold, a big uppercase heading on a 2 px rule). Groups are mono legends on hairlines, and rows are label left, control right, McMaster style:
   - **This curve:** Sending status, **Copy points** (the one heavy ink key), Share files, Send again, downloads CSV · DXF · SVG · JSON. Curves send automatically to the paired computer and queue when it's off, as today.
   - **Chain:** Chain preset, Colours.
   - **Live camera:** Speed, Shows. Plus the system-camera fallback (see 4).
   - **Focal length:** the phone's focal length, the lenses list, Teach from a photo, the override, the offer ("Use it / No"), and the error line.
   - **Photos:** Photos kept, space used, Delete photos keep curves, Clear history, the privacy line.
   - **For a good photo:** the tips.
   - The footer with version and build.

## Expanded view
- Animated expand: the card grows to fill the screen (FLIP from the card's rect, about 320 ms), and collapses back the same way. Respect `prefers-reduced-motion`.
- Top bar: the name (or "Curve 18:42") with the photo's meta line, and the tag. No key here.
- The photo viewport with the full editor: pin outlines, pan, pinch, wheel zoom, tap a pin to remove it, tap a dot to add one. **Double-tap fits the chain to the view; there is no Fit key.** Single taps wait out the double-tap window (~300 ms) before editing. Swap side floats bottom left; **collapse floats bottom right**, where the card's expand key was.
- Below it, a sheet: Undo, Reset, Rename, Delete; the hint line; **Pins in this photo** stepper (the per-photo override); the **Measurement** spec table (Pins found, Curve points, Camera tilt, Link length error rms+max, Focal length + source, Object side), with check rows in amber; then the warnings and notes.
- **Check: a 4 px amber border around the whole expanded view. Failed: red.**
- Remove "Tap its side" everywhere, including the code path that picks the side by tapping the photo (`side-tap`). Swap side is the only side control.

## Camera view (Scan)
- Same layout as today, recoloured, **minus the shutter button and the "Upload instead" pill** (m, 2026-11-24). The photo is taken only by the existing auto-capture once the chain holds still. The hold progress (today the shutter's ring) becomes a thin orange bar under the coach's words. Keep the close key and the camera menu.
- The trace is white links with green pin rings, the count is mono ("13 / 13 PINS") with the found number in green, and the coach title is uppercase condensed.
- **All the diagnostics (`#cam-info`) go behind a `DEBUG` key** at the bottom left. It's closed by default, and when open it's a panel just above it. The teach-focal button stays where it is.

## Look
- **McMaster:** near-white page `#F6F6F4`, white cards, hairline rules `#D5D5D0`, mono spec tables with right-aligned tabular numbers, small uppercase mono legends. Corners 2–3 px. No shadows except the keys' bottom edge.
- **Teenage Engineering:** flat grey keys `#E3E3DF` with a 4 px darker bottom edge, tiny corner legends (A, B) on the capture keys, and one colour per job:
  - Orange `#FF5A14`: capture (Scan key, the camera's hold bar).
  - Blue `#2456F0`: the curve (photo overlay, tiles, engine line).
  - Green `#12A150`: linked, sent, pins found.
  - Yellow `#FFBF00`: favourites.
  - Ink key `#232323`: Copy points.
  - Amber `#EF8A00` = Check, red `#DC2F1C` = Failed. These are the semantic colours, kept apart from the accents.
- **Dark theme:** page `#111`, cards `#191919`, rules `#2C2C2C`, ink `#EDEDEA`, lighter accents (blue `#6F93FF`, orange `#FF6A26`, green `#2FC46E`). The ink key inverts to bone white. The photo stays greyscale in both themes.
- **Type:** keep the bundled IBM Plex Sans Condensed (UI) and IBM Plex Mono (data, wordmark, legends). No new font files.
- Put all the colours in `:root` tokens in `phone.css`, with a dark block. Tap targets stay at least 40–44 px.

## Done when
- The page matches the mockup in light and dark, in all three working-curve states and all three computer states.
- "Tap its side" is gone. Every other control is reachable.
- `npm test`, `npm run typecheck` and the phone e2e pass.
- Screenshots of the new page (light, dark, expanded, camera with Debug open) are in the PR.
