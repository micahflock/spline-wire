# Phone page redesign: implementation plan

Implements `BRIEF.md` (approved mockup v2) in `web/phone/index.html`, `web/src/phone/phone.css`,
`web/src/phone/main.ts`, `web/src/phone/camera-view.ts` and `web/src/photo/photo.css`. Started 2026-11-25
on `claude/optimistic-franklin-ofrlqc`. No unmerged branch touches these files (the five remote
branches without a merge base predate the history on main).

## What the code already gives

- **Tiles:** `doc.curve.pins` and `doc.curve.spline` are in mm and survive "photo deleted", so a tile
  draws the chain and the curve without the photo.
- **Card crop:** `doc.view.pins` and `doc.view.curve` are in photo px; the card crops the grey preview
  to the pins' box and draws `view.curve` in blue with a white halo.
- **Star and name:** `setStar` / `setName` in `send.ts` already sync to the computer.
- **Computer name:** the desktop sends `hello { device }`; the phone ignores it today. Keep it (page
  state only, no protocol change). "Sent 18:42" is the time the outbox last settled a curve, kept by the page.
- **Editor:** `PhotoEditor` (`src/photo/view.ts`) is shared with the desktop page, so double-tap-to-fit
  goes in a phone-side tap filter (new `src/phone/taps.ts`), not in the editor.
- **Camera:** the shutter is already gone (#61). The hold shows as lock-on rings; the bar is new.

## Steps (each one leaves the page working)

1. **Look and shell.** Colour tokens in `phone.css` (light, dark), keys, legends. Header: wordmark,
   `PINS [n] [+] [−]` (the chain's pin count, today's Settings → Pins in the chain), chain strip
   (new `src/phone/drawings.ts`: serpentine layout, alternating thick and thin links, pop on +).
   Engine line (2 px + mono label; errors go to the card). Scan / Upload keys; the mode switch goes
   (`settings.photoMode` stays in storage, unused); the system-camera input moves to Settings → Live camera.
2. **Computer pill.** Lamp + name + status in three states (linked, off and queued, not paired), and,
   on tap, the pairing text, outbox and Forget in place.
3. **Working curve card.** Grey photo cropped to the chain, blue curve with halo; star and name field;
   time + OK / CHECK / FAILED; Swap side; expand key; Check / Failed borders; problem box; "Photo deleted ·
   curve kept" drawing. Remove "Tap its side" and the `picking` path.
4. **History grid.** 4 columns, 12 tiles, newest first, chain + curve drawings, star + name, amber or red
   dot; tap selects and scrolls up; `•••` / `▲` toggles the full history.
5. **Settings.** Groups: This curve (sending status, Copy points ink key, Share, Send, CSV · DXF · SVG ·
   JSON · Fusion CSV), Chain, Live camera (+ system camera), Focal length, Photos, For a good photo, footer.
6. **Expanded view.** FLIP from the card's rect (~320 ms, none under reduced motion); top bar (name,
   meta, tag); the editor with Swap side and collapse floating; double-tap fits, single taps wait ~300 ms;
   sheet: Undo, Reset, Rename, Delete, hint, Pins in this photo, Measurement spec table, warnings, notes;
   4 px amber or red border.
7. **Camera view.** Recolour (white links, green pin rings, mono count with the found number green,
   uppercase coach title), remove "Upload instead", orange hold bar under the coach's words, diagnostics
   behind a `DEBUG` key.
8. **Tests and docs.** Unit tests for `drawings.ts` and `taps.ts`; update `e2e/phone.spec.ts` and the
   camera specs' selectors (the side test becomes Swap side; per-photo delete moves to the expanded
   view); `npm test`, `npm run typecheck`, phone e2e. Screenshots (light, dark, expanded, camera with
   Debug open) for the PR. `docs/web-build.md` Status and Log, `web/README.md` code map.

## Defaults taken where the brief is silent (to confirm with m)

- **Camera lock-on rings:** kept, with the new orange bar added under the coach's words.
- **Trace colour while adjusting:** the pin rings keep today's phase colour (amber while adjusting,
  green once steady); the links are white.
- **Downloads:** Fusion CSV stays beside CSV · DXF · SVG · JSON ("keep every existing control").
- **While measuring:** the card shows the photo with a `…` tag and the status line ("Measuring…").
- **Theme:** follows the system only; the mockup's theme switch is part of the bench, not the page.

- **Measurement table:** a phone-only `measurementRows()` in `wording.ts` (the brief's rows); `numbers()`
  stays as it was for the desktop page, which also uses it.
- **Send:** "Send to computer" stays as today (shown for a curve taken before pairing); there is no
  separate "Send again".

## Progress

- [x] 1 Look and shell
- [x] 2 Computer pill
- [x] 3 Working curve card
- [x] 4 History grid
- [x] 5 Settings
- [x] 6 Expanded view
- [x] 7 Camera view
- [x] 8 Tests and docs (unit 1218 + 1 skipped; e2e phone 18, camera specs 18; screenshots in
  `web/test-results/phone/` and `web/test-results/camera/`, to attach to the PR)

Left: try it on a phone; confirm the defaults above with m; a PR when asked.
