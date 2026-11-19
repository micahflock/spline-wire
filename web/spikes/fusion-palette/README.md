# Fusion palette spike

A test page for the Fusion add-in that pairs with the phone directly (design and unknowns:
[docs/fusion-palette.md](../../../docs/fusion-palette.md)). Before the `?fusion` desktop page and the
real add-in are built, this answers what only Fusion can: which browser its palettes have, and whether
the site's relay client, crypto, storage and a one-step insert work there. Nothing else is built until
the owner's results are in.

- `index.html` starts with a **watchdog**, a classic script in old JavaScript: it reports the user agent
  and Chromium version, when `window.adsk` appears, `isSecureContext`, any error, and copies the
  summary. It receives what the add-in pushes (`window.fusionJavaScriptHandler.handle`). If the module
  below never runs (a Chromium older than 89), it says so, and the summary still copies.
- `main.ts` is an es2022 module with a top-level await, as the site's pages are. It uses the site's own
  `src/relay/crypto.ts` (HKDF, AES-GCM, a tampered frame refused), opens two sockets to the live relay
  in one room (ping/pong, peer count 2, a sealed frame across), records loads and add-in starts in
  localStorage (so a restart of Fusion shows whether it survives), and talks to the add-in. Its
  actions, which `fusion/SplineWirePaletteSpike/` answers, are listed at its top.
- `fusion/SplineWirePaletteSpike/` (in the repo root) is the add-in: it opens this page in a palette from
  its https address (a local file sends Origin `null`, which the relay refuses), answers
  `adsk.fusionSendData` with `returnData`, inserts a fixed 3-point spline through
  `app.fireCustomEvent` → a command's `execute()` (one undo step) and pushes the result back with
  `sendInfoToHTML`, and makes the palette again after a workspace switch deletes it.

Tests without Fusion: `e2e/fusion-palette.spec.ts` (Playwright, a dev relay and a stubbed
`adsk.fusionSendData`, both as a Promise and as plain text) and `tests/test_fusion_addin.py`
(pytest, a fake `adsk` with palettes, HTML events, custom events and commands).

## For the owner

The page must be live first: it is at `https://micahflock.github.io/spline-wire/spikes/fusion-palette/`
once this branch is merged and the Deploy workflow has run. Do the following once in Fusion on Windows
and once on a Mac. About 10 minutes each.

1. **Get the add-in.** On GitHub, Code → Download ZIP, unzip it, and keep the folder
   `fusion/SplineWirePaletteSpike` somewhere permanent (it is used where it is, not copied).
2. **Add it.** In Fusion: Utilities tab → Add-Ins (or Shift+S) → the Add-Ins tab → the **+** next to
   My Add-Ins → pick the `SplineWirePaletteSpike` folder. Select **SplineWirePaletteSpike** and press
   **Run**. A palette, **Spline Wire spike**, opens. (If you close it: Utilities → Add-Ins panel →
   **Spline Wire spike**.)
3. **Wait 10 seconds** and look at sections 1 and 2. If the palette stays blank, write down what it
   shows instead.
4. **Insert into a sketch.** Create a sketch on the XY plane and, while it is open, press
   **Insert a test spline**. A spline through three points, 40 mm wide and 10 mm high, should appear in
   that sketch, and section 3 should say so. Press Ctrl+Z (Cmd+Z) **once** and pick what happened in
   the list under the button.
5. **Insert with no sketch open.** Finish the sketch, press **Insert a test spline** again: a new sketch,
   "Spline Wire spike", should appear with the spline. Press Ctrl+Z (Cmd+Z) once and note in your
   message whether it removed the spline only, or the sketch too.
6. **Switch workspaces.** Switch to another workspace (e.g. Design → Render) and back to Design. Is the
   palette there again on its own? Press **Ask Fusion again**: the "palette" line counts the switches and
   how often the palette had to be made again.
7. **Restart Fusion.** Quit Fusion, start it again, open Utilities → Add-Ins, select
   **SplineWirePaletteSpike** and press **Run**. In section 2, "localStorage across restarts" should now
   say "yes" or stay "not tested yet" (that means storage was lost).
8. **Send the results.** Press **Copy the summary** and paste it into a message, with anything you saw
   that the summary does not say (a blank palette, a link opening where it should not, the palette's
   look). If Copy fails, press **Save it to a file**: it writes `spline-wire-palette-spike.txt` in your
   home folder (`C:\Users\<you>` or `/Users/<you>`); attach that.
9. **Remove it.** Add-Ins dialog → select **SplineWirePaletteSpike** → **Stop**, then remove it from the
   list (the − button or its menu) and delete the folder.
