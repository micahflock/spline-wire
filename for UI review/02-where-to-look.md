# Where to look: existing UI information

There is no dedicated UI design doc. UI decisions are scattered through the docs, the code and the git history.
This is the map. Paths are relative to the repo root.

## 1. Docs that say something about the UI

| Where | What's in it | Read when |
|---|---|---|
| `README.md` → "How you use it", "How the software works", "Results so far" | The intended user flow (wrap, lay flat, photograph, paste) and the claim "two clicks from photo to sketch". Also two UI screenshots. | First. |
| `docs/architecture.md` → **"App — `webapp/` (SplineWire.exe)"** (end of file) | **The best single description of the current UI**: every control on the desktop page and phone page, what clicking does, the security model, the Windows Firewall handling. Also the section on **Hand corrections** (`edits.py`), which explains what the pin editor's clicks actually do (snap to a dot, else fit to pitch). | Before reviewing the editor or the phone/computer link. |
| `docs/next-steps.md` → **"Known issues from the first real-chain photos"** and **item 4 "Phone → computer"** | Three UI-relevant defects the author already knows about (look-alike warnings coloured orange; the truth file applied to every photo; phone uploads losing EXIF), plus open tests on the iPhone (does the focal length survive the upload; timing; the firewall). | To avoid reporting known issues as new, and to see what's still unverified on real hardware. |
| `docs/open-questions.md` → **"Workflow and CAD"** and **"Contact model"** | Open design questions with a UI side: *"The object side must be right. The app now draws the contact edge on the photo and lets you pick the side there; does that make mistakes rare in practice?"*; *"is 'zoom in and click the middle of where the dot should be' enough guidance?"* for hand-placed pins; origin/orientation in CAD; Fusion script vs add-in. | When judging the side picker and the editor. |
| `docs/fusion-import.md` | Why the Fusion path is a clipboard add-in; every option compared by clicks and scale risk; the manual test protocol. | When reviewing the Copy → Paste hand-off. |
| `docs/open-questions.md` → "Accuracy in the real world" | Why the app exposes **focal length** settings at all (iOS strips EXIF on web upload). | When judging the Settings dialog's Camera section. |
| `docs/img/real-chain-app.jpg` | **The only screenshot of the app with a real photo** (13-pin O-ring chain on a dotted mousepad). Taken on an earlier build; layout has since gained the per-photo pin/side row. | For what real data looks like in the UI. |
| `docs/img/hero.jpg`, `docs/img/real-chain-fusion.png` | The phone photo with the overlay, and the curve pasted into a Fusion sketch. | For the end result of the flow. |

## 2. The code (single files, no framework)

All UI is in two self-contained HTML files with inline `<style>` and `<script>`.

| File | How to read it |
|---|---|
| `splinewire/webapp/static/desktop.html` | CSS tokens (light and dark, `:root` and `@media (prefers-color-scheme: dark)`) are at the top: `--bg --panel --ink --muted --line --accent --ok --warn --bad`. Then markup (header, phone card, photo list, `<dialog id="settings">`). Script: `render()` and `renderMain()` build the page from `state`; `titleCard()`/`optionsRow()` the top card; `viewsCard()` + `curveSvg()` the two views; `detailsCard()` the diagnostics; `renderNetwork()` the phone-connection panel and its firewall help; `statusLine()` the text and colour for each photo's status; the pin editor is the block starting `// Pin editor:` (`mountEditor`, `drawOverlay`, `hint`, `onEditorClick`, `sendEdits`). The settings dialog is at the end. |
| `splinewire/webapp/static/phone.html` | `card()`/`setStatus()` upload cards, `watch()` polling, `describe()` the focal-length line, `showFix()`/`draw()` the post-measurement photo with tap-to-switch-side, `stepper()` the +/− control. |
| `splinewire/webapp/server.py` | Behind the UI: `AppState._summary()`, `.detail()`, `._view()` produce the JSON each page renders; `_api()` lists every endpoint. Status values are `queued`, `processing`, `done`, `error`. Defaults the pages pick up (pin count, side) come from `state.defaults`. |
| `splinewire/pipeline.py` (search for `warnings.append`) | **Every message the user can see as a warning**, and the conditions behind them. The error strings that surface raw in the UI (e.g. "need at least 5 links to rectify, got 1") come from `splinewire/rectify.py` and `pipeline.py` (`raise ValueError(...)`). |
| `splinewire/settings.py`, `server.py` `_with_defaults`/`_migrate_chain` | What Settings persists (`%APPDATA%\SplineWire\settings.json`, or `~/.config/splinewire/settings.json` off Windows). |
| `splinewire/webapp/firewall.py` | The Windows-only state behind the "Allow phone connections" notice. |
| `fusion/SplineWire/SplineWire.py` | The Fusion add-in: button names, tooltips and the message boxes. |
| `tests/test_webapp.py` | What the UI's HTTP contract promises, and the states tests cover. |

Windows specifics (**Allow phone connections**, `%APPDATA%`, the Fusion add-in folder) cannot be exercised on
Linux; those screenshots here are mocked or absent. See `screenshots/README.md`.

## 3. Git history (why it looks like this)

`git log --oneline` has the story; the UI-related commits worth reading:

- `d1ca941` Replace the Tkinter app with a local web app and phone upload (why it is a web page at all)
- `f5092e4` App: version in title bar; toolbar layout that survives display scaling
- `fd2d2a6` Detect Windows Firewall blocking the phone and fix it in one click
- `a9892ae`, `a8b5a1c` Pin editor in the app; remove or add pins by hand, with click snapping
- `e5f0467` Per-photo pin count and object side, on desktop and phone (the newest big UI change)
- `9e1348b` Docs: first photos of the real chain, pasted into Fusion

## 4. Running it yourself

```bash
uv sync
uv run splinewire-app --no-browser --workspace /tmp/sw-ws     # http://localhost:8765/
```

No chain? Make simulated photos (all from the repo's photo simulator) and drop them on the page:
`uv run python "for UI review/tools/make_sample_photos.py" /tmp/sw-photos`. Use `APPDATA=/tmp/x` to keep
settings out of `~/.config`. The whole screenshot set can be re-taken with
`for UI review/tools/capture_screenshots.mjs` (`tools/README.md`).

To see the phone page as a phone sees it, open `http://<this machine's non-loopback IP>:8765/phone#t=<token>`
(the token is at `/api/local-token` on localhost). The server treats any non-loopback client as "the phone",
so loopback access gets the desktop treatment even on `/phone`.

## 5. Project rules that constrain a redesign

From `CLAUDE.md` and the architecture doc:

- **Packaged by PyInstaller.** The two HTML files are bundled as data files; `uv run splinewire-app --selftest log.txt`
  fetches them from the packaged build. A UI change must keep that working (no external requests, nothing
  fetched from a CDN: the app runs offline, and the Content-Security-Policy in `server.py::_page`
  allows only `'self'` plus inline script/style).
- **No framework, no build step** today. Introducing one is a decision for the owner, not a default.
- The title and pages show the version and CI build (`{{VERSION}}` is substituted by the server).
- Tests are pytest; the UI is exercised over HTTP, not in a browser. A browser-level test does not exist.
- The user is a maker, not a developer: copy should be plain. Engineering strings leaking into the UI are a
  defect, not a feature.
