# Fusion add-in paired with the phone (design notes)

[← Back to the README](../README.md) · [future-work.md](future-work.md) · [web-build.md](web-build.md) · [fusion-import.md](fusion-import.md)

Research of 2026-11-18. The spike below ran in Fusion on Windows on 2026-11-19 (results at the end); macOS is deferred (D25 in web-build.md). The `?fusion` page and the add-in are next. The goal: curves go phone → relay → Fusion with no desktop page and no clipboard (was O7). The owner's aim is the fewest steps from "measure with chain" to "points in CAD".

## Recommendation

The add-in opens an **HTML palette that loads the hosted desktop page in a trimmed `?fusion` mode** (`https://micahflock.github.io/spline-wire/?fusion`). The page reuses the relay client, the pairing, the crypto and the inbox unchanged; Python only validates and inserts. **First step: a spike** (below) that the owner runs once in Fusion on Windows and on a Mac, before anything else is built.

## Today's pairing and relay (what is reused)

- **Pairing secret.** The desktop page makes one 128-bit secret on first use (localStorage, `splinewire-desktop:`; `web/src/relay/pairing.ts:127-133`, `web/src/desktop/main.ts:55,60`). The QR is the phone page's URL with `#pair=<secret>` (`pairing.ts:140-142`). The phone stores it and strips it from the URL; it holds one pairing at a time (`pairing.ts:153-161`). No expiry; Unpair makes a new secret (`desktop/main.ts:505-516`).
- **Crypto.** HKDF gives a room id (seen by the relay) and an AES-GCM key (not seen). Frames are `{"v":1,"s":role,"n","c"}` with room and role bound in (`relay/crypto.ts:10-14,59-104`).
- **Relay.** Both pages open `wss://…/room/<room>`; the relay forwards sealed frames, answers pings and announces peer counts (`web/relay/` `rules.ts:15-20`). Limits: 4 sockets a room, 64 KB a frame, 60 frames per 10 s (`rules.ts:25-32`). Only the origin `https://micahflock.github.io` is accepted, and no Origin header is refused (`worker.ts:45-47`, `rules.ts:117-119`).
- **Delivery.** Messages `hello`, `curve`, `ack` (`relay/messages.ts:302-307`); a curve has points and pins in mm, object side, diagnostics and gzipped files, validated on arrival (≤ 2000 points, |coord| ≤ 1e5 mm; `messages.ts:268,317-380`). The phone resends until acked (`outbox.ts:153-182`); the desktop stores by id and always acks (`inbox.ts:41-55`). Reconnect with backoff 1–30 s and jitter, ping every 25 s (`link.ts:61-71,171-218`).

## Options considered

| Option | Verdict |
|---|---|
| **(a) HTML palette loading the hosted page** | **Chosen.** Fusion palettes use the Qt web browser (the CEF flag is ignored) and can load an https URL ([Palettes.add](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/core_Palettes_add.htm), [palettes](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/Palettes_UM.htm)). JS → Python: `adsk.fusionSendData(action, data)` (a Promise in Qt) → `incomingFromHTML`, reply with `returnData`. Python → JS: `palette.sendInfoToHTML(action, data)` → `window.fusionJavaScriptHandler.handle` (must return a non-empty string). Palettes are deleted on a workspace switch and must be recreated. |
| (b) WebSocket client in the add-in's Python | No. Fusion's Python has no AES-GCM and no WebSocket client (vendored per-OS wheels or pure-Python AES), macOS builds may lack a CA bundle, the add-in would have to fake an Origin header, and the link/outbox/crypto would be written twice. Threading itself is workable (thread → queue → `app.fireCustomEvent`; [threading](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/Threading_UM.htm)). |
| (c) Localhost bridge from the desktop page, or an add-in watching the clipboard | No. The first still needs the browser page and meets Chrome's local-network prompt and Safari's https→http block; the second still needs Copy. |

A palette page loaded from a **local file** sends Origin `null`, which the relay refuses: the palette must load the hosted page.

## Unknowns only real Fusion answers (the spike)

- The Qt browser's Chromium version (forums call it old). The build targets es2022 with top-level await (`desktop/main.ts:92`): Chromium ≥ 89.
- When `window.adsk` appears; `isSecureContext`, WebCrypto HKDF and AES-GCM; a WebSocket to the relay.
- Whether the palette's localStorage survives a Fusion restart (if not, the add-in keeps the secret in a file under its user data and hands it to the page).
- One undo step for an insert; palette recreation after a workspace switch.
- O5 (the add-in install path) is itself still unverified in Fusion.

## Web changes for `?fusion`

- `src/desktop/fusion.ts`: a bridge with an injectable `send`, detecting `window.adsk?.fusionSendData` (`?fusion` only sets the layout); versioned actions, e.g. `{v:1, type:"insert", id, photo, points_mm}`; status back from Python (the active sketch's name).
- A narrow layout (~320 px): hide downloads, the add-in help and drop-to-measure (D9). Links shown as text (they must not navigate the palette).
- Its own pairing and QR (the palette's storage is not the browser's). The phone holds one pairing, so pairing Fusion replaces the desktop page's pairing.

## UX

- **Add from phone** in Sketch → Create (and Insert) toggles the palette: the QR until a phone pairs, then folded away as on the desktop page.
- A curve arrives **while a sketch is being edited**: auto-insert (on by default, a toggle), with a note "13 points into Sketch3 — Ctrl+Z to undo". **No sketch active**: the curve waits in the list with **Insert**, a command with a plane/planar-face selection (XY by default) that creates the sketch.
- Insertion runs as a command (`fireCustomEvent` → `cmdDef.execute()` → the refactored `paste_points`, `fusion/SplineWire/SplineWire.py:91-118`): one undo step, and no deadlock from API calls inside HTML event handlers.
- Curves stay in the list and can be inserted again. Units: mm on the wire, /10 for the API. Origin and orientation as today (first pin, photo axes); an "align to a point and a line" option (`sketch.move`) can come later (future-work.md).

## Security

The pairing (128-bit secret, end-to-end AES-GCM, a relay that sees nothing) is enough: whoever holds the secret can push undoable sketch geometry, not run code. Fold the QR away once paired; keep Unpair. New trust point: the GitHub account and the Pages deploy now control JavaScript running inside Fusion, so Python is the boundary and must accept only a fixed set of actions, parse JSON (never evaluate), cap the data size, take 2–2000 finite points with |coord| ≤ 1e5 mm, cap name and text lengths, and reject an unknown `v` with a message rather than a traceback.

## Effort (about 4–5 days)

1. Spike (½ day, below).
2. `?fusion` mode and `fusion.ts` (1–1.5 days).
3. Add-in: palette, insert command, validation (pure Python, e.g. `curvedata.py`), workspace recreation (1–1.5 days).
4. Tests (1 day): Playwright with a stubbed `adsk.fusionSendData`, paired with the test phone through the dev relay (pairing, insert calls, acks, reload); pytest with `tests/test_fusion_addin.py`'s fake `adsk` extended to palettes, HTML events, custom events and `execute`.

Risks: the Qt Chromium's age; a site deploy breaking the add-in's contract (hence versioned actions); the 4-socket room cap; one pairing per phone.

## The spike

As S1–S5 in web-build.md: a page `web/spikes/fusion-palette/` (deployed with the site) and a ~60-line add-in `SplineWirePaletteSpike` that opens it in a palette. The page reports the user agent and Chromium version, when `adsk` appears, `isSecureContext`, HKDF and AES-GCM, a relay ping/pong and peer count, whether localStorage survives a restart, and a `fusionSendData` round trip; a button inserts a fixed 3-point spline through custom event → command (check: one Ctrl+Z). The owner runs it on Windows and macOS; results go in web-build.md's Log.

**Built 2026-11-19:** the page `web/spikes/fusion-palette/` (its README has the owner's steps for Windows and macOS) and the add-in `fusion/SplineWirePaletteSpike/`, tested outside Fusion (`e2e/fusion-palette.spec.ts`, `tests/test_fusion_addin.py`). Results (S6 in web-build.md, Log 2026-11-19):

| Question | Windows, Fusion 2705.1.30 | macOS |
|---|---|---|
| Browser | `Neutron/2705.1.30`, no Chrome token; the site's es2022 build with top-level await runs | not run (D25) |
| `window.adsk` | appears 142 ms after navigation, 41 ms after the first script: wait for it | |
| `isSecureContext`, HKDF, AES-GCM | yes, ok, ok | |
| Live relay from the palette | accepted (Origin is the site's); connect 402 ms, ping/pong 57 ms | |
| `fusionSendData` | returns a Promise; round trip 1 ms | |
| Insert as custom event → command | one Ctrl+Z: the spline only in an open sketch; the spline and its new sketch otherwise | |
| `sendInfoToHTML` push | arrives (66 ms click to push) | |
| localStorage across a Fusion restart | kept: the pairing can live in the palette | |
| Workspace switch | the palette stayed (recreation kept as a safeguard) | |
| Clipboard | Copy does not work; the `?fusion` page offers none | |

