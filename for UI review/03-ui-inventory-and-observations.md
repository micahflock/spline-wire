# UI inventory and first-pass observations

Everything the user can see or click, with the screenshot that shows it (`screenshots/`, indexed in
`screenshots/README.md`), then observations. **How sure each observation is** is marked:

- **[seen]**: visible in a screenshot in this folder.
- **[code]**: read in the source, not confirmed by looking at it run.
- **[docs]**: stated in the repo's own docs.

These are a starting list from someone who set the screenshots up, not a review. Nothing has been changed in the app.

## A. Desktop page (`desktop.html`, 1440 × 900 in the shots; responsive down to a single column at ≤ 860 px)

### Header (every state, `d01`)
| Control | Does |
|---|---|
| "Spline Wire" + version string | Version and CI build (`0.7.0 (dev)` locally). |
| **Add photos…** | File picker (images, HEIC). Drag and drop onto the window does the same (`d14`). |
| **Settings** | Opens the Settings dialog (below). |
| **Install Fusion add-in** | Copies the add-in into Fusion's folder; reports with a native `alert()` giving the path and restart instructions. |
| **Quit** | Native `confirm()`, then stops the server and replaces the page with "Spline Wire has stopped." |
| Red bar "Spline Wire isn't running. Start SplineWire.exe again…" | Shown when polling fails. Not captured. |

### Left column
**Phone card** (`d01`, `d15`–`d17`). QR code, "Send photos from your iPhone", a sentence on same-Wi-Fi or hotspot, the
phone URL in monospace, and a network area with three states:
- default: a collapsed **"Phone can't connect?"** list of five causes (`d01`);
- Windows Firewall will block the phone: an amber note, **Allow phone connections** (primary) and **Check again**, with the same help expanded (`d15`, `d16`, *mocked*);
- phone has reached the app: green "✓ Your phone reached Spline Wire (last at 05:45 AM)" (`d17`, *mocked*).

**Photo list** (`d03`). One row per photo: thumbnail, file name, a coloured status line, and
"From phone / Added here · time". Empty: "No photos yet." The newest upload becomes the selected photo.

Status line colours: green = fine; amber = has warnings or (when a truth file is loaded) error over 1 mm; red = failed; grey = "Waiting…" / "Measuring…".

### Main column, selected photo (`d02`, `d03`)
1. **Title card.**
   - File name and status line.
   - **Copy points** (primary), with the hint "then **Paste points** in a Fusion sketch". Toast: "Copied 13 points. In Fusion, click Paste points while editing a sketch." (`d08`). Ctrl+C does the same when no text is selected.
   - Download buttons **DXF, CSV, SVG, JSON, Fusion CSV**; and **Remove** (a link-style button).
   - Options row: **Pins in chain** (number input), "Object touched the **inside of the bend**" with **Flip side** and **Pick on photo** (`d06`: while picking, the button reads "Click the photo…", the hint says "Click on the side of the chain that touched the object. Esc to cancel."). Changing either re-measures the photo (`d07`).
2. **Measured curve** (SVG, in mm, with a 10 mm scale bar): the chain's pin line (grey), pins (white circles), the contact curve (black) and its sample points (red).
3. **Pins** (the pin editor). The photo with detections drawn on it, a hint line under it, and a legend.
   Toolbar: **Undo, Reset, Zoom to chain, Whole photo**. Marker colours: green = pin on the chain (numbered),
   blue = pin you added (dashed if there was no dot to snap to), red = a detection left out of the chain (click to use it),
   dashed grey with a cross = a pin you removed (click to restore), pink line = the edge that touched the object.
   - Click a green pin to remove it (`d05` shows the gap, ghost, and the chain re-linked); click empty space to add one (`d05b`); hover shows what a click would do (`d04`); scroll to zoom; drag to pan; Ctrl+Z undoes.
4. **Details card.** "Measurement" rows: pins found, curve points, camera tilt, focal length (and where it came from),
   link-length error (rms / max), object side, stray look-alikes ignored, hand edits, and, when a truth file is loaded,
   error against it. "Photo file" rows: format and size, camera, focal length in file ("missing" in amber when absent).
   Warnings are listed under it (`d03`: "ignored 3 look-alike detection(s) not on the chain").

### Settings dialog (`d13`, `d13b`, `d19`)
Fieldsets: **Chain** (pitch, half width, fiducial dot/ring, fiducial diameter, ring hole diameter, sharpest bend),
**Camera** (iPhone focal length used when a photo has none; override for every photo), **Test part** (load a truth file to
see each photo's error), **Files** (workspace path; Open photos folder / Open results folder). **Cancel / Save**.
Save re-measures every photo. In a 900 px window the dialog scrolls inside itself (`d13` vs `d13b`).

### Other states
| State | Shot |
|---|---|
| No photos yet | `d01` |
| Measured, no warnings | `d11` (steep tilt) |
| Measured, with an amber warning | `d03`, `d12` |
| Failed: dots washed out by glare | `d09` |
| Failed: no chain in the photo | `d10` |
| Concave object (the "cove" shape) | `d12` |
| Dark mode | `d18`, `d19` |
| Narrow window (820 px), one column | `d20` |

## B. Phone page (`phone.html`, iPhone 14 viewport, 390 CSS px wide)

| Element | Notes |
|---|---|
| Title + version | |
| Red "problem" card | No QR token (`p02`: "This page needs the link from the QR code…") or can't reach the computer. |
| **Next photo** card | **Pins in chain** (− / number / +) and **Object** (Inside bend / Outside bend), with one hint. Starts from what the computer last used (`p01`). |
| **Take photo** (primary), **Choose from library** | Take photo opens the camera (`capture="environment"`). |
| Upload card per photo | Thumbnail, name, progress bar, status line, one meta line: format, size, focal length ✓ or why it's missing (`p03`, `p04`). |
| After measuring | "Object touched the **inside of the bend**. Tap the photo on the other side of the chain to switch." The photo with the chain (yellow) and contact edge (pink); a tap shows a dashed marker and re-measures (`p05`); another **Pins in chain** stepper (`p04`). |
| Failure | "Couldn't find the chain in this photo." + the raw reason + "Try again: whole chain in frame, dots facing up, no glare." (`p06`, `p07`) |
| "For a good measurement" card | Four tips, always at the bottom. |
| Dark mode | `p08` |

The phone page cannot edit pins. That is only on the computer.

## C. Fusion add-in (not screenshotted; it is inside Fusion)
A **Paste points** button in Sketch > Create and the Insert menu, tooltip "Paste the points copied from Spline Wire into
the sketch being edited (or a new sketch on the XY plane), with a spline through them." Failure messages are Fusion
message boxes ("Open a design (not a drawing or CAM setup) before pasting points.", or a traceback for unexpected
errors, **[code]**). The pasted result is in [docs/img/real-chain-fusion.png](../docs/img/real-chain-fusion.png).

## D. Observations to start from

Not prioritised. Items 1 and 2 are the ones I'd look at first.

1. **"Zoom to chain" (and the editor's initial view) frame the stray detections too, so the chain can stay small [seen, measured].**
   `chainBox()` in `desktop.html` includes pins *plus* left-out and removed detections. On `s-curve-bare-chain.jpg`
   the 13 pins span 843 × 642 px, but with the 16 stray red detections the box is 3662 × 2500 px of a 4032 × 3024 photo, so
   the chain fills ~20% of the editor and its pins are hard to click (`d08b`; compare `d04`–`d06`, where the chain is large).
   On such photos the initial view is nearly the whole photo already (the editor's view is 4541 px wide on a 4032 px photo), so "Zoom to chain" and "Whole photo" barely differ.
2. **Amber "warning" for a normal result [seen, docs].** "ignored N look-alike detection(s) not on the chain" turns the photo amber in
   the list and on the phone ("Measured, with warnings"). In these shots 3 of the 4 measured photos are amber
   (`d03`, `d12`, `p04`); only the one with no clutter is green (`d11`). The author lists this as a known issue
   ([docs/next-steps.md](../docs/next-steps.md), "Known issues"). It matters more than it looks because the status colour is
   the main "did it work" signal in the list.
3. **Failure messages are engineering strings [seen].** "need at least 5 links to rectify, got 1" (glare) and
   "found 1 fiducials; need the whole chain in view" (no chain) are shown as the status and, on the phone, under "Couldn't find
   the chain in this photo.". The list truncates the second one at the sidebar width ("…need the whole chai…", `d03`).
   Both failures get the same generic advice (`d09`, `d10`) even though the causes differ (washed-out dots vs no chain). On
   the desktop, a failed photo shows the editor below with "or, if you can see the dots, click each pin" (a good
   recovery path, but only mentioned in one clause).
4. **"Remove" is immediate and unlabelled about what it does [code].** It has no confirm and no undo, while **Quit** has a confirm. It only drops the photo from
   the list; files stay on disk (`delete_photo` pops it from memory). The user can't tell that from the label.
5. **Native browser dialogs [code].** **Install Fusion add-in** reports success with `alert()` and **Quit** asks with `confirm()`.
   Everything else uses the toast or in-page UI.
6. **Selection jumps to new uploads [code].** The desktop page selects the newest photo whenever the photo count grows (`refresh()`).
   If someone is mid-edit on photo A and a phone upload arrives, the view switches to B. Possibly intended (follow uploads), possibly a surprise.
7. **Settings leaks internals [seen].** With **Fiducial = Dot**, the "Ring hole diameter (mm, rings only)" field is still shown, showing 0.
   Labels such as "Sharpest bend at a pin (°; 80, or 125 for the O-ring chain)" and "Half width, pin line to edge (mm)" assume
   knowledge of the hardware. Most users will have one chain and never open this. The two things they might need are
   the focal length and the pin count, which are in different places (focal in Settings, pin count per photo).
8. **Narrow layout [seen, `d20`].** At 820 px the layout collapses to one column with the whole photo list *above* the selected photo, so as the
   list grows the result moves further down. The editor toolbar wraps onto two lines.
9. **Undo remains enabled after Reset [seen, `d06`].** After pressing Reset (which is then greyed out), Undo is still live. The undo history survives a reset. Small.
10. **The pin editor has no keyboard path [code].** Only pointer events plus Ctrl+C / Ctrl+Z / Esc. The SVG has `role="img"`. Accessibility has not been audited in any way.
11. **Phone result line wraps oddly [seen, `p04`].** "focal length 26 mm ✓" breaks so the check mark sits alone on its own line.
12. **Two places to set the pin count on the phone [seen, `p04`].** The "Next photo" card at the top and the stepper under each photo; one is for the next photo and one for this photo (changing the second also updates the first). The wording says so only in the hint.
13. **The dark pin-editor background is fixed in light mode [seen, `d03`].** Probably deliberate (photos read better on dark); noting it.
14. **First-run text names iPhone only [seen, `d01`].** "Send photos from your iPhone". The page accepts any image; only iPhone focal-length handling is built in.

## E. Questions the owner has asked themselves (from the docs)

- Does the contact-side picker make wrong-side mistakes rare? ([docs/open-questions.md](../docs/open-questions.md))
- Is "zoom in and click the middle of where the dot should be" enough guidance for hand-placed pins?
- Should the user be able to choose the origin and orientation of the points in CAD (now: the first pin, axes follow the photo)?
- Target: **under 30 s** from photo to curve in Fusion, **under 15 s** from shutter to the photo appearing on the computer ([docs/next-steps.md](../docs/next-steps.md)).
- How do iPhone web uploads behave (EXIF stripped; which button was used)? Both known failures so far are on the iPhone path.

## F. What I could not check

- **Windows rendering.** Shots are Chromium on Linux (the font is the system fallback, not Segoe UI). Real display scaling is untested.
- **Real photos.** All screenshots use simulated photos. Real ones have noisier detections, more look-alikes and slower measuring.
- **The firewall notice** (only runs on Windows); `d15`–`d17` are the real page with the app's `/api/state` response edited in the browser.
- **iPhone Safari itself.** The phone shots are an iPhone 14 viewport and user agent in Chromium. Safari's camera sheet, safe-area insets and
  `capture` behaviour are untested.
- **Anything inside Fusion.**
