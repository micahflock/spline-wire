# Screenshots

31 PNGs of Spline Wire 0.7.0 (dev build), taken with Playwright/Chromium against a running app.
`d` = desktop page, `p` = phone page. Desktop is 1440 × 900 at 1× unless noted; phone is an iPhone 14
viewport (390 CSS px) at 2×. "Full page" shots are the whole scrollable page.

**All photos are simulated** (the repo's photo simulator), not real chains. Real-photo screenshots of the app and
of the Fusion result are in `docs/img/` (see `../02-where-to-look.md`). The path shown in the Settings
dialog (`d13b`, `d19`) is a scratch folder from the machine they were taken on.

Marked **mocked** = the real page, but its data was altered in the browser to reach a state that only occurs on
Windows or with a real phone. Marked **forced** = a normally momentary state held on.

## Desktop

| File | Shows |
|---|---|
| `d01-empty-no-photos` | First run: QR code, "Phone can't connect?" help collapsed, empty list, empty-state card |
| `d02-photo-measured-viewport` | A measured photo (`pipe-clutter`, 13 pins), what fits in 900 px |
| `d03-photo-measured-full-page` | Same, full page: title card, curve, pin editor, details, a photo list holding all the states |
| `d04-editor-hover-pin` | Pin editor: hovering a pin ("Pin 5. Click to remove it.") |
| `d05-editor-pin-removed-gap` | Pin 6 removed: dashed ghost, gap in the chain, Undo/Reset enabled |
| `d05b-editor-added-pin-that-does-not-fit` | A pin added away from the chain: dashed blue, and the hint says it doesn't fit |
| `d06-pick-side-mode` | **Pick on photo** active: button reads "Click the photo…", hint changes |
| `d07-side-flipped` | After **Flip side**: curve and edge move to the other side of the chain (`s-curve` photo) |
| `d08-copy-points-toast` | The toast after **Copy points** |
| `d08b-zoom-to-chain-with-stray-detections` | `s-curve` photo on a busy background: the chain stays small in the editor (observation 1) |
| `d09-photo-error-glare-washed-out-dots` | Failure: a lamp reflection hides the dots. Red status, generic help card, editor still offered |
| `d10-photo-error-no-chain-in-photo` | Failure: nothing chain-like in the photo |
| `d11-photo-steep-tilt` | Measured with no warnings (green status), steep camera angle |
| `d12-photo-concave-cove` | A concave (cove) shape on a wood table, with warnings |
| `d13-settings-dialog` | Settings dialog in a 900 px window (scrolls inside) |
| `d13b-settings-dialog-whole` | Settings dialog, whole, in a 1150 px window |
| `d14-drop-overlay` | Drag-and-drop overlay (**forced**) |
| `d15-phone-panel-firewall-warning-mocked` | Windows Firewall blocking the phone (**mocked**; cropped to the left 420 px) |
| `d16-phone-panel-help-expanded-mocked` | Same, with the five-step help open (**mocked**) |
| `d17-phone-panel-connected-mocked` | "✓ Your phone reached Spline Wire" (**mocked**) |
| `d18-dark-photo-measured-full-page` | Dark mode (OS colour scheme), full page |
| `d19-dark-settings-dialog` | Dark mode, Settings |
| `d20-narrow-820px-full-page` | 820 px wide: single column, list above the content |

## Phone

| File | Shows |
|---|---|
| `p01-phone-start` | The page after scanning the QR code |
| `p02-phone-no-token` | Opened without the QR link: red "needs the link from the QR code" card |
| `p03-phone-uploading-or-measuring` | Right after choosing a photo: thumbnail and status (a transient state; the exact moment may vary) |
| `p04-phone-measured-with-fix-view` | Measured with warnings: chain and contact edge on the photo, tap-to-switch, pin stepper |
| `p05-phone-side-tapped` | A tap on the far side of the chain: dashed tap marker, "Measuring again…" |
| `p06-phone-error-no-chain` | Failure card (no chain in the photo) |
| `p07-phone-several-photos` | Two failures and one success stacked |
| `p08-phone-dark-measured` | Dark mode |

To retake all of them: `../tools/README.md`.
