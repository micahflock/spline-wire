# For UI review

A hand-off package for an independent review of Spline Wire's user interface. Everything a reviewer needs to start
without reading the whole repo. It adds files only; no app code was changed.

## Start here

| Read | For |
|---|---|
| [01-what-the-program-does.md](01-what-the-program-does.md) | What the product is, who uses it and when, how it works (photo → CAD points), the end-to-end flow, and which surfaces are in scope. Condensed from the [GitHub README](../README.md). |
| [02-where-to-look.md](02-where-to-look.md) | Where existing UI information lives: docs, code (by function), git history, known issues, open questions; how to run the app; constraints on a redesign. |
| [03-ui-inventory-and-observations.md](03-ui-inventory-and-observations.md) | Every screen, control and state with the screenshot that shows it, then 14 first-pass observations (marked seen / code / docs), the owner's own open questions, and what could not be checked. |
| [screenshots/](screenshots/README.md) | 31 screenshots: desktop (light, dark, narrow), the phone page, errors, editor states, Settings. Indexed. |
| [tools/](tools/README.md) | The scripts that made the screenshots, to retake them after a change. |

## Three things to know before looking at the pictures

1. **The photos in the screenshots are simulated.** The product has only been run on a few real photos. Real-photo
   screenshots of the app and the Fusion result are in `docs/img/` (`real-chain-app.jpg`, `real-chain-fusion.png`, `hero.jpg`).
2. **Three screenshots are mocked** (`d15`–`d17`, the Windows Firewall / "phone connected" notices) and one forced
   (`d14`, the drop overlay). The captions say so. Windows rendering, iPhone Safari and the Fusion add-in's own UI were not seen.
3. **The user is a maker, not a developer or a CAD specialist**, working at a bench with a phone in one hand. The aim is
   photo-to-sketch in under a minute, and a UI mistake (wrong pin count, wrong side of the chain) costs
   accuracy, not just polish. See the "why the UI asks what it asks" table in `01`.

## Scope

In scope: the desktop page, the phone page, the Settings dialog, the console window's first impression, and the
Copy → Paste hand-off to Fusion. Out of scope unless asked: the command line, the 3D-printed chain designs, the
detection algorithms.

## Contents

```
for UI review/
├── README.md
├── 01-what-the-program-does.md
├── 02-where-to-look.md
├── 03-ui-inventory-and-observations.md
├── screenshots/            31 PNG + README.md (index)
└── tools/                  capture_screenshots.mjs, make_sample_photos.py, README.md
```

Taken against version 0.7.0 (dev), the `claude/sweet-allen-vyjeyd` branch. If the UI has changed since, retake with `tools/`.
