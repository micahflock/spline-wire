# What Spline Wire does, and how

A brief for someone reviewing the UI who has not seen the project. It condenses the
[README](../README.md) and [docs/architecture.md](../docs/architecture.md). Where those have more detail, this says so.

## One paragraph

You want a CAD sketch of a curve that is awkward to measure: a pipe profile, a molding, a door handle's
silhouette. You wrap a 3D-printed **chain** around it. The links are rigid and 10 mm pin to pin, and the
joints are stiff enough to hold the pose. You lift the chain off, lay it flat, and photograph it with a phone.
**Spline Wire** turns that photo into millimetre-accurate points on the object's curve and gets them into
Autodesk Fusion in two clicks: **Copy points** in the app, then **Paste points** in Fusion.
The goal is under a minute from object to sketch, with sub-millimetre accuracy and nothing else in the photo
(no ruler, no reference card).

## Who uses it, and where

- A maker / hobbyist / product designer with a 3D printer, who already uses Fusion. Not a computer-vision person,
  not necessarily a CAD expert.
- Setting: at a bench or workshop. Phone in one hand, the chain in the other, the computer nearby. The
  object-to-CAD loop is repeated: several curves in one session is normal.
- Platform: primarily a **Windows** desktop app (`SplineWire.exe`), plus the user's **iPhone** browser. Everything
  is local: no account, no cloud, no phone app. Anything that touches the network is a local Wi-Fi connection.
- The software is one person's project, built with an AI coding agent. There is no UX designer's pass behind the
  current UI; this review is that pass.

## The physical thing (why the UI asks what it asks)

| Fact about the chain | Consequence in the UI |
|---|---|
| Links are exactly one **pitch** (10 mm) apart pin to pin. This is the only critical dimension. | The chain is its own ruler: the software recovers scale and camera tilt from the photo alone. So the UI never asks for a scale or reference object. |
| An identical **light dot** (5 mm) sits on every pin, unlabelled. | The software finds dots and works out their order by geometry. If it misses one or picks up a look-alike (a washer, a table pattern), the user fixes it by clicking pins on the photo (the **pin editor**). |
| Chains can have any number of pins (13 for the current one). | **Pins in chain** is a per-photo input. A wrong count makes the chain unrecoverable. |
| The object touches the chain's *edge*, not its pin line. | The software offsets the curve by the link half-width, either toward the inside or the outside of the bend. It cannot always know which, so **object side** is a per-photo input (inside/outside, or a point picked on the photo). A wrong side puts the curve ~8 mm off. |
| Joints bend at most 74° (O-ring chain) or ~80°. | Not exposed, except as a chain setting (**sharpest bend at a pin**). |

## The pipeline (what happens when a photo arrives)

```
photo ─► detect ─► order ─► rectify ─► contact ─► curve points ─► CAD
         (dots)    (chain)  (perspective)  (offset to edge)
```

1. **Detect.** Classical computer vision (OpenCV) finds the dot centres to ~0.1–0.2 px. No AI, no labels.
   It is built to cope with shadows, glare, wood grain, printed text and look-alike washers. One failure
   remains: a lamp reflected in shiny filament washes the dots out (see screenshot `d09`).
2. **Order.** Starting from each dot, walk one pitch at a time, turning as little as possible. The longest walk is the chain.
3. **Rectify (remove perspective).** All pins lie on one plane and neighbours are exactly one pitch apart. With
   the camera's focal length (from the photo's EXIF data) that fixes the plane's tilt and distance. 13 pins give
   12 equations for 3 unknowns. The focal length is the one number that can go missing: **iPhone uploads from
   the browser lose all EXIF**, so the app falls back to a per-phone setting (24 mm on iPhone 14 Pro and
   later, 26 mm earlier) or estimates it from the chain.
4. **Contact offset.** Offset from the pin line to the object's edge: at link midpoints where the chain wraps
   a convex shape, at the pins where it presses into a hollow. A cubic spline through the contact points is the curve.

Detail: [docs/architecture.md](../docs/architecture.md). Detection robustness and the photo simulator:
[docs/cv-robustness.md](../docs/cv-robustness.md).

## What the app is

A small **local web server** (Python standard library `http.server`) started by `SplineWire.exe` in a console
window. Its UI is **two plain HTML pages** shown in a browser: a desktop page on the computer and a
phone page the phone reaches by scanning a QR code on the desktop page. There is no framework and no build step
(just `desktop.html` and `phone.html` with inline CSS and JS).
Photos are measured by a worker thread, within seconds (re-measuring after a pin edit is faster, because the detections are kept); the pages poll for changes.

### The end-to-end flow

1. Start `SplineWire.exe`. A console window opens (keep it open) and the browser opens the desktop page.
2. On the phone, scan the QR code with the Camera app (same Wi-Fi, or the computer on the phone's hotspot).
   On Windows the firewall often blocks the phone first; the desktop page detects this and offers a one-click
   fix (**Allow phone connections**, one admin prompt).
3. On the phone page, set **Pins in chain** and **Inside/Outside bend** for the next photo, then **Take photo**
   or **Choose from library**. The phone shows "Measuring…", then the result with the chain and the contact
   edge drawn over the photo. Tapping the photo on the other side of the chain flips the side; +/− changes the pin count.
4. On the desktop the photo appears in the list (the newest is selected automatically) with:
   the **measured curve** (SVG, in mm), the **photo with detections drawn on it** (an editor: click to add/remove
   pins, scroll to zoom, drag to pan), the per-photo **pins** and **side** controls, and a diagnostics table.
5. **Copy points** puts a tab-separated x/y table (mm) on the clipboard. In Fusion, **Paste points** (a button the app
   installs via **Install Fusion add-in**) adds sketch points and a fitted spline to the sketch being edited.
6. Fallbacks: DXF (mm), SVG (1:1), CSV, JSON and a Fusion-CSV download for other CAD tools.

Photos can also be added on the computer (**Add photos…** or drag and drop), and there is a command-line
tool (`splinewire measure photo.jpg`) that is not part of the UI review.

## Surfaces the UI review covers

| Surface | File | Notes |
|---|---|---|
| Desktop page (main app) | `splinewire/webapp/static/desktop.html` | ~900 lines, inline CSS + JS |
| Phone page | `splinewire/webapp/static/phone.html` | ~310 lines, phone-first, iPhone Safari |
| Settings dialog | inside `desktop.html` | chain parameters, camera focal length, test-part truth file, folders |
| Fusion add-in | `fusion/SplineWire/SplineWire.py` | One **Paste points** button (Sketch > Create panel and Insert menu) and a couple of message boxes. Not in this folder's screenshots: it lives inside Fusion. A real screenshot of the result is [docs/img/real-chain-fusion.png](../docs/img/real-chain-fusion.png) |
| Console window | `splinewire/webapp/app.py` | Prints the URLs and "keep this window open". Not a designed surface, but it is the first thing users see |
| Command line | `splinewire/cli.py` | Out of scope unless asked |

## What is measured vs what is simulated (read before judging the screenshots)

Almost everything is validated on **simulated photos**. The riskiest open item is real-world accuracy.
The first real photos of the real chain worked (all 13 pins found, link lengths within 0.12 mm RMS of the pitch)
and were pasted into Fusion, but the curve has not yet been checked against an object of known size. See
[docs/next-steps.md](../docs/next-steps.md). The screenshots in this folder use **simulated photos** (so the numbers
shown, like "15° tilt" or "link length error 0.011 mm", are cleaner than real ones). The only real-photo screenshot
of the app is [docs/img/real-chain-app.jpg](../docs/img/real-chain-app.jpg), from an earlier build.

## Status of the product

Working end to end: detect, order, rectify, contact offset; Windows app with phone upload and pin editing;
export to DXF/SVG/CSV; Fusion add-in (tested in Fusion, under 5 s from Copy to points). Not done: a test series
of real photos (angles, lighting, phones), timing the whole path, and accuracy against a physical object.
Not planned for now: a native phone app, other CAD packages, non-planar curves.
