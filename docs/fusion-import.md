# Getting the curve into Fusion — feasibility

Goal (next-steps item 3): measured curve in the active Fusion sketch, correctly scaled in mm, with as little clicking as possible. Points are acceptable; a spline is a bonus.

## What Fusion can and can't do (researched 2026-09)

- **No native paste of coordinates.** Fusion's clipboard only carries its own sketch geometry; there is no way to paste text coordinates into a sketch, and no built-in "type in a table of points" tool.
- **Built-in `ImportSplineCSV` sample script** (Utilities > Scripts and Add-Ins). Reads headerless lines of `x,y,z` in **centimeters** (Fusion's internal API unit; the script doesn't convert), always makes a **new sketch on the XY plane**, and adds a fitted spline through the points.
- **Insert > Insert DXF** keeps scale when the file declares its units (`$INSUNITS`); if it doesn't, the document units are assumed.
- **Insert > Insert SVG** has a history of wrong scale (pixel-per-inch assumptions); not recommended for dimensional work.
- **Add-ins** (Python, installed per user) can add a toolbar button that reads the clipboard and creates sketch points and fitted splines (`sketchPoints.add`, `sketchFittedSplines.add`) in the sketch being edited, in one undo step. Add-in commands can be promoted onto a toolbar panel.

## Options compared

| Route | One-time setup | Clicks per curve | Lands in active sketch? | Scale risk |
|---|---|---|---|---|
| **Add-in "Paste points"** (recommended) | Install add-in (button in SplineWire.exe), restart Fusion | Copy points → Paste points (2) | Yes (else new XY sketch) | None: mm → cm in code |
| Insert DXF (`-curve.dxf`) | None | ~6: Insert > Insert DXF, pick plane, browse to file, OK | Its own sketch on a chosen plane | Low: file says mm |
| ImportSplineCSV (`-fusion-cm.csv`) | None | ~5: Scripts and Add-Ins, pick script, Run, browse to file | No: always a new XY sketch | Low: file already in cm |
| Insert SVG (`-curve.svg`) | None | ~6 | Its own sketch | High: known scale issues |
| Paste text into a sketch | — | Not possible | — | — |

The clipboard idea only works with the add-in; the "table" idea is covered by the same copied text, which is a tab-separated `x_mm / y_mm` table that also pastes cleanly into a spreadsheet.

## What's built

- `SplineWire.exe`:
  - **Copy points** (or Ctrl+C in the photo list) puts the selected photo's curve points on the clipboard as that table.
  - **Install Fusion add-in…** copies `fusion/SplineWire` into `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\SplineWire`.
  - Every processed photo also writes `-curve.dxf` (mm, spline + points on separate layers) and `-fusion-cm.csv` (for ImportSplineCSV) as no-install fallbacks.
- `fusion/SplineWire/` add-in: a **Paste points** button in the sketch Create panel and the Insert menu. It reads the clipboard (tab/comma/space separated, optional header, mm by default, `cm`/`in` headers honored), adds a sketch point at each curve point and a fitted spline through them, in the sketch being edited or a new "Spline Wire" sketch on XY. One undo step. The spline can be deleted to keep only the points.

Tested outside Fusion only: the add-in runs in the test suite against a stand-in for Fusion's API (unit conversion, sketch choice, toolbar registration and cleanup), and the DXF is read back to check units and that the spline passes through every point. **Nothing has run inside Fusion yet.**

One API detail is unverified: whether `sketchFittedSplines.add` accepts existing sketch points as fit points (Autodesk's samples pass plain coordinates). The add-in tries sketch points first and falls back to coordinates, which leaves a second coincident point at each spot. Either way the pasted points survive deleting the spline.

## Test protocol (in Fusion)

1. SplineWire.exe → **Install Fusion add-in…** → restart Fusion. Check that **Paste points** appears in Sketch > Create and in Insert. If not: Utilities > Add-Ins, select Spline Wire, Run, tick Run on Startup.
2. Process a photo (a synthetic one from `splinewire synth` is fine) → **Copy points**.
3. In Fusion, create or edit a sketch → **Paste points**. Check:
   - the points and spline appear in *that* sketch;
   - a distance between two points, measured with Inspect > Measure, matches the JSON/CSV values in mm;
   - one Ctrl+Z removes everything pasted;
   - timing from Copy to visible points (target < 5 s).
4. Paste with no sketch open → a new "Spline Wire" sketch on the XY plane.
5. Paste with junk on the clipboard → a friendly message, nothing created.
6. Fallbacks: Insert > Insert DXF with `-curve.dxf` (check the units field says mm and the size matches), and ImportSplineCSV with `-fusion-cm.csv`.
7. Optional: see whether Fusion lets you assign a keyboard shortcut to Paste points (toolbar commands have a ⋮ menu with a shortcut option; unconfirmed for add-in commands). That would make pasting a single keystroke.
