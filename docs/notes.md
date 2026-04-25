# Notes & Reminders

Scratchpad for reminders, stray thoughts, and things to revisit. Not a design doc — messy is fine.

## Open reminders

- **Tentative rename:** project is tentatively renamed from `spline-wire` → `spline-link` (noted 2026-04-22). Repo directory, CLAUDE.md, and docs still use `spline-wire`. Decide whether to finalize before anything ships.

## Random notes

- `data/test-strip-spec.yaml` has `aruco_dict: DICT_4X4_50` for Set C, but the loader doesn't read it and `TileSet` has no field for it. Decide when Task 7/9 gets to ArUco real-photo extraction: either drop the YAML key or plumb it through `TileSet.aruco_dict`. For now it's harmless metadata.
- `validation/schemas.py` imports `dataclasses.field` but doesn't use it. Plan-prescribed; leave until a later polish pass.
