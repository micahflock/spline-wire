from __future__ import annotations

from pathlib import Path

import yaml

from validation.schemas import FiducialType, GroundTruthTile, StripSpec, TileSet


def load_strip_spec(path: Path) -> StripSpec:
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sets: dict[str, TileSet] = {}
    for name, entry in raw["sets"].items():
        sets[name] = _build_tile_set(name, entry)
    return StripSpec(units=raw["units"], sets=sets)


def _build_tile_set(name: str, entry: dict) -> TileSet:
    fid_type = FiducialType(entry["fiducial_type"])
    origin_x, origin_y = entry["origin_mm"]
    pitch = entry["tile_pitch_mm"]
    count = entry["count"]
    orientation = entry["orientation_deg"]

    if fid_type is FiducialType.CIRCLE_GLYPH:
        alphabet = entry["glyph_alphabet"]
        if len(alphabet) < count:
            raise ValueError(
                f"Set {name}: glyph_alphabet has {len(alphabet)} chars, need {count}"
            )
        ids = list(alphabet[:count])
    else:
        aruco_ids = entry["aruco_ids"]
        if len(aruco_ids) < count:
            raise ValueError(
                f"Set {name}: aruco_ids has {len(aruco_ids)} entries, need {count}"
            )
        ids = [str(i) for i in aruco_ids[:count]]

    tiles = [
        GroundTruthTile(
            tile_id=ids[i],
            center_mm=(origin_x + i * pitch, origin_y),
            orientation_deg=orientation,
        )
        for i in range(count)
    ]
    return TileSet(
        name=name,
        fiducial_type=fid_type,
        tile_face_mm=entry["tile_face_mm"],
        tile_pitch_mm=pitch,
        tiles=tiles,
    )
