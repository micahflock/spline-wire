from __future__ import annotations

PROMPT_V1_CIRCLE_GLYPH = """\
You are looking at a photograph of a planar test strip containing a \
horizontal row of fiducial tiles.

Each tile has TWO features:
1. A solid dark FILLED CIRCLE (the position reference).
2. A DIGIT from 0-9 printed next to the circle (the ID reference).

Within each tile, the circle and digit are aligned along an axis \
perpendicular to the strip's long direction. Treat the circle's center \
as the tile's position. Treat the digit as the tile's ID.

For EVERY tile visible in the image, return a JSON array with one object \
per tile:

[
  {
    "tile_id": "<digit as string, e.g. '0'>",
    "circle_xy_px": [<circle center x>, <circle center y>],
    "glyph_xy_px": [<digit center x>, <digit center y>],
    "confidence": <float 0-1>
  },
  ...
]

Coordinates are pixel coordinates in the original image, with (0,0) at \
top-left. Use your best estimate of each feature's center. If you cannot \
read the digit, set "tile_id" to "?" and set "confidence" below 0.5.

Return ONLY the JSON array, no prose, no markdown fences.\
"""

PROMPT_V1_ARUCO = """\
You are looking at a photograph of a planar test strip containing a \
horizontal row of ArUco 4x4 fiducial markers.

For EACH visible ArUco marker, return a JSON array with one object per \
marker:

[
  {
    "tile_id": "<marker id as string, e.g. '0'>",
    "circle_xy_px": [<marker center x>, <marker center y>],
    "glyph_xy_px": null,
    "confidence": <float 0-1>
  },
  ...
]

The marker ID is encoded in the black-and-white grid. Coordinates are \
pixel coordinates in the original image, with (0,0) at top-left. Return \
ONLY the JSON array.\
"""


def prompt_for_set(set_name: str, fiducial_type: str) -> str:
    if fiducial_type == "circle_glyph":
        return PROMPT_V1_CIRCLE_GLYPH
    if fiducial_type == "aruco_4x4":
        return PROMPT_V1_ARUCO
    raise ValueError(f"no prompt for fiducial type {fiducial_type}")
