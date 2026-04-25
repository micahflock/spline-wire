from __future__ import annotations

PROMPT_V2_RING_GLYPH = """\
You are looking at a photograph of a planar test strip containing a \
horizontal row of fiducial tiles.

Each tile has TWO features:
1. A RING (an annulus / hollow circle) — the position reference. Treat \
   the ring's center as the tile's position.
2. A 7-SEGMENT-DISPLAY-STYLE HEX DIGIT to the right of the ring — the ID \
   reference. Characters are from 0-9, A, b, C, d, E, F (mixed case, \
   industry-standard 7-segment hex convention).

The ring → digit axis is horizontal (digit to the right of the ring) for \
every tile.

For EVERY tile visible in the image, return a JSON array with one object \
per tile:

[
  {
    "tile_id": "<character as string, e.g. '0' or 'A' or 'b'>",
    "circle_xy_px": [<ring center x>, <ring center y>],
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


def prompt_for_strip() -> str:
    """Return the extraction prompt for the canonical ring + 7-seg-hex strip."""
    return PROMPT_V2_RING_GLYPH
