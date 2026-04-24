from __future__ import annotations

import math

import numpy as np

from validation.homography import apply_homography, fit_homography
from validation.schemas import (
    GroundTruthTile,
    ImageDetection,
    ScoringResult,
)

# Distance between circle center and glyph center along the tile's orientation
# axis. Must match GLYPH_OFFSET_MM in validation/synthetic.py.
GLYPH_OFFSET_MM = 3.0


def score_detection(
    detection: ImageDetection,
    ground_truth: list[GroundTruthTile],
) -> ScoringResult:
    truth_by_id = {t.tile_id: t for t in ground_truth}
    matched = [d for d in detection.detections if d.tile_id in truth_by_id]

    # Collect correspondences. Each matched detection contributes its circle.
    # If the detection also reports a glyph, add that as a second correspondence
    # offset by GLYPH_OFFSET_MM along the tile's orientation axis — this breaks
    # collinearity when all circles sit on a single strip row.
    corr_px: list[tuple[float, float]] = []
    corr_mm: list[tuple[float, float]] = []
    for d in matched:
        tile = truth_by_id[d.tile_id]
        corr_px.append(d.circle_xy_px)
        corr_mm.append(tile.center_mm)
        if d.glyph_xy_px is not None:
            rad = math.radians(tile.orientation_deg)
            glyph_mm = (
                tile.center_mm[0] + GLYPH_OFFSET_MM * math.cos(rad),
                tile.center_mm[1] + GLYPH_OFFSET_MM * math.sin(rad),
            )
            corr_px.append(d.glyph_xy_px)
            corr_mm.append(glyph_mm)

    if len(corr_px) < 4:
        raise ValueError(
            f"need at least 4 correspondences to fit homography; "
            f"got {len(corr_px)} (from {len(matched)} matched detections)"
        )

    pts_px = np.array(corr_px, dtype=float)
    pts_mm = np.array(corr_mm, dtype=float)
    H = fit_homography(pts_px, pts_mm)

    # Per-tile error is the circle-center reprojection error vs ground truth.
    # The circle is the tile's position reference per the extraction prompt.
    circle_px = np.array([d.circle_xy_px for d in matched], dtype=float)
    circle_mm = np.array(
        [truth_by_id[d.tile_id].center_mm for d in matched], dtype=float
    )
    projected_mm = apply_homography(H, circle_px)

    per_point: dict[str, float] = {}
    errors_mm: list[float] = []
    for d, proj, truth_xy in zip(matched, projected_mm, circle_mm):
        err = float(np.linalg.norm(proj - truth_xy))
        per_point[d.tile_id] = err
        errors_mm.append(err)

    mean_err = float(np.mean(errors_mm)) if errors_mm else 0.0

    # Homography RMSE across all correspondences (circles and glyphs).
    H_inv = np.linalg.inv(H)
    reproj_px = apply_homography(H_inv, pts_mm)
    rmse_px = float(np.sqrt(np.mean(np.sum((reproj_px - pts_px) ** 2, axis=1))))

    return ScoringResult(
        image_path=detection.image_path,
        per_point_error_mm=per_point,
        mean_error_mm=mean_err,
        homography_rmse_px=rmse_px,
    )
