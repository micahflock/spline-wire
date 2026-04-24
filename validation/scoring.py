from __future__ import annotations

import numpy as np

from validation.homography import apply_homography, fit_homography
from validation.schemas import (
    GroundTruthTile,
    ImageDetection,
    ScoringResult,
)


def score_detection(
    detection: ImageDetection,
    ground_truth: list[GroundTruthTile],
) -> ScoringResult:
    truth_by_id = {t.tile_id: t for t in ground_truth}
    matched = [d for d in detection.detections if d.tile_id in truth_by_id]
    if len(matched) < 4:
        raise ValueError(
            f"need at least 4 matched detections to fit homography; "
            f"got {len(matched)}"
        )

    pts_px = np.array([d.circle_xy_px for d in matched], dtype=float)
    pts_mm = np.array(
        [truth_by_id[d.tile_id].center_mm for d in matched], dtype=float
    )
    H = fit_homography(pts_px, pts_mm)

    projected_mm = apply_homography(H, pts_px)

    per_point: dict[str, float] = {}
    errors_mm: list[float] = []
    for d, proj in zip(matched, projected_mm):
        truth_xy = np.array(truth_by_id[d.tile_id].center_mm)
        err = float(np.linalg.norm(proj - truth_xy))
        per_point[d.tile_id] = err
        errors_mm.append(err)

    mean_err = float(np.mean(errors_mm)) if errors_mm else 0.0

    # homography reprojection RMSE in px: project truth-mm back to pixels
    # via inverse H and compare to detected pixels
    H_inv = np.linalg.inv(H)
    reproj_px = apply_homography(H_inv, pts_mm)
    rmse_px = float(np.sqrt(np.mean(np.sum((reproj_px - pts_px) ** 2, axis=1))))

    return ScoringResult(
        image_path=detection.image_path,
        per_point_error_mm=per_point,
        mean_error_mm=mean_err,
        homography_rmse_px=rmse_px,
    )
