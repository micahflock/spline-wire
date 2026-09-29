"""Process one photo end to end and write its output files.

Shared by the CLI and the GUI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from splinewire.camera import focal_px_from_35mm
from splinewire.chain import ChainSpec
from splinewire.contact import Side
from splinewire.output import render_preview, write_csv, write_image, write_json, write_svg
from splinewire.pipeline import Measurement, compare_to_truth, load_photo, measure

PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp")


@dataclass
class PhotoResult:
    photo: Path
    measurement: Measurement
    image: np.ndarray                       # grayscale photo, upright
    preview: np.ndarray                     # BGR photo with detections drawn on
    truth_comparison: dict | None = None
    outputs: dict[str, Path] = field(default_factory=dict)


def process_photo(
    photo: Path,
    spec: ChainSpec,
    out_dir: Path,
    focal_35mm: float | None = None,
    side: Side = "inside",
    truth_path: Path | None = None,
) -> PhotoResult:
    """Measure the chain in `photo` and write JSON, CSV, SVG and preview to out_dir.

    focal_35mm overrides the photo's EXIF focal length when given.
    """
    photo = Path(photo)
    image, focal = load_photo(photo)
    if focal_35mm is not None:
        focal = focal_px_from_35mm(focal_35mm, (image.shape[1], image.shape[0]))
    m = measure(image, spec, focal, side=side)

    extra: dict = {"photo": photo.name}
    truth_comparison = None
    if truth_path is not None:
        truth = np.array(json.loads(Path(truth_path).read_text(encoding="utf-8"))["pin_points"])
        truth_comparison = compare_to_truth(m.pins_mm, truth)
        extra["truth_comparison"] = truth_comparison

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = photo.stem
    outputs = {
        "json": out_dir / f"{stem}.json",
        "csv": out_dir / f"{stem}-curve.csv",
        "svg": out_dir / f"{stem}-curve.svg",
        "preview": out_dir / f"{stem}-preview.jpg",
    }
    write_json(outputs["json"], m, extra)
    write_csv(outputs["csv"], m.contacts_mm)
    write_svg(outputs["svg"], m)
    preview = render_preview(image, m)
    write_image(outputs["preview"], preview)
    return PhotoResult(
        photo=photo, measurement=m, image=image, preview=preview,
        truth_comparison=truth_comparison, outputs=outputs,
    )
