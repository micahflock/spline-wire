"""Process one photo end to end and write its output files.

Shared by the CLI and the web app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from PIL import Image

from splinewire.camera import focal_px_from_35mm
from splinewire.chain import ChainSpec
from splinewire.contact import Side
from splinewire.detect import Fiducial
from splinewire.edits import Edits
from splinewire.output import (
    render_preview, write_csv, write_dxf, write_fusion_csv, write_image, write_json, write_svg,
)
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
    focal_source: str = "exif"    # "exif", "override", "default" or "estimated"
    info: dict = field(default_factory=dict)   # what the photo file carried; see photo_info


_EXIF_IFD = 0x8769
_TAGS = {"make": 0x010F, "model": 0x0110}
_EXIF_TAGS = {"focal_mm": 0x920A, "focal_35mm": 0xA405}


def photo_info(path: Path) -> dict:
    """Format, size and camera metadata of a photo file.

    Shown after a phone upload, because some upload paths strip metadata
    (iOS Safari has at times dropped EXIF, including the focal length).
    """
    with Image.open(path) as im:
        exif = im.getexif()
        sub = exif.get_ifd(_EXIF_IFD)
        info = {"format": im.format, "width": im.size[0], "height": im.size[1]}
        for key, tag in _TAGS.items():
            value = exif.get(tag)
            info[key] = str(value).strip("\x00 ") if value else None
        for key, tag in _EXIF_TAGS.items():
            value = sub.get(tag) or exif.get(tag)
            info[key] = float(value) if value else None
    return info


def process_photo(
    photo: Path,
    spec: ChainSpec,
    out_dir: Path,
    focal_35mm: float | None = None,
    side: Side = "inside",
    truth_path: Path | None = None,
    default_focal_35mm: float | None = None,
    edits: Edits | None = None,
    loaded: tuple[np.ndarray, float | None] | None = None,
    detected: list[Fiducial] | None = None,
) -> PhotoResult:
    """Measure the chain in `photo` and write JSON, CSV, SVG and preview to out_dir.

    Focal length, in order of preference: focal_35mm (an explicit override),
    the photo's EXIF, default_focal_35mm (e.g. the user's phone camera, for
    uploads that lost their EXIF), or an estimate from the chain itself.

    edits: pins removed or added by hand (splinewire.edits). loaded and
    detected: what load_photo and detect_fiducials already returned for this
    photo, to measure it again without repeating them (e.g. after an edit).
    """
    photo = Path(photo)
    image, focal = loaded if loaded is not None else load_photo(photo)
    size = (image.shape[1], image.shape[0])
    source = "exif" if focal is not None else "estimated"
    if focal_35mm is not None:
        focal, source = focal_px_from_35mm(focal_35mm, size), "override"
    elif focal is None and default_focal_35mm is not None:
        focal, source = focal_px_from_35mm(default_focal_35mm, size), "default"
    m = measure(image, spec, focal, side=side, edits=edits, detected=detected)

    extra: dict = {"photo": photo.name}
    if edits:
        extra["edits"] = edits.to_json()
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
        "dxf": out_dir / f"{stem}-curve.dxf",
        "fusion_csv": out_dir / f"{stem}-fusion-cm.csv",
        "preview": out_dir / f"{stem}-preview.jpg",
    }
    write_json(outputs["json"], m, extra)
    write_csv(outputs["csv"], m.contacts_mm)
    write_svg(outputs["svg"], m)
    write_dxf(outputs["dxf"], m.contacts_mm)
    write_fusion_csv(outputs["fusion_csv"], m.contacts_mm)
    preview = render_preview(image, m)
    write_image(outputs["preview"], preview)
    return PhotoResult(
        photo=photo, measurement=m, image=image, preview=preview,
        truth_comparison=truth_comparison, outputs=outputs,
        focal_source=source, info=photo_info(photo),
    )
