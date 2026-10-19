"""Process one photo end to end and write its output files.

Shared by the CLI and the web app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
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
    # With a truth file: "compared", or why not: "off" (asked not to), "pins"
    # (a different pin count), "shape" (a different shape: not the test part).
    truth_status: str | None = None


# A photo of the test part matches its truth file to well under a millimetre
# (1 mm is the pass mark). Pins this far off, after fitting position, angle and
# scale, mean a different shape: not the test part, so the error means nothing.
NOT_THE_TEST_PART_PITCHES = 0.5


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
    side: Side | tuple[float, float] = "inside",
    truth_path: Path | None = None,
    default_focal_35mm: float | None = None,
    edits: Edits | None = None,
    loaded: tuple[np.ndarray, float | None] | None = None,
    detected: list[Fiducial] | None = None,
    compare_truth: bool | None = None,
) -> PhotoResult:
    """Measure the chain in `photo` and write JSON, CSV, SVG and preview to out_dir.

    Focal length, in order of preference: focal_35mm (an explicit override),
    the photo's EXIF, default_focal_35mm (e.g. the user's phone camera, for
    uploads that lost their EXIF), or an estimate from the chain itself.

    side: "inside" or "outside" of the chain's bend, or a photo point (x, y)
    on the object's side of the chain (see measure).
    edits: pins removed or added by hand (splinewire.edits). loaded and
    detected: what load_photo and detect_fiducials already returned for this
    photo, to measure it again without repeating them (e.g. after an edit).
    compare_truth: with truth_path, True compares whenever the pin counts
    match, False never, None (auto) only when the shape matches too, so a
    leftover truth file does not score photos of something else.
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
    if m.side_px is not None:
        extra["object_side_picked_px"] = [round(v, 1) for v in m.side_px]
    truth_comparison = truth_status = None
    if truth_path is not None:
        truth_comparison, truth_status, why = _compare(m, Path(truth_path), spec, compare_truth)
        if truth_comparison is not None:
            extra["truth_comparison"] = truth_comparison
        elif why and compare_truth:
            m = replace(m, warnings=[*m.warnings, why])     # asked for, and couldn't
        elif why:
            m = replace(m, notes=[*m.notes, why])

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
        focal_source=source, info=photo_info(photo), truth_status=truth_status,
    )


def _compare(m: Measurement, truth_path: Path, spec: ChainSpec,
             compare: bool | None) -> tuple[dict | None, str, str | None]:
    """The truth comparison, or None with the reason and a message for people."""
    if compare is False:
        return None, "off", None
    truth = np.array(json.loads(truth_path.read_text(encoding="utf-8"))["pin_points"])
    if len(truth) != len(m.pins_mm):
        return None, "pins", (f"not compared to {truth_path.name}: it has {len(truth)} pins "
                              f"and this chain {len(m.pins_mm)}")
    t = compare_to_truth(m.pins_mm, truth)
    off = NOT_THE_TEST_PART_PITCHES * spec.pitch_mm
    if compare is None and t["max_error_scaled_mm"] > off:
        return None, "shape", (f"not compared to {truth_path.name}: a different shape "
                               f"(pins up to {t['max_error_scaled_mm']:.1f} mm apart), "
                               "so not the test part")
    return t, "compared", None
