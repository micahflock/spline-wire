"""Process one photo end to end and write its output files.

Shared by the CLI and the web app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from splinewire.chain import ChainSpec
from splinewire.contact import Side
from splinewire.detect import Fiducial
from splinewire.edits import Edits
from splinewire.output import render_preview, write_image
from splinewire.pipeline import Measurement, load_photo
from splinewire.session import (  # noqa: F401  (photo_info and the constant are part of this module's API)
    NOT_THE_TEST_PART_PITCHES, OUTPUT_NAMES, Options, PhotoSession, Truth, photo_info,
)

PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp")


@dataclass
class PhotoResult:
    photo: Path
    measurement: Measurement
    image: np.ndarray                       # grayscale photo, upright
    preview: np.ndarray | None              # BGR photo with detections drawn on (the app drops it once shown)
    truth_comparison: dict | None = None
    outputs: dict[str, Path] = field(default_factory=dict)
    focal_source: str = "exif"    # "exif", "override", "default" or "estimated"
    info: dict = field(default_factory=dict)   # what the photo file carried; see photo_info
    # With a truth file: "compared", or why not: "off" (asked not to), "pins"
    # (a different pin count), "shape" (a different shape: not the test part).
    truth_status: str | None = None


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
    session = PhotoSession(photo.name, image, focal, photo_info(photo), detected=detected)
    truth = None
    if truth_path is not None:
        truth_path = Path(truth_path)
        truth = Truth.from_json(json.loads(truth_path.read_text(encoding="utf-8")), truth_path.name)
    r = session.run(Options(
        spec=spec, side=side, focal_override_35mm=focal_35mm, saved_focal_35mm=default_focal_35mm,
        edits=edits or Edits(), truth=truth, compare_truth=compare_truth,
    ))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    texts = session.outputs()
    for kind, pattern in OUTPUT_NAMES.items():
        outputs[kind] = out_dir / pattern.format(stem=photo.stem)
        # newline="": the text carries its own line endings
        with outputs[kind].open("w", encoding="utf-8", newline="") as f:
            f.write(texts[kind])
    outputs["preview"] = out_dir / f"{photo.stem}-preview.jpg"
    preview = render_preview(image, r.measurement)
    write_image(outputs["preview"], preview)
    return PhotoResult(
        photo=photo, measurement=r.measurement, image=image, preview=preview,
        truth_comparison=r.truth_comparison, outputs=outputs,
        focal_source=r.focal_source, info=session.info, truth_status=r.truth_status,
    )
