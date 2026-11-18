"""One photo, measured in memory: bytes in, JSON-ready documents out, no files.

This is the entry point of the browser build (docs/web-build.md, step 1): the
web worker holds one PhotoSession per photo and calls it over postMessage.
`process.process_photo` is a wrapper that adds the files, so the CLI, the
local app and the browser cannot drift apart.

A session keeps what is slow to get (the decoded photo, its detections), so
measuring again after an edit, a new pin count or a new object side only
orders and solves.

    session = PhotoSession.open(data, "IMG_0001.jpg")
    doc = session.measure({"chain": {...}, "side": "inside"})     # see measure()
    files = session.outputs()                                     # name -> text
"""
from __future__ import annotations

import io
import traceback
from dataclasses import dataclass, field, replace

import numpy as np
from PIL import Image

from splinewire.camera import (
    EXIF_IFD, TAG_FOCAL_LENGTH_35MM, focal_35mm_from_px, focal_px_from_35mm,
)
from splinewire.chain import ChainSpec, spec_from_dict
from splinewire.contact import Side, spline_samples
from splinewire.detect import Fiducial, detect_fiducials
from splinewire.edits import Edits, remove_detections
from splinewire.output import csv_text, dxf_text, fusion_csv_text, json_text, points_tsv, svg_text
from splinewire.pipeline import Measurement, compare_to_truth, load_photo_bytes, measure

# The document measure() returns. Bump the number when a field changes meaning
# or goes away; adding a field does not need it. web/src/core/types.ts mirrors it.
SCHEMA = "spline-wire/measurement@1"

# A photo of the test part matches its truth file to well under a millimetre
# (1 mm is the pass mark). Pins this far off, after fitting position, angle and
# scale, mean a different shape: not the test part, so the error means nothing.
NOT_THE_TEST_PART_PITCHES = 0.5

_TAGS = {"make": 0x010F, "model": 0x0110}
_EXIF_TAGS = {"focal_mm": 0x920A, "focal_35mm": TAG_FOCAL_LENGTH_35MM}


def photo_info(source) -> dict:
    """Format, size and camera metadata of a photo (a path or a binary file).

    Shown with each photo, because some ways of handing a photo to a web page
    strip its metadata (iOS Safari has at times dropped EXIF, including the
    focal length).
    """
    with Image.open(source) as im:
        exif = im.getexif()
        sub = exif.get_ifd(EXIF_IFD)
        info = {"format": im.format, "width": im.size[0], "height": im.size[1]}
        for key, tag in _TAGS.items():
            value = exif.get(tag)
            info[key] = str(value).strip("\x00 ") if value else None
        for key, tag in _EXIF_TAGS.items():
            value = sub.get(tag) or exif.get(tag)
            info[key] = float(value) if value else None
    return info


@dataclass(frozen=True)
class Truth:
    """Known pin positions of a test part, to score a measurement against."""
    name: str                   # the file's name, for messages
    pins_mm: np.ndarray

    @classmethod
    def from_json(cls, doc, name: str = "the truth file") -> "Truth":
        try:
            pins = np.asarray(doc["pin_points"], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{name} is not a truth file: it has no pin_points list") from None
        if pins.ndim != 2 or pins.shape[1] != 2:
            raise ValueError(f"{name} is not a truth file: pin_points must be [x, y] pairs")
        return cls(name=name, pins_mm=pins)


@dataclass(frozen=True)
class Options:
    """Everything a person can set for one measurement."""
    spec: ChainSpec
    side: Side | tuple[float, float] = "inside"    # or a photo point on the object's side
    focal_override_35mm: float | None = None       # beats the photo's EXIF
    saved_focal_35mm: float | None = None          # this phone's camera, for photos without EXIF
    edits: Edits = field(default_factory=Edits)
    truth: Truth | None = None
    compare_truth: bool | None = None              # None: only when the shape matches too

    @classmethod
    def from_json(cls, doc: dict, size: tuple[int, int] | None = None) -> "Options":
        """Options from the page's JSON, checked: it comes from a browser.

        {"chain": {pitch_mm, half_width_mm, fiducial_mm, n_pins, ...},     required
         "side": "inside" | "outside" | [x, y],
         "focal_override_35mm": number | null, "saved_focal_35mm": number | null,
         "edits": {"add": [[x, y], ...], "remove": [[x, y], ...]},
         "truth": {"name": str, "pin_points": [[x, y], ...]} | null,
         "compare_truth": true | false | null}
        """
        if not isinstance(doc, dict) or not isinstance(doc.get("chain"), dict):
            raise ValueError("options must be an object with a chain")
        try:
            spec = spec_from_dict(doc["chain"])
        except KeyError as err:
            raise ValueError(f"the chain is missing {err.args[0]}") from None
        side = doc.get("side", "inside")
        if not isinstance(side, str):
            try:
                side = (float(side[0]), float(side[1]))
            except (TypeError, ValueError, IndexError):
                raise ValueError("side must be inside, outside or a photo point [x, y]") from None
        elif side not in ("inside", "outside"):
            raise ValueError("side must be inside, outside or a photo point [x, y]")
        truth = doc.get("truth")
        return cls(
            spec=spec, side=side,
            focal_override_35mm=_focal(doc.get("focal_override_35mm"), "focal_override_35mm"),
            saved_focal_35mm=_focal(doc.get("saved_focal_35mm"), "saved_focal_35mm"),
            edits=Edits.from_json(doc.get("edits") or {}, size),
            truth=Truth.from_json(truth, str(truth.get("name") or "the truth file"))
            if isinstance(truth, dict) else None,
            compare_truth=doc.get("compare_truth"),
        )


def _focal(value, name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number of millimetres") from None
    if not 5.0 <= f <= 500.0:
        raise ValueError(f"{name} must be a 35 mm-equivalent focal length between 5 and 500 mm, got {f:g}")
    return f


@dataclass
class Measured:
    """The outcome of one PhotoSession.measure call that found a chain."""
    measurement: Measurement
    options: Options
    focal_source: str                   # "exif", "override", "default" or "estimated"
    truth_comparison: dict | None = None
    # With a truth file: "compared", or why not: "off" (asked not to), "pins"
    # (a different pin count), "shape" (a different shape: not the test part).
    truth_status: str | None = None


class PhotoSession:
    """A photo and what has been found in it so far."""

    def __init__(self, name: str, image: np.ndarray, exif_focal_px: float | None, info: dict,
                 detected: list[Fiducial] | None = None) -> None:
        self.name = name
        self.image = image                      # grayscale, upright
        self.exif_focal_px = exif_focal_px
        self.info = info
        self._detected = detected
        self._detected_for: tuple | None = None
        self.last: Measured | None = None       # the latest successful measurement
        self.last_error: str | None = None
        self._edits = Edits()                   # the edits of the latest attempt, successful or not

    @classmethod
    def open(cls, data: bytes, name: str) -> "PhotoSession":
        """Decode a photo. ValueError (for the person) if it is not one."""
        image, focal = load_photo_bytes(data, name)
        return cls(name, image, focal, photo_info(io.BytesIO(data)))

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def detections(self, spec: ChainSpec) -> list[Fiducial]:
        """The photo's fiducials, found once per kind of fiducial."""
        key = (spec.fiducial, spec.ring_inner_mm / spec.fiducial_mm if spec.fiducial == "ring" else 0.0,
               spec.fiducial_shade)
        if self._detected is None or (self._detected_for is not None and self._detected_for != key):
            self._detected = detect_fiducials(self.image, spec)
        self._detected_for = key
        return self._detected

    def run(self, options: Options) -> Measured:
        """Measure with these options. Raises ValueError with a message for the
        person when no chain can be measured; `measure` reports that instead."""
        size = self.size
        focal, source = self.exif_focal_px, "exif" if self.exif_focal_px is not None else "estimated"
        if options.focal_override_35mm is not None:
            focal, source = focal_px_from_35mm(options.focal_override_35mm, size), "override"
        elif focal is None and options.saved_focal_35mm is not None:
            focal, source = focal_px_from_35mm(options.saved_focal_35mm, size), "default"
        self._edits = options.edits
        self.last = None
        m = measure(self.image, options.spec, focal, side=options.side, edits=options.edits,
                    detected=self.detections(options.spec))
        comparison = status = None
        if options.truth is not None:
            comparison, status, why = _compare(m, options.truth, options.spec, options.compare_truth)
            if why and options.compare_truth:
                m = replace(m, warnings=[*m.warnings, why])     # asked for, and couldn't
            elif why:
                m = replace(m, notes=[*m.notes, why])
        self.last = Measured(m, options, source, comparison, status)
        return self.last

    def measure(self, options: dict | Options) -> dict:
        """Measure and describe the outcome as one JSON-ready document:

        {"schema": SCHEMA, "name": str, "info": photo_info,
         "ok": bool, "error": str | None,          error: for the person, as is
         "result": {...} | None,                   the numbers (see _result)
         "curve": {"points", "pins", "spline"} | None,     mm
         "view": {...}}                            what to draw on the photo (see view)

        A photo in which no chain is found is not an exception: the document
        has ok false, the message, and a view of the detections, so the person
        can fix it by hand. Anything other than a ValueError is a bug and
        comes back as its type and message.
        """
        try:
            if not isinstance(options, Options):
                options = Options.from_json(options, self.size)
            self.run(options)
            self.last_error = None
        except ValueError as err:
            self.last_error = str(err)
        except Exception as err:        # a bug: show what it was, so it can be reported
            self.last_error = "".join(traceback.format_exception_only(type(err), err)).strip()
        return self.document()

    def document(self) -> dict:
        doc = {"schema": SCHEMA, "name": self.name, "info": self.info,
               "ok": self.last is not None, "error": self.last_error,
               "result": None, "curve": None, "view": self.view()}
        if self.last is not None:
            m = self.last.measurement
            doc["result"] = self._result()
            doc["curve"] = {
                "points": _rounded(m.contacts_mm),
                "pins": _rounded(m.pins_mm),
                "spline": _rounded(spline_samples(m.contacts_mm)),
            }
        return doc

    def _result(self) -> dict:
        r = self.last
        m, rect = r.measurement, r.measurement.rectification
        return {
            "pins": len(m.order.indices),
            "missing_pins": len(m.order.gaps),
            "rejected": len(m.order.rejected),
            "curve_points": len(m.contacts_mm),
            "tilt_deg": round(rect.tilt_deg, 1),
            "focal_35mm": round(focal_35mm_from_px(rect.focal_px, self.size), 1),
            "focal_source": r.focal_source,
            "link_residual_rms_mm": round(rect.residual_rms_mm, 3),
            "link_residual_max_mm": round(rect.residual_max_mm, 3),
            "object_side": m.object_side,
            "warnings": list(m.warnings),
            "notes": list(m.notes),
            "truth": r.truth_comparison,
            "truth_status": r.truth_status,
        }

    def view(self) -> dict | None:
        """What the pin editor draws over the photo, in photo pixels: the pins
        on the chain in order, detections left out of it, pins removed by hand
        (as ghosts, to bring them back), and the edits that produced all this.
        Also there when measuring failed, so a photo that could not be read
        can still be fixed by hand. None before anything was detected."""
        if self._detected is None:
            return None
        edits, r = self._edits, self.last

        def marker(f: Fiducial, **extra) -> dict:
            return {"x": round(f.center_px[0], 2), "y": round(f.center_px[1], 2),
                    "r": round(f.outer_axes_px[0] / 2, 1), **extra}

        side_px = None
        if r is not None:
            m = r.measurement
            by_hand = {mp.fiducial: mp for mp in m.manual}

            def mark(i: int, **extra) -> dict:
                mp = by_hand.get(i)
                return marker(m.fiducials[i], **extra, **({"edit": mp.edit, "snapped": mp.snapped} if mp else {}))

            pins = [mark(i, k=k) for k, i in enumerate(m.order.indices)]
            unused = [mark(i) for i in m.order.rejected]
            gaps = list(m.order.gaps)
            # The contact edge, where the object touched the chain, drawn on the photo.
            curve = _rounded(m.to_photo_px(spline_samples(m.contacts_mm, per_segment=6)), 1)
            side_px = list(m.side_px) if m.side_px is not None else None
        else:
            pins, gaps, curve = [], [], []
            unused = [marker(f) for f in remove_detections(self._detected, edits)]
        radii = [p["r"] for p in pins or unused]
        r_default = float(np.median(radii)) if radii else 12.0
        if r is None:      # no measurement to say where added pins ended up: where they were clicked
            unused += [{"x": x, "y": y, "r": r_default, "edit": j, "snapped": False}
                       for j, (x, y) in enumerate(edits.add_px)]
        w, h = self.size
        return {
            "width": w, "height": h, "r": r_default, "pins": pins, "gaps": gaps, "unused": unused,
            "curve": curve, "side_px": side_px,
            "removed": [{"x": x, "y": y, "edit": j} for j, (x, y) in enumerate(edits.remove_px)],
            "edits": edits.to_json(),
        }

    def json_extra(self) -> dict:
        """What the JSON output says beyond the measurement itself."""
        r = self._need()
        m = r.measurement
        extra: dict = {"photo": self.name}
        if r.options.edits:
            extra["edits"] = r.options.edits.to_json()
        if m.side_px is not None:
            extra["object_side_picked_px"] = [round(v, 1) for v in m.side_px]
        if r.truth_comparison is not None:
            extra["truth_comparison"] = r.truth_comparison
        return extra

    def outputs(self) -> dict[str, str]:
        """The output files of the latest measurement, as text, by kind. "tsv"
        is the clipboard table that the Fusion add-in's Paste points reads."""
        m = self._need().measurement
        return {
            "json": json_text(m, self.json_extra()),
            "csv": csv_text(m.contacts_mm),
            "svg": svg_text(m),
            "dxf": dxf_text(m.contacts_mm),
            "fusion_csv": fusion_csv_text(m.contacts_mm),
            "tsv": points_tsv(m.contacts_mm),
        }

    def _need(self) -> Measured:
        if self.last is None:
            raise ValueError("this photo has no measurement yet")
        return self.last


# Output kind -> the file's name after the photo's stem. The browser build and
# process_photo name downloads and files the same way.
OUTPUT_NAMES = {
    "json": "{stem}.json",
    "csv": "{stem}-curve.csv",
    "svg": "{stem}-curve.svg",
    "dxf": "{stem}-curve.dxf",
    "fusion_csv": "{stem}-fusion-cm.csv",
}


def _compare(m: Measurement, truth: Truth, spec: ChainSpec,
             compare: bool | None) -> tuple[dict | None, str, str | None]:
    """The truth comparison, or None with the reason and a message for people."""
    if compare is False:
        return None, "off", None
    if len(truth.pins_mm) != len(m.pins_mm):
        return None, "pins", (f"not compared to {truth.name}: it has {len(truth.pins_mm)} pins "
                              f"and this chain {len(m.pins_mm)}")
    t = compare_to_truth(m.pins_mm, truth.pins_mm)
    off = NOT_THE_TEST_PART_PITCHES * spec.pitch_mm
    if compare is None and t["max_error_scaled_mm"] > off:
        return None, "shape", (f"not compared to {truth.name}: a different shape "
                               f"(pins up to {t['max_error_scaled_mm']:.1f} mm apart), "
                               "so not the test part")
    return t, "compared", None


def _rounded(pts: np.ndarray, digits: int = 4) -> list[list[float]]:
    return [[round(float(x), digits), round(float(y), digits)] for x, y in np.asarray(pts, dtype=float)]
