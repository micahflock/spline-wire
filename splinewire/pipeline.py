"""Photo in, curve points out."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # HEIC/HEIF, the iPhone default photo format
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

from splinewire.camera import focal_px_from_exif
from splinewire.chain import ChainSpec
from splinewire.contact import Side, contact_points, object_sign
from splinewire.detect import Fiducial, detect_fiducials
from splinewire.order import ChainOrder, order_chain
from splinewire.rectify import Rectification, rectify_chain


@dataclass(frozen=True)
class Measurement:
    fiducials: list[Fiducial]            # every fiducial detected in the photo
    order: ChainOrder
    rectification: Rectification
    object_side: Side
    contacts_mm: np.ndarray      # points on the target curve, in chain order
    warnings: list[str]

    @property
    def pins_mm(self) -> np.ndarray:
        return self.rectification.pins_mm

    @property
    def pins_px(self) -> np.ndarray:
        return np.array([self.fiducials[i].center_px for i in self.order.indices])


def load_photo(path: Path) -> tuple[np.ndarray, float | None]:
    """Grayscale pixels (upright per EXIF orientation) and EXIF focal length in px.

    Rotating to upright keeps the image diagonal, so the focal length holds.
    """
    with Image.open(path) as im:
        focal = focal_px_from_exif(im)
        upright = ImageOps.exif_transpose(im)
        return np.asarray(upright.convert("L")), focal


def measure(
    image: np.ndarray,
    spec: ChainSpec,
    focal_px: float | None,
    side: Side = "inside",
) -> Measurement:
    warnings: list[str] = []
    fiducials = detect_fiducials(image, spec)
    if len(fiducials) < 3:
        raise ValueError(f"found {len(fiducials)} fiducials; need the whole chain in view")

    h, w = image.shape[:2]
    centers = np.array([r.center_px for r in fiducials])
    excluded: set[int] = set()
    for _ in range(4):
        order = _order(fiducials, spec, excluded)
        rect = rectify_chain(centers[order.indices], order.links, spec.pitch_mm, (w, h), focal_px)
        misfits = _misfits(fiducials, order, rect, spec)
        if not misfits:
            break
        excluded |= misfits
    n_on_chain = len(order.indices)
    if order.rejected:
        warnings.append(f"ignored {len(order.rejected)} look-alike detection(s) not on the chain")
    if order.gaps:
        warnings.append(f"{len(order.gaps)} pin(s) not detected; the curve bridges those gaps")
    expected = spec.n_pins - len(order.gaps)
    if n_on_chain != expected:
        warnings.append(
            f"found {n_on_chain} pins on the chain but the spec has {spec.n_pins} "
            f"({len(order.gaps)} gap(s)); is the whole chain in the photo, and free of glare "
            "(a lamp's reflection on the links)?"
        )
    if focal_px is None:
        warnings.append(
            "no focal length (EXIF or --focal-35mm); estimating it from the chain, "
            "which is unreliable for nearly straight chains"
        )
    if rect.residual_max_mm > 0.05 * spec.pitch_mm:
        warnings.append(
            f"link lengths deviate from the pitch by up to {rect.residual_max_mm:.2f} mm; "
            "check the detections in the preview image"
        )

    sign = object_sign(rect.pins_mm, side)
    contacts = contact_points(rect.pins_mm, spec.half_width_mm, sign)
    return Measurement(
        fiducials=fiducials, order=order, rectification=rect, object_side=side,
        contacts_mm=contacts, warnings=warnings,
    )


def _order(fiducials: list[Fiducial], spec: ChainSpec, excluded: set[int]) -> ChainOrder:
    """Chain order over the fiducials, trying each polarity on its own.

    Every fiducial on the chain has the same polarity (light on the dark link),
    while look-alikes often have the other one (printed letters are dark on
    light), so the polarities are never mixed. The longer chain wins.
    """
    best = None
    for polarity in (1, -1):
        idx = [i for i, r in enumerate(fiducials) if r.polarity == polarity and i not in excluded]
        if len(idx) < 3:
            continue
        o = order_chain(
            np.array([fiducials[i].center_px for i in idx]),
            axes_px=np.array([fiducials[i].outer_axes_px for i in idx]),
            pitch_per_diameter=spec.pitch_mm / spec.fiducial_mm,
            n_pins=spec.n_pins,
            strength=np.array([fiducials[i].contrast for i in idx]),
        )
        mapped = ChainOrder(
            indices=[idx[i] for i in o.indices], links=o.links, gaps=o.gaps,
            rejected=sorted(set(range(len(fiducials))) - {idx[i] for i in o.indices}),
        )
        if best is None or len(mapped.indices) > len(best.indices):
            best = mapped
    if best is None:
        raise ValueError(f"found {len(fiducials)} fiducials; need the whole chain in view")
    return best


def _misfits(fiducials: list[Fiducial], order: ChainOrder, rect: Rectification, spec: ChainSpec) -> set[int]:
    """Pins that do not belong on the chain, judged after deskewing.

    - A fiducial whose physical size differs from its chain neighbours' by more
      than 12%: a washer or other look-alike that happened to sit about one
      pitch from the end of the chain.
    - An end pin whose link is far from one pitch: a stray look-alike next to the
      end. One bad link would otherwise bend the whole solution.
    - With more pins than the spec's chain has, whichever end looks least
      like the rest of the chain.
    """
    idx = order.indices
    if len(idx) < 6:
        return set()
    major = np.array([fiducials[i].outer_axes_px[0] for i in idx])
    size_mm = major * rect.depth_mm / rect.focal_px
    bad = set()
    for k in range(len(idx)):
        nb = [j for j in range(max(0, k - 2), min(len(idx), k + 3)) if j != k]
        if abs(size_mm[k] / np.median(size_mm[nb]) - 1.0) > 0.12:
            bad.add(idx[k])
    if not bad and len(idx) + len(order.gaps) > spec.n_pins:
        # More pins than the chain has, so a look-alike lies about one
        # pitch past an end, the same size as a pin's (a small washer). A real
        # pin's fiducial sits on its link; a look-alike sits on the table. Drop
        # the end that stands out more by what surrounds it, and by size.
        # Compared with its neighbours only: light changes along the chain.
        surround = np.array([fiducials[i].surround for i in idx])
        spread = max(5.0, 1.4826 * float(np.median(np.abs(np.diff(surround)))))

        def oddness(k: int) -> float:
            nb = [1, 2] if k == 0 else [len(idx) - 2, len(idx) - 3]
            return (abs(surround[k] - np.median(surround[nb])) / spread
                    + abs(size_mm[k] / np.median(size_mm[nb]) - 1.0) / 0.03)

        bad.add(idx[0] if oddness(0) > oddness(len(idx) - 1) else idx[-1])
        return bad
    res = rect.link_residuals_mm
    links = order.links
    if links and len(links) >= 6:
        worst = float(np.max(np.abs(res)))
        for end_link, end_pin in ((0, 0), (len(links) - 1, len(idx) - 1)):
            a, b = links[end_link]
            if end_pin not in (a, b):
                continue            # the end is across a gap; leave it
            if abs(res[end_link]) > 0.08 * spec.pitch_mm and abs(res[end_link]) >= 0.999 * worst:
                bad.add(idx[end_pin])
    return bad


def compare_to_truth(pins_mm: np.ndarray, truth_mm: np.ndarray) -> dict[str, float]:
    """Error of measured pins against known pins.

    max/rms_error_mm: after the best rigid fit (rotation + translation).
    max_error_scaled_mm: after also fitting a uniform scale, which removes
    a test part's own print or paper scale error. scale is measured/true
    size from that fit; far from 1 means either the part is off-size
    (check it with calipers) or the chain spec's pitch is wrong.

    The chain has no labels, so both directions along the chain are tried.
    Reflections are not allowed: a mirrored result is an error.
    """
    if len(pins_mm) != len(truth_mm):
        raise ValueError(f"measured {len(pins_mm)} pins but truth has {len(truth_mm)}")
    best = None
    for t in (truth_mm, truth_mm[::-1]):
        err = rigid_fit_errors(pins_mm, t)
        scaled_err, scale = similarity_fit_errors(pins_mm, t)
        if best is None or err.max() < best[0].max():
            best = (err, scaled_err, scale)
    err, scaled_err, scale = best
    return {
        "max_error_mm": float(err.max()),
        "rms_error_mm": float(np.sqrt(np.mean(err ** 2))),
        "max_error_scaled_mm": float(scaled_err.max()),
        "scale": float(scale),
    }


def rigid_fit_errors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point distance after rotating and translating a onto b (no reflection)."""
    errors, _ = _fit(a, b, with_scale=False)
    return errors


def similarity_fit_errors(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-point distance after rotating, translating and uniformly scaling
    a onto b (no reflection), plus the size of a relative to b."""
    errors, s = _fit(a, b, with_scale=True)
    return errors, 1.0 / s


def _fit(a: np.ndarray, b: np.ndarray, with_scale: bool) -> tuple[np.ndarray, float]:
    a0, b0 = a - a.mean(axis=0), b - b.mean(axis=0)
    u, sv, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, d]) @ vt
    s = (sv[0] + d * sv[1]) / np.sum(a0 ** 2) if with_scale else 1.0
    return np.linalg.norm(s * a0 @ rot - b0, axis=1), s
