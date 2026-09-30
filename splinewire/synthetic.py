"""Synthetic chain poses and photos with exactly known geometry.

Used by the tests and by `splinewire synth` to exercise the pipeline
without hardware.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from splinewire.camera import Camera, focal_px_from_35mm, look_at_plane
from splinewire.chain import ChainSpec, pins_from_turns

TRUTH_SCHEMA = "spline-wire/truth@1"
_EXIF_IFD, _TAG_FOCAL_LENGTH_35MM = 0x8769, 0xA405

TABLE_GRAY = 205
LINK_GRAY = 45
RING_GRAY = 235


def s_curve_pins(spec: ChainSpec, bend_deg: float = 14.0) -> np.ndarray:
    """Chain bent one way for its first half and the other way for the rest."""
    n_turns = spec.n_pins - 2
    half = n_turns // 2
    turns = np.radians(np.r_[np.full(half, bend_deg), np.full(n_turns - half, -bend_deg)])
    return pins_from_turns(spec.pitch_mm, turns)


def circle_wrap_pins(spec: ChainSpec, radius_mm: float, concave: bool = False) -> np.ndarray:
    """Pins of a chain touching a circle of radius_mm centered at the origin.

    Convex (wrapped around a disc): each link's inner edge is tangent to the
    circle at the link's midpoint. Concave (pressed into a circular hollow):
    each pin's rounded end touches the circle.
    """
    p, w = spec.pitch_mm, spec.half_width_mm
    pin_radius = math.hypot(radius_mm + w, p / 2) if not concave else radius_mm - w
    step = 2 * math.asin(p / (2 * pin_radius))
    angles = -step * (spec.n_pins - 1) / 2 + step * np.arange(spec.n_pins)
    return pin_radius * np.c_[np.cos(angles), np.sin(angles)]


def render_photo(
    pins_mm: np.ndarray,
    spec: ChainSpec,
    camera: Camera,
    rng: np.random.Generator | None = None,
    noise_gray: float = 3.0,
    blur_px: float = 0.7,
    supersample: int = 3,
    texture_px_per_mm: float = 20.0,
) -> np.ndarray:
    """Grayscale photo of the chain lying on a table, seen by `camera`.

    Link bodies are dark with light ring fiducials over each pin. The chain
    is drawn flat on the plane at high resolution, warped through the
    camera, then downsampled, blurred and given sensor noise.
    """
    rng = rng or np.random.default_rng(0)
    tau = texture_px_per_mm
    margin = spec.half_width_mm + 5.0
    x0, y0 = pins_mm.min(axis=0) - margin
    x1, y1 = pins_mm.max(axis=0) + margin
    tex_w, tex_h = int(math.ceil((x1 - x0) * tau)) + 1, int(math.ceil((y1 - y0) * tau)) + 1
    tex = np.full((tex_h, tex_w), TABLE_GRAY, np.uint8)

    shift = 4
    scale = 1 << shift

    def to_tex(p: np.ndarray) -> tuple[int, int]:
        return (int(round((p[0] - x0) * tau * scale)), int(round((y1 - p[1]) * tau * scale)))

    link_thickness = max(1, int(round(2 * spec.half_width_mm * tau)))
    for a, b in zip(pins_mm[:-1], pins_mm[1:]):
        cv2.line(tex, to_tex(a), to_tex(b), LINK_GRAY, link_thickness, cv2.LINE_AA, shift)
    for p in pins_mm:
        c = to_tex(p)
        cv2.circle(tex, c, int(round(spec.ring_outer_mm / 2 * tau * scale)), RING_GRAY, -1, cv2.LINE_AA, shift)
        cv2.circle(tex, c, int(round(spec.ring_inner_mm / 2 * tau * scale)), LINK_GRAY, -1, cv2.LINE_AA, shift)

    # texture pixel -> plane mm -> image pixel (supersampled grid)
    T = np.array([[1 / tau, 0, x0], [0, -1 / tau, y1], [0, 0, 1]])
    ss = supersample
    S = np.array([[ss, 0, (ss - 1) / 2], [0, ss, (ss - 1) / 2], [0, 0, 1]])
    w, h = camera.image_size
    hi = cv2.warpPerspective(
        tex, S @ camera.plane_homography @ T, (w * ss, h * ss),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=TABLE_GRAY,
    )
    img = cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
    if blur_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_px)
    img += rng.normal(0.0, noise_gray, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def write_synthetic_photo(
    path: Path,
    pins_mm: np.ndarray,
    spec: ChainSpec,
    image_size: tuple[int, int],
    focal_35mm: float = 26.0,
    distance_mm: float = 200.0,
    tilt_deg: float = 25.0,
    seed: int = 0,
) -> None:
    """Render a tilted phone photo of the chain and save it with EXIF focal length."""
    focal = focal_px_from_35mm(focal_35mm, image_size)
    cam = look_at_plane(focal, image_size, distance_mm, tilt_deg=tilt_deg,
                        tilt_direction_deg=35.0, roll_deg=10.0, target_mm=tuple(pins_mm.mean(axis=0)))
    img = render_photo(pins_mm, spec, cam, rng=np.random.default_rng(seed))
    exif = Image.Exif()
    exif.get_ifd(_EXIF_IFD)[_TAG_FOCAL_LENGTH_35MM] = int(round(focal_35mm))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, quality=92, exif=exif)


def write_truth(path: Path, pins_mm: np.ndarray) -> None:
    doc = {"schema": TRUTH_SCHEMA, "units": "mm",
           "pin_points": [[round(float(x), 6), round(float(y), 6)] for x, y in pins_mm]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
