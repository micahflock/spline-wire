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

from splinewire.camera import Camera, apply_homography, focal_px_from_35mm, look_at_plane
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
    each pin's rounded end touches the circle. A chain with round contact
    touches with the sleeve about each pin either way.
    """
    p, w = spec.pitch_mm, spec.half_width_mm
    if concave:
        pin_radius = radius_mm - w
    else:
        pin_radius = radius_mm + w if spec.contact == "round" else math.hypot(radius_mm + w, p / 2)
    step = 2 * math.asin(p / (2 * pin_radius))
    angles = -step * (spec.n_pins - 1) / 2 + step * np.arange(spec.n_pins)
    return pin_radius * np.c_[np.cos(angles), np.sin(angles)]


# The snap-fit chain's relief, mm below the plane of its rims: the table,
# the sleeve tops, and the pin and arm tops where the rim's white ends.
SNAP_TABLE_MM = 12.4
SNAP_SLEEVE_MM = 2.4
SNAP_RIM_MM = 0.4
SNAP_ARM_MM = 5.0          # arm width


def render_snap_photo(
    pins_mm: np.ndarray,
    spec: ChainSpec,
    camera: Camera,
    rng: np.random.Generator | None = None,
    table_gray: int = TABLE_GRAY,
    noise_gray: float = 3.0,
    blur_px: float = 0.7,
    supersample: int = 3,
    texture_px_per_mm: float = 20.0,
) -> np.ndarray:
    """Grayscale photo of the snap-fit chain (contact: round), with its relief.

    Unlike the flat chains, this one stands 12 mm tall with its rims on top
    and its sleeves 2.4 mm below them, so a tilted camera sees the sleeves
    and the table shifted against the rims. The chain is drawn as a stack
    of flat slices, each put where the camera sees it on the rim plane
    (z = 0, where pins_mm are): table, sleeves and arms up to the sleeve
    tops, arms and pin tops, then the white rims with their 0.4 mm walls.
    """
    rng = rng or np.random.default_rng(0)
    tau = texture_px_per_mm
    w, h = camera.image_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    seen = apply_homography(np.linalg.inv(camera.plane_homography), corners)
    reach = spec.half_width_mm + SNAP_TABLE_MM * 1.5 + 5.0
    x0, y0 = np.maximum(pins_mm.min(axis=0) - reach, seen.min(axis=0))
    x1, y1 = np.minimum(pins_mm.max(axis=0) + reach, seen.max(axis=0))
    tex_w, tex_h = int(math.ceil((x1 - x0) * tau)) + 1, int(math.ceil((y1 - y0) * tau)) + 1
    T = np.array([[1 / tau, 0, x0], [0, -1 / tau, y1], [0, 0, 1]])
    shift = 4
    scale = 1 << shift

    def to_tex(p: np.ndarray) -> tuple[int, int]:
        return (int(round((p[0] - x0) * tau * scale)), int(round((y1 - p[1]) * tau * scale)))

    def disc(img, p, radius_mm):
        cv2.circle(img, to_tex(p), int(round(radius_mm * tau * scale)), 255, -1, cv2.LINE_AA, shift)

    def arms(img):
        for a, b in zip(pins_mm[:-1], pins_mm[1:]):
            cv2.line(img, to_tex(a), to_tex(b), 255, int(round(SNAP_ARM_MM * tau)), cv2.LINE_AA, shift)

    blank = np.zeros((tex_h, tex_w), np.uint8)
    lower, upper, rim = blank.copy(), blank.copy(), blank.copy()
    arms(lower)
    arms(upper)
    for k, p in enumerate(pins_mm):
        # Every pin but the first sits in the next link's sleeve.
        disc(lower, p, spec.half_width_mm if k else 3.0)
        disc(upper, p, spec.fiducial_mm / 2)
        disc(rim, p, spec.fiducial_mm / 2)
        cv2.circle(rim, to_tex(p), int(round(spec.ring_inner_mm / 2 * tau * scale)), 0, -1, cv2.LINE_AA, shift)

    def lowered(mask, depth_mm):
        """The slice as the camera sees it on the rim plane from depth_mm below."""
        if depth_mm == 0:
            return mask
        below = camera.K @ np.c_[camera.R[:, 0], camera.R[:, 1], camera.t - depth_mm * camera.R[:, 2]]
        M = np.linalg.inv(T) @ np.linalg.inv(camera.plane_homography) @ below @ T
        return cv2.warpPerspective(mask, M, (tex_w, tex_h), flags=cv2.INTER_LINEAR)

    tex = np.full((tex_h, tex_w), float(table_gray), np.float32)
    slices = [(lower, d, LINK_GRAY) for d in np.arange(SNAP_TABLE_MM, SNAP_SLEEVE_MM - 1e-9, -0.5)]
    slices += [(upper, d, LINK_GRAY) for d in np.arange(SNAP_SLEEVE_MM, SNAP_RIM_MM - 1e-9, -0.5)]
    slices += [(rim, d, RING_GRAY) for d in (SNAP_RIM_MM, SNAP_RIM_MM / 2, 0.0)]
    for mask, depth, gray in slices:
        a = lowered(mask, float(depth)).astype(np.float32) / 255.0
        tex = tex * (1 - a) + gray * a

    ss = supersample
    S = np.array([[ss, 0, (ss - 1) / 2], [0, ss, (ss - 1) / 2], [0, 0, 1]])
    hi = cv2.warpPerspective(
        np.clip(tex, 0, 255).astype(np.uint8), S @ camera.plane_homography @ T, (w * ss, h * ss),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=int(table_gray),
    )
    img = cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
    if blur_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_px)
    img += rng.normal(0.0, noise_gray, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


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

    Link bodies are dark with light dot or ring fiducials over each pin. The chain
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
        cv2.circle(tex, c, int(round(spec.fiducial_mm / 2 * tau * scale)), RING_GRAY, -1, cv2.LINE_AA, shift)
        if spec.fiducial == "ring":
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
    render = render_snap_photo if spec.contact == "round" else render_photo
    img = render(pins_mm, spec, cam, rng=np.random.default_rng(seed))
    save_photo(path, img, focal_35mm)


def save_photo(path: Path, image: np.ndarray, focal_35mm: float, quality: int = 92) -> None:
    """Save as JPEG with the 35 mm-equivalent focal length in EXIF, like a phone."""
    exif = Image.Exif()
    exif.get_ifd(_EXIF_IFD)[_TAG_FOCAL_LENGTH_35MM] = int(round(focal_35mm))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=quality, exif=exif)


def write_truth(path: Path, pins_mm: np.ndarray) -> None:
    doc = {"schema": TRUTH_SCHEMA, "units": "mm",
           "pin_points": [[round(float(x), 6), round(float(y), 6)] for x, y in pins_mm]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
