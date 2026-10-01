"""Realistic synthetic photos: print defects, tables, light, optics, sensor.

synthetic.py draws an ideal chain through an ideal camera. This module adds
what a phone photo of a 3D-printed chain or test plaque brings, so detection
can be stress-tested, and fiducial designs compared, without hardware:

- Print (FDM, 0.4 mm nozzle): convex black corners are rounded to about half
  the line width, black features narrower than a line vanish, narrow white
  gaps close, every edge is over- or under-extruded, edges wobble, and each
  perimeter loop leaves a seam blob. The black layer stands 0.4 mm proud of
  the white, so its walls hide part of each window at a tilt. The top
  surface's extrusion lines modulate brightness and gloss.
- Tables: paper, graph paper, printed text, wood, black, grey, green cutting
  mat, fabric, terrazzo; clutter (washers, nuts, coins) near the chain.
- Light: ambient plus a lamp with distance falloff and a specular lobe (glare
  where its reflection lands on the chain), a gradient, a soft cast shadow
  (the phone, a hand), vignetting.
- Camera: residual radial lens distortion, depth-dependent defocus from the
  lens aperture, hand shake, lens blur.
- Sensor and ISP: auto-exposure with highlight clipping, signal-dependent
  noise, edge-preserving denoising, sharpening, sRGB tone curve, JPEG.

Rendering is in luma only: the detector works on gray images.

    scene = render_scene(pins_mm, spec, PRESETS["shadow"], seed=1)
    scene.image, scene.pins_px     # uint8 photo, true pin centres in pixels
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from splinewire.camera import Camera, focal_px_from_35mm, look_at_plane
from splinewire.chain import ChainSpec
from splinewire.fiducials import Design, chain_design, draw_windows

TAU_PRINT = 30.0          # print raster, px per mm (0.033 mm)
PLATE_MM = 1.6            # white plate under the black layer (plaque)
RIM_MM = 1.5


@dataclass(frozen=True)
class PrintQuality:
    """How an FDM print departs from the design, in mm."""
    corner_radius_mm: float = 0.2    # ~half the line width: rounds convex black corners
    gap_close_mm: float = 0.08       # white gaps narrower than twice this close up
    edge_offset_mm: float = 0.05     # over-extrusion: black edges move outward (<0: inward)
    wobble_mm: float = 0.02          # random edge waviness (std), ~1 mm correlation length
    seam_mm: float = 0.08            # blob on each circular edge at the loop's seam
    relief_mm: float = 0.4           # black layer height above the white
    line_period_mm: float = 0.42     # top-surface extrusion lines
    line_contrast: float = 0.06      # brightness modulation from those lines


PERFECT_PRINT = PrintQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42, 0.0)
LASER_PRINT = PrintQuality(0.03, 0.0, 0.03, 0.01, 0.0, 0.0, 0.42, 0.0)   # toner on paper
BAD_PRINT = PrintQuality(0.25, 0.12, 0.15, 0.05, 0.18, 0.4, 0.45, 0.12)


@dataclass(frozen=True)
class Environment:
    name: str = "daylight"
    # --- camera ---------------------------------------------------------
    image_size: tuple[int, int] = (4032, 3024)   # 12 MP, the common phone default
    focal_35mm: float = 26.0
    px_per_mm: float = 10.0          # at the chain's centre; sets the distance
    tilt_deg: float = 15.0
    tilt_direction_deg: float = 30.0
    roll_deg: float = 10.0
    offset_frac: tuple[float, float] = (0.0, 0.0)   # chain centre off the image centre
    distortion_k1: float = 0.0       # residual radial distortion after in-camera correction
    distortion_k2: float = 0.0
    aperture_mm: float = 3.0         # physical aperture diameter (6 mm lens at f/2)
    focus_error: float = 0.0         # focus distance / true distance - 1
    lens_blur_px: float = 0.6        # lens + demosaic softness (Gaussian sigma)
    motion_blur_px: float = 0.0      # hand shake: blur streak length
    motion_angle_deg: float = 20.0
    # --- light ------------------------------------------------------------
    ambient: float = 0.5             # share of diffuse ambient light at the chain
    lamp_mm: tuple[float, float, float] = (250.0, 150.0, 700.0)   # relative to the chain centre
    lamp_on_reflection: bool = False # move the lamp so its reflection lands on the chain
    gradient: float = 0.1            # relative illumination change per 100 mm
    gradient_deg: float = 0.0
    shadow: str | None = None        # None, "edge" or "band" (a soft shadow across the chain)
    shadow_depth: float = 0.7        # fraction of the lamp light blocked
    shadow_penumbra_mm: float = 6.0
    vignetting: float = 0.15         # falloff at the image corners
    # --- surfaces ---------------------------------------------------------
    background: str = "paper"
    background_gloss: float = 0.0    # glossy table: sharp lamp reflection
    clutter: int = 0                 # washers/nuts/coins near the chain
    mount: str = "plaque"            # "plaque" (white plate + black layer), "chain", "paper"
    print_quality: PrintQuality = field(default_factory=PrintQuality)
    black_albedo: float = 0.045
    white_albedo: float = 0.70
    filament: str = "basic"          # label for FILAMENTS' gloss settings below
    pla_gloss: float = 0.04          # PLA specular reflectance
    pla_shininess: float = 40.0      # lobe width; higher is glossier
    # --- sensor and processing -------------------------------------------
    exposure_ev: float = 0.0
    noise: float = 1.0               # sensor gain: 1 = daylight base ISO, ~16 = dim room
    denoise: float = 0.3
    sharpen: float = 0.6
    jpeg_quality: int = 90


@dataclass(frozen=True)
class Scene:
    image: np.ndarray               # uint8 gray photo
    pins_px: np.ndarray             # where each pin centre appears in the photo
    pins_mm: np.ndarray
    focal_px: float
    camera: Camera
    environment: Environment
    design: Design


# ---------------------------------------------------------------------------
# Presets: a good baseline, then one hard factor at a time, then combinations.

BASE = Environment()
PRESETS: dict[str, Environment] = {e.name: e for e in [
    replace(BASE, name="ideal", print_quality=PERFECT_PRINT, gradient=0.0, vignetting=0.0,
            ambient=1.0, noise=0.5, denoise=0.0, sharpen=0.0, jpeg_quality=97, aperture_mm=0.0),
    BASE,
    replace(BASE, name="far", px_per_mm=5.0),
    replace(BASE, name="very-far", px_per_mm=3.2, image_size=(2048, 1536)),
    replace(BASE, name="close", px_per_mm=18.0),
    replace(BASE, name="steep", tilt_deg=45.0, px_per_mm=14.0, aperture_mm=3.8),
    replace(BASE, name="misfocus", focus_error=0.15, aperture_mm=3.8, px_per_mm=12.0),
    replace(BASE, name="shake", motion_blur_px=7.0),
    replace(BASE, name="dim", noise=16.0, exposure_ev=-0.7, denoise=0.8, ambient=0.8,
            motion_blur_px=2.5),
    replace(BASE, name="shadow", shadow="edge", shadow_depth=0.8, shadow_penumbra_mm=3.0,
            ambient=0.25),
    replace(BASE, name="shadow-band", shadow="band", shadow_depth=0.75, shadow_penumbra_mm=4.0,
            ambient=0.3),
    replace(BASE, name="glare", lamp_on_reflection=True, lamp_mm=(0.0, 0.0, 600.0),
            pla_gloss=0.06, pla_shininess=60.0, ambient=0.35, tilt_deg=5.0),
    replace(BASE, name="glare-matte", lamp_on_reflection=True, lamp_mm=(0.0, 0.0, 600.0),
            filament="matte", pla_gloss=0.025, pla_shininess=12.0, ambient=0.35, tilt_deg=5.0),
    replace(BASE, name="glossy-print", lamp_on_reflection=True, lamp_mm=(0.0, 0.0, 900.0),
            filament="glossy", pla_gloss=0.08, pla_shininess=250.0, ambient=0.4,
            print_quality=replace(PrintQuality(), line_contrast=0.12)),
    replace(BASE, name="black-table", background="black", background_gloss=0.04),
    replace(BASE, name="grey-table", background="grey"),
    replace(BASE, name="wood", background="wood"),
    replace(BASE, name="cutting-mat", background="cutting-mat"),
    replace(BASE, name="graph-paper", background="graph"),
    replace(BASE, name="fabric", background="fabric", px_per_mm=6.0),
    replace(BASE, name="terrazzo", background="terrazzo"),
    replace(BASE, name="text", background="text"),
    replace(BASE, name="clutter", background="grey", clutter=10),
    replace(BASE, name="lens", distortion_k1=0.03, offset_frac=(0.45, 0.35), px_per_mm=7.0),
    replace(BASE, name="jpeg", jpeg_quality=45, image_size=(1600, 1200), px_per_mm=4.0,
            sharpen=1.2),
    replace(BASE, name="bad-print", print_quality=BAD_PRINT),
    replace(BASE, name="paper-print", mount="paper", print_quality=LASER_PRINT),
    replace(BASE, name="bare-chain", mount="chain", background="wood"),
    replace(BASE, name="worst", background="wood", clutter=6, shadow="edge", shadow_penumbra_mm=4.0,
            ambient=0.35, tilt_deg=35.0, px_per_mm=6.0, noise=8.0, denoise=0.6,
            motion_blur_px=3.0, print_quality=BAD_PRINT, jpeg_quality=70, focus_error=0.08),
]}


# Specular reflectance and lobe sharpness ranges for kinds of PLA. Matte
# ("PLA Matte", PolyTerra) scatters the lamp over a wide angle; standard PLA
# prints a satin top surface; silk and glossy blends are near mirrors.
FILAMENTS = {"matte": ((0.015, 0.03), (6.0, 20.0)),
             "basic": ((0.03, 0.05), (25.0, 70.0)),
             "glossy": ((0.05, 0.08), (80.0, 200.0))}


def random_environment(rng: np.random.Generator, name: str = "random") -> Environment:
    """An environment with every factor drawn from a plausible range at once."""
    u = rng.uniform
    filament = str(rng.choice(list(FILAMENTS), p=[0.35, 0.45, 0.2]))
    (g0, g1), (n0, n1) = FILAMENTS[filament]
    pq = PrintQuality(
        corner_radius_mm=u(0.15, 0.25), gap_close_mm=u(0.0, 0.12),
        edge_offset_mm=u(-0.08, 0.15), wobble_mm=u(0.0, 0.04), seam_mm=u(0.0, 0.15),
        relief_mm=float(rng.choice([0.2, 0.4, 0.6])), line_contrast=u(0.0, 0.1),
    )
    small = rng.random() < 0.2
    return Environment(
        name=name,
        image_size=(2048, 1536) if small else (4032, 3024),
        focal_35mm=float(rng.choice([24.0, 26.0, 28.0])),
        px_per_mm=u(3.5, 7.0) if small else u(5.0, 17.0),
        tilt_deg=u(0.0, 40.0), tilt_direction_deg=u(0, 360), roll_deg=u(-180, 180),
        offset_frac=(u(-0.4, 0.4), u(-0.3, 0.3)),
        distortion_k1=u(-0.015, 0.015),
        aperture_mm=u(2.2, 3.9), focus_error=u(-0.06, 0.06),
        lens_blur_px=u(0.4, 0.9), motion_blur_px=u(0.0, 1.0) ** 2 * 5, motion_angle_deg=u(0, 180),
        ambient=u(0.2, 0.9), lamp_mm=(u(-500, 500), u(-500, 500), u(400, 1500)),
        lamp_on_reflection=rng.random() < 0.2, gradient=u(0.0, 0.3), gradient_deg=u(0, 360),
        shadow=rng.choice([None, None, "edge", "band"]), shadow_depth=u(0.3, 0.85),
        shadow_penumbra_mm=u(2.0, 20.0), vignetting=u(0.0, 0.3),
        background=str(rng.choice(BACKGROUNDS)), background_gloss=u(0.0, 0.05),
        clutter=int(rng.choice([0, 0, 0, 4, 8])),
        mount=str(rng.choice(["plaque", "plaque", "chain"])), print_quality=pq,
        black_albedo=u(0.03, 0.07), white_albedo=u(0.55, 0.8),
        filament=filament, pla_gloss=u(g0, g1), pla_shininess=u(n0, n1),
        exposure_ev=u(-0.7, 0.4), noise=float(np.exp(u(0.0, np.log(16.0)))),
        denoise=u(0.0, 0.8), sharpen=u(0.0, 1.2), jpeg_quality=int(u(70, 95)),
    )


BACKGROUNDS = ("paper", "graph", "text", "wood", "black", "grey", "cutting-mat", "fabric", "terrazzo")


# ---------------------------------------------------------------------------

def render_scene(
    pins_mm: np.ndarray,
    spec: ChainSpec,
    env: Environment = BASE,
    design: Design | None = None,
    seed: int = 0,
) -> Scene:
    """Photograph a printed chain (or plaque) lying on a table under env."""
    rng = np.random.default_rng(seed)
    design = design or chain_design(spec)
    pins_mm = np.asarray(pins_mm, dtype=float)
    size = env.image_size
    w, h = size
    focal = focal_px_from_35mm(env.focal_35mm, size)
    cam = _camera(pins_mm, env, focal)
    center3 = -cam.R.T @ cam.t                                  # camera centre, world mm

    # Where every pixel's ray meets the chain plane (z=0) and the table below it.
    table_z = -PLATE_MM if env.mount == "plaque" else (-2.5 if env.mount == "chain" else -0.1)
    rays = _pixel_rays(cam, env)
    X0, Y0, depth = _hit_plane(rays, center3, 0.0)
    Xb, Yb, _ = _hit_plane(rays, center3, table_z)
    del rays
    footprint = depth / focal                                    # mm per pixel, roughly

    # Printed chain: albedo, gloss and coverage in plane coordinates, then into the image.
    fg = _print_layer(pins_mm, spec, design, env, center3, rng)
    near = _near_layer(pins_mm, spec, env, rng)
    blur_fg = 0.45 * fg.tau / env.px_per_mm
    fg_alpha = _sample(fg.alpha, fg, X0, Y0, blur_fg)
    fg_albedo = _sample(fg.albedo * fg.alpha, fg, X0, Y0, blur_fg)
    fg_gloss = _sample(fg.gloss * fg.alpha, fg, X0, Y0, blur_fg)

    bg_albedo, bg_gloss = _background(env.background, Xb, Yb, footprint, rng, env.background_gloss)
    if near is not None:
        blur_near = 0.45 * near.tau / env.px_per_mm
        a = _sample(near.alpha, near, Xb, Yb, blur_near)
        bg_albedo = (1 - a) * bg_albedo + _sample(near.albedo * near.alpha, near, Xb, Yb, blur_near)
        metal = _sample(near.gloss * near.alpha, near, Xb, Yb, blur_near)
        bg_gloss = None if bg_gloss is None else (1 - a) * bg_gloss
    else:
        metal = None
    # Premultiplied composite: the print over the table.
    albedo = fg_albedo + (1 - fg_alpha) * bg_albedo
    table_gloss = None if bg_gloss is None else (1 - fg_alpha) * bg_gloss
    metal_gloss = None if metal is None else (1 - fg_alpha) * metal
    del fg_albedo, bg_albedo, Xb, Yb

    radiance = _light(env, pins_mm, center3, X0, Y0, albedo, fg_gloss, table_gloss, metal_gloss)
    del albedo, fg_gloss, table_gloss, metal_gloss, X0, Y0

    radiance = _optics(radiance, depth, env, focal, cam, pins_mm, rng)
    del depth, footprint
    image = _sensor(radiance, env, rng)
    # The black layer's walls hide the same share of every window edge, so a
    # relief of height h makes the pattern look as if it lay at h/2: still one
    # plane, so deskewing is unaffected. Pixel truth uses that plane.
    relief = env.print_quality.relief_mm if env.mount != "paper" else 0.0
    return Scene(image=image, pins_px=project_pins(cam, pins_mm, env, relief / 2), pins_mm=pins_mm,
                 focal_px=focal, camera=cam, environment=env, design=design)


def project_pins(cam: Camera, pins_mm: np.ndarray, env: Environment, z_mm: float = 0.0) -> np.ndarray:
    """Where the pins (at height z_mm) appear in the photo, including lens distortion."""
    X = np.c_[pins_mm, np.full(len(pins_mm), z_mm)] @ cam.R.T + cam.t
    ideal = (X @ cam.K.T)[:, :2] / X[:, 2:]
    if env.distortion_k1 == 0 and env.distortion_k2 == 0:
        return ideal
    K = cam.K
    xu = (ideal - K[:2, 2]) / K[0, 0]
    xd = xu.copy()
    for _ in range(30):     # invert x_u = x_d * g(|x_d|^2)
        r2 = np.sum(xd ** 2, axis=1, keepdims=True)
        xd = xu / (1 + env.distortion_k1 * r2 + env.distortion_k2 * r2 ** 2)
    return xd * K[0, 0] + K[:2, 2]


# ---------------------------------------------------------------------------
# Geometry

def _camera(pins_mm: np.ndarray, env: Environment, focal: float) -> Camera:
    w, h = env.image_size
    center = pins_mm.mean(axis=0)
    distance = focal / env.px_per_mm
    ox, oy = env.offset_frac
    for shrink in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):   # pull the chain back into frame if needed
        shift = shrink * np.array([ox * w / 2, -oy * h / 2]) / env.px_per_mm
        r = math.radians(env.roll_deg)
        shift = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]]) @ shift
        cam = look_at_plane(focal, env.image_size, distance, tilt_deg=env.tilt_deg,
                            tilt_direction_deg=env.tilt_direction_deg, roll_deg=env.roll_deg,
                            target_mm=tuple(center - shift))
        px = project_pins(cam, pins_mm, env)
        margin = 8.0 * env.px_per_mm
        if (px.min(axis=0) > margin).all() and (px.max(axis=0) < np.array([w, h]) - margin).all():
            return cam
    return cam


def _pixel_rays(cam: Camera, env: Environment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World directions of every pixel's ray, scaled so the camera-z component is 1."""
    w, h = env.image_size
    K = cam.K
    xd = ((np.arange(w) - K[0, 2]) / K[0, 0]).astype(np.float32)[None, :]
    yd = ((np.arange(h) - K[1, 2]) / K[1, 1]).astype(np.float32)[:, None]
    if env.distortion_k1 or env.distortion_k2:
        r2 = xd ** 2 + yd ** 2
        g = (1 + env.distortion_k1 * r2 + env.distortion_k2 * r2 ** 2).astype(np.float32)
        xu, yu = xd * g, yd * g
    else:
        xu, yu = np.broadcast_to(xd, (h, w)), np.broadcast_to(yd, (h, w))
    Rt = cam.R.T.astype(np.float32)
    return tuple(Rt[i, 0] * xu + Rt[i, 1] * yu + Rt[i, 2] for i in range(3))


def _hit_plane(rays, center3, z: float):
    dx, dy, dz = rays
    cx, cy, cz = (np.float32(v) for v in center3)
    t = (np.float32(z) - cz) / dz                       # = depth along the optical axis
    return cx + t * dx, cy + t * dy, t


@dataclass
class _Layer:
    x0: float
    y1: float
    tau: float
    albedo: np.ndarray
    alpha: np.ndarray
    gloss: np.ndarray

    def to_px(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        return np.c_[(pts[:, 0] - self.x0) * self.tau, (self.y1 - pts[:, 1]) * self.tau]


def _sample(raster: np.ndarray, layer: _Layer, X: np.ndarray, Y: np.ndarray, blur: float) -> np.ndarray:
    """Look up a plane raster at plane coordinates (X, Y), pre-blurred to about
    one image pixel so it does not alias."""
    src = raster.astype(np.float32)
    if blur > 0.3:
        src = cv2.GaussianBlur(src, (0, 0), blur)
    mx = ((X - np.float32(layer.x0)) * np.float32(layer.tau)).astype(np.float32)
    my = ((np.float32(layer.y1) - Y) * np.float32(layer.tau)).astype(np.float32)
    return cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


# ---------------------------------------------------------------------------
# The print

def _print_layer(pins_mm, spec: ChainSpec, design: Design, env: Environment, center3, rng) -> _Layer:
    pq = env.print_quality
    tau = float(np.clip(4.0 * env.px_per_mm, 15.0, TAU_PRINT))   # finer than the photo can show
    rim = RIM_MM if env.mount == "plaque" else (10.0 if env.mount == "paper" else 0.0)
    margin = spec.half_width_mm + rim + 2.0
    x0, y0 = pins_mm.min(axis=0) - margin
    x1, y1 = pins_mm.max(axis=0) + margin
    shape = (int(math.ceil((y1 - y0) * tau)) + 1, int(math.ceil((x1 - x0) * tau)) + 1)
    layer = _Layer(x0, y1, tau, None, None, None)
    cover, body = _print_black(pins_mm, spec, design, pq, layer, shape, rng)

    if pq.relief_mm > 0 and env.mount != "paper":
        cover = _relief(cover, layer, center3, pq.relief_mm)

    # Plate (white under the black) outline.
    if env.mount == "plaque":
        plate = _sdf(body > 0, tau) + rim
    elif env.mount == "chain":
        plate = _sdf(body > 0, tau)
    else:
        plate = np.full(shape, 1.0)
        plate[:1, :] = plate[-1:, :] = plate[:, :1] = plate[:, -1:] = -1.0  # sheet edge
    alpha = np.clip(0.5 + plate * tau, 0.0, 1.0).astype(np.float32)

    white = env.white_albedo if env.mount != "paper" else 0.8
    black_albedo = env.black_albedo
    albedo = white + (black_albedo - white) * cover
    gloss = np.full(shape, env.pla_gloss * 0.6, np.float32) + env.pla_gloss * 0.4 * cover
    if pq.line_contrast > 0 and env.mount != "paper":
        jj, ii = np.meshgrid(np.arange(shape[1], dtype=np.float32), np.arange(shape[0], dtype=np.float32))
        u_top = (jj + ii) / (tau * math.sqrt(2))     # top black layer lines at +45 deg
        u_floor = (jj - ii) / (tau * math.sqrt(2))   # white plate's top layer at -45 deg
        wave = np.where(cover > 0.5, np.cos(2 * np.pi * u_top / pq.line_period_mm),
                        np.cos(2 * np.pi * u_floor / pq.line_period_mm)).astype(np.float32)
        albedo = albedo * (1 + pq.line_contrast * wave)
        gloss = gloss * (1 + 0.8 * wave)
    layer.albedo, layer.alpha, layer.gloss = albedo.astype(np.float32), alpha, gloss.astype(np.float32)
    return layer


def _print_black(pins_mm, spec: ChainSpec, design: Design, pq: PrintQuality, layer: _Layer, shape, rng):
    """Black coverage (0..1) of the printed top layer seen straight down,
    and the link bodies' mask."""
    tau = layer.tau
    ONE = 16

    def fixed(pts):
        return np.round(layer.to_px(pts) * ONE).astype(np.int32)

    body = np.zeros(shape, np.uint8)
    thickness = int(round(2 * spec.half_width_mm * tau))
    for a, b in zip(fixed(pins_mm[:-1]), fixed(pins_mm[1:])):
        cv2.line(body, tuple(a), tuple(b), 255, thickness, cv2.LINE_8, 4)
    windows = np.zeros(shape, np.uint8)
    draw_windows(windows, design, pins_mm, layer.to_px, tau)
    black = (body > 0) & (windows == 0)

    d = _sdf(black, tau)
    if pq.corner_radius_mm > 0:                     # opening: nozzle-rounded convex corners
        d = _sdf(d >= pq.corner_radius_mm, tau) + pq.corner_radius_mm
    if pq.gap_close_mm > 0:                         # closing: narrow white gaps fill in
        d = _sdf(d > -pq.gap_close_mm, tau) - pq.gap_close_mm
    d = d + pq.edge_offset_mm
    if pq.wobble_mm > 0:
        d += pq.wobble_mm * _smooth_noise(shape, 0.7 * tau, rng)
    if pq.seam_mm > 0:
        _add_seams(d, design, pins_mm, layer, pq.seam_mm, rng)
    return np.clip(0.5 + d * tau, 0.0, 1.0).astype(np.float32), body


def printed_fiducial(
    spec: ChainSpec, design: Design, quality: PrintQuality = PrintQuality(), seed: int = 0,
    half_size_mm: float = 5.0,
) -> tuple[np.ndarray, float]:
    """How one pin's fiducial comes off the printer, seen straight down.

    Returns black coverage (0..1) in a square raster centred on the pin
    (row 0 at +y), and the raster's px per mm.
    """
    pins = np.array([[-spec.pitch_mm, 0.0], [0.0, 0.0], [spec.pitch_mm, 0.0]])
    m = half_size_mm
    layer = _Layer(-m - spec.pitch_mm, m, TAU_PRINT, None, None, None)
    n = int(round(2 * m * TAU_PRINT)) + 1
    shape = (n, int(round((2 * m + 2 * spec.pitch_mm) * TAU_PRINT)) + 1)
    cover, _ = _print_black(pins, spec, design, quality, layer, shape, np.random.default_rng(seed))
    j0 = int(round(spec.pitch_mm * TAU_PRINT))
    return cover[:, j0:j0 + n], TAU_PRINT


def _sdf(mask: np.ndarray, tau: float) -> np.ndarray:
    """Signed distance to the mask's edge in mm, positive inside."""
    m = mask.astype(np.uint8)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (np.where(m > 0, inside - 0.5, 0.5 - outside) / tau).astype(np.float32)


def _smooth_noise(shape, corr_px: float, rng) -> np.ndarray:
    """Gaussian noise with unit std and correlation length ~corr_px."""
    step = max(1.0, corr_px)
    small = rng.normal(size=(int(shape[0] / step) + 3, int(shape[1] / step) + 3)).astype(np.float32)
    big = cv2.resize(small, (int(small.shape[1] * step), int(small.shape[0] * step)),
                     interpolation=cv2.INTER_CUBIC)[:shape[0], :shape[1]]
    return big / max(1e-6, float(big.std()))


def _add_seams(d, design: Design, pins_mm, layer: _Layer, seam_mm: float, rng) -> None:
    """A blob of extra black on every circular edge, where the loop starts and ends."""
    s = 0.15 * layer.tau
    half = int(4 * s) + 1
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    for p in pins_mm:
        for r in design.circle_edges_mm:
            a = rng.uniform(0, 2 * math.pi)
            c = layer.to_px([p + r * np.array([math.cos(a), math.sin(a)])])[0]
            ci, cj = int(round(c[1])), int(round(c[0]))
            fy, fx = c[1] - ci, c[0] - cj
            blob = seam_mm * rng.uniform(0.5, 1.5) * np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * s * s))
            i0, j0 = ci - half, cj - half
            if i0 < 0 or j0 < 0 or i0 + blob.shape[0] > d.shape[0] or j0 + blob.shape[1] > d.shape[1]:
                continue
            d[i0:i0 + blob.shape[0], j0:j0 + blob.shape[1]] += blob


def _relief(cover: np.ndarray, layer: _Layer, center3, height_mm: float) -> np.ndarray:
    """What the camera sees of the black layer, standing height_mm above the white.

    Rendered in floor coordinates: a floor point is hidden if the ray from it
    to the camera passes through black anywhere between the floor and the top
    of the layer (a wall, or the layer itself).
    """
    h, w = cover.shape
    jj, ii = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    X = layer.x0 + jj / layer.tau
    Y = layer.y1 - ii / layer.tau
    k = height_mm / float(center3[2])
    vx = (k * (center3[0] - X) * layer.tau).astype(np.float32)     # raster px, at the top
    vy = (-k * (center3[1] - Y) * layer.tau).astype(np.float32)
    steps = int(math.ceil(float(np.hypot(vx, vy).max()))) + 1
    out = cover.copy()
    for s in np.linspace(0, 1, steps + 1, dtype=np.float32)[1:]:
        out = np.maximum(out, cv2.remap(cover, jj + s * vx, ii + s * vy, cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0))
    return out


# ---------------------------------------------------------------------------
# Tables and clutter

def _near_layer(pins_mm, spec: ChainSpec, env: Environment, rng) -> _Layer | None:
    """Clutter and printed text, drawn in plane coordinates around the chain."""
    if env.clutter == 0 and env.background not in ("text", "terrazzo"):
        return None
    tau = float(np.clip(1.5 * env.px_per_mm, 8.0, 20.0))
    margin = 70.0
    x0, y0 = pins_mm.min(axis=0) - margin
    x1, y1 = pins_mm.max(axis=0) + margin
    shape = (int((y1 - y0) * tau) + 1, int((x1 - x0) * tau) + 1)
    layer = _Layer(x0, y1, tau, np.zeros(shape, np.float32), np.zeros(shape, np.float32),
                   np.zeros(shape, np.float32))
    ONE = 16

    def fixed(p):
        return tuple(int(v) for v in np.round(layer.to_px([p])[0] * ONE))

    def paint(mask_fn, albedo, gloss):
        m = np.zeros(shape, np.uint8)
        mask_fn(m)
        a = m.astype(np.float32) / 255.0
        layer.albedo[:] = layer.albedo * (1 - a) + albedo * a
        layer.gloss[:] = layer.gloss * (1 - a) + gloss * a
        layer.alpha[:] = np.maximum(layer.alpha, a)

    def cut_hole(p, r):
        hole = np.zeros(shape, np.uint8)
        cv2.circle(hole, fixed(p), int(r * tau * ONE), 255, -1, cv2.LINE_AA, 4)
        layer.alpha[:] = layer.alpha * (1 - hole.astype(np.float32) / 255.0)

    keep_out = spec.half_width_mm + 4.0

    def free_spot(radius):
        for _ in range(200):
            p = rng.uniform([x0 + radius, y0 + radius], [x1 - radius, y1 - radius])
            if np.min(np.linalg.norm(pins_mm - p, axis=1)) > keep_out + radius:
                return p
        return None

    if env.background == "text":
        # A printed page under the chain: rows of text, some large.
        chars = "oOo0aeQD@8bdpgqOoOe" + "     abcdefghijklmnrstuvwxyz"
        ink = np.zeros(shape, np.uint8)
        y = y1 - 4.0
        while y > y0:
            size_mm = float(rng.choice([2.5, 3.0, 3.5, 5.0, 8.0]))
            scale = size_mm * tau / 22.0      # HERSHEY_SIMPLEX cap height ~22 px at scale 1
            org = layer.to_px([[x0 + rng.uniform(0, 10), y]])[0]
            cv2.putText(ink, "".join(rng.choice(list(chars), 60)), (int(org[0]), int(org[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, 255, max(1, int(round(scale * 2))), cv2.LINE_AA)
            y -= size_mm * 1.8
        paint(lambda m: np.copyto(m, ink), 0.04, 0.0)
    if env.background == "terrazzo":
        chips = {alb: np.zeros(shape, np.uint8) for alb in (0.04, 0.1, 0.3, 0.9)}
        for _ in range(int((x1 - x0) * (y1 - y0) / 40)):
            p = rng.uniform([x0, y0], [x1, y1])
            axes = tuple(int(v * tau * ONE) for v in rng.uniform(0.5, 4.0, 2))
            m = chips[float(rng.choice(list(chips)))]
            cv2.ellipse(m, fixed(p), axes, rng.uniform(0, 180), 0, 360, 255, -1, cv2.LINE_AA, 4)
        for alb, m in chips.items():
            paint(lambda dst, m=m: np.copyto(dst, m), alb, 0.0)
    for k in range(env.clutter):
        kind = ["washer", "washer", "nut", "coin", "washer-flat"][k % 5]
        if kind.startswith("washer"):
            od, idia = [(7.0, 3.2), (9.0, 4.3), (10.0, 5.3), (6.0, 2.5)][int(rng.integers(4))]
            if k == 0 and len(pins_mm) > 1:
                # One washer in line with a chain end, one or two pitches out: can it
                # be mistaken for another pin?
                end, prev = (pins_mm[-1], pins_mm[-2]) if rng.random() < 0.5 else (pins_mm[0], pins_mm[1])
                d = (end - prev) / np.linalg.norm(end - prev)
                p = end + d * spec.pitch_mm * rng.uniform(1.0, 2.2)
            else:
                p = free_spot(od / 2)
            if p is None:
                continue
            alb, gl = (0.45, 0.5) if kind == "washer" else (0.8, 0.05)   # steel, or a white nylon washer
            paint(lambda m, p=p, r=od / 2: cv2.circle(m, fixed(p), int(r * tau * ONE), 255, -1, cv2.LINE_AA, 4),
                  alb, gl)
            cut_hole(p, idia / 2)
        elif kind == "nut":
            p = free_spot(5.0)
            if p is None:
                continue
            ang = rng.uniform(0, math.pi)
            hexagon = p + 4.0 * np.c_[np.cos(ang + np.arange(6) * math.pi / 3), np.sin(ang + np.arange(6) * math.pi / 3)]
            pts = np.round(layer.to_px(hexagon) * ONE).astype(np.int32)
            paint(lambda m, pts=pts: cv2.fillPoly(m, [pts], 255, cv2.LINE_AA, 4), 0.35, 0.4)
            cut_hole(p, 2.5)
        else:
            r = float(rng.uniform(8.0, 12.5))
            p = free_spot(r)
            if p is None:
                continue
            paint(lambda m, p=p, r=r: cv2.circle(m, fixed(p), int(r * tau * ONE), 255, -1, cv2.LINE_AA, 4),
                  0.35, 0.35)
    return layer


def _noise_tex(rng, size=256) -> np.ndarray:
    t = rng.normal(size=(size, size)).astype(np.float32)
    t = cv2.GaussianBlur(np.pad(t, 8, mode="wrap"), (0, 0), 1.0)[8:-8, 8:-8]   # tileable
    return t / t.std()


def _value_noise(X, Y, cell_mm: float, rng) -> np.ndarray:
    """Smooth unit-variance noise with features ~cell_mm, tiled every 256 cells."""
    tex = _noise_tex(rng)
    k = np.float32(1.0 / cell_mm)
    mx, my = X * k, Y * k
    mx -= np.floor(mx * np.float32(1 / 256)) * 256
    my -= np.floor(my * np.float32(1 / 256)) * 256
    return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _lines(u, period, width, footprint) -> np.ndarray:
    """Anti-aliased coverage of lines of the given width every `period` mm along u."""
    dist = np.abs(u - period * np.round(u / period))
    return np.clip((width / 2 - dist) / np.maximum(footprint, 1e-3) + 0.5, 0.0, 1.0)


def _attenuate(amplitude: float, period_mm: float, footprint) -> np.ndarray:
    """Contrast left of a texture of this period after the pixel's footprint averages it."""
    return amplitude * np.exp(-2.0 * (np.pi * footprint / period_mm) ** 2)


def _background(kind: str, X, Y, footprint, rng, gloss: float):
    """Table albedo and gloss at plane coordinates."""
    if kind in ("paper", "text", "graph"):
        a = 0.80 + 0.012 * _value_noise(X, Y, 0.25, rng) + 0.02 * _value_noise(X, Y, 25.0, rng)
        if kind == "graph":
            minor = np.maximum(_lines(X, 1.0, 0.08, footprint), _lines(Y, 1.0, 0.08, footprint))
            major = np.maximum(_lines(X, 5.0, 0.18, footprint), _lines(Y, 5.0, 0.18, footprint))
            a = a * (1 - 0.15 * minor) * (1 - 0.35 * major)
    elif kind == "wood":
        ang = rng.uniform(0, np.pi)
        warp = 2.5 * _value_noise(X, Y, 18.0, rng)
        u = X * math.cos(ang) + Y * math.sin(ang) + warp
        period = float(rng.uniform(1.5, 3.0))
        grain = (0.5 + 0.5 * np.sin(2 * np.pi * u / period)) ** 3
        a = (0.25 + _attenuate(0.12, period, footprint) * grain
             + 0.03 * _value_noise(X, Y, 0.4, rng) + 0.05 * _value_noise(X, Y, 40.0, rng))
    elif kind == "black":
        a = 0.035 + 0.004 * _value_noise(X, Y, 0.3, rng)
    elif kind == "grey":
        a = 0.18 + 0.006 * _value_noise(X, Y, 0.3, rng) + 0.01 * _value_noise(X, Y, 30.0, rng)
    elif kind == "cutting-mat":
        a = 0.10 + 0.006 * _value_noise(X, Y, 0.3, rng)
        grid = np.maximum(_lines(X, 10.0, 0.25, footprint), _lines(Y, 10.0, 0.25, footprint))
        bold = np.maximum(_lines(X, 50.0, 0.6, footprint), _lines(Y, 50.0, 0.6, footprint))
        diag = _lines((X + Y) / math.sqrt(2), 50.0, 0.25, footprint)
        a = a + 0.45 * np.maximum(np.maximum(grid, bold), diag)
    elif kind == "fabric":
        ang = rng.uniform(0, np.pi)
        u = X * math.cos(ang) + Y * math.sin(ang)
        v = -X * math.sin(ang) + Y * math.cos(ang)
        p = float(rng.uniform(0.8, 1.4))
        weave = np.sign(np.sin(2 * np.pi * u / p)) * np.sign(np.sin(2 * np.pi * v / p))
        a = (0.22 + _attenuate(0.07, p, footprint) * weave * np.abs(np.sin(2 * np.pi * u / p))
             + 0.03 * _value_noise(X, Y, 6.0, rng))
    elif kind == "terrazzo":
        a = 0.55 + 0.03 * _value_noise(X, Y, 0.5, rng)
    else:
        raise ValueError(f"unknown background {kind!r}")
    return np.clip(a, 0.005, 0.95).astype(np.float32), (gloss if gloss > 0 else None)


# ---------------------------------------------------------------------------
# Light

def _light(env: Environment, pins_mm, center3, X, Y, albedo, pla_gloss, table_gloss, metal_gloss):
    """Radiance: albedo times irradiance, plus specular lobes weighted by gloss.

    Light varies smoothly (shadow penumbras are millimetres wide), so the
    fields are computed on a 4x coarser grid and upsampled.
    """
    f32 = np.float32
    step = 4
    Xs, Ys = X[::step, ::step], Y[::step, ::step]
    c = pins_mm.mean(axis=0)
    lamp = np.array([c[0] + env.lamp_mm[0], c[1] + env.lamp_mm[1], env.lamp_mm[2]])
    if env.lamp_on_reflection:
        # Mirror the camera through the chain plane about the chain centre: the
        # lamp's reflection then lands on the chain.
        v = center3 - np.array([c[0], c[1], 0.0])
        lamp = np.array([c[0], c[1], 0.0]) + env.lamp_mm[2] / v[2] * np.array([-v[0], -v[1], v[2]])
        lamp[:2] += np.array(env.lamp_mm[:2]) * 0.05
    lx, ly, lz = f32(lamp[0]) - Xs, f32(lamp[1]) - Ys, f32(lamp[2])
    r2 = lx * lx + ly * ly + lz * lz
    r = np.sqrt(r2)
    ref = lamp[2] / (np.hypot(lamp[0] - c[0], lamp[1] - c[1]) ** 2 + lamp[2] ** 2) ** 1.5
    direct = (lz / (r2 * r)) / f32(ref)                 # Lambert with 1/r^2, 1.0 at the chain

    shade = np.ones_like(Xs)
    if env.shadow:
        a = math.radians(37.0 + 90.0 * (env.shadow == "band"))
        s = (Xs - f32(c[0])) * f32(math.cos(a)) + (Ys - f32(c[1])) * f32(math.sin(a)) + f32(3.0)
        pen = f32(max(0.1, env.shadow_penumbra_mm))
        if env.shadow == "band":
            inside = np.clip((f32(7.5) - np.abs(s)) / pen + f32(0.5), 0, 1)
        else:
            inside = np.clip(s / pen + f32(0.5), 0, 1)
        inside = inside * inside * (3 - 2 * inside)     # smoothstep
        shade = 1 - f32(env.shadow_depth) * inside
    g = math.radians(env.gradient_deg)
    grad = np.maximum(f32(0.05), 1 + f32(env.gradient / 100.0) * (
        (Xs - f32(c[0])) * f32(math.cos(g)) + (Ys - f32(c[1])) * f32(math.sin(g))))
    lamp_light = f32(1 - env.ambient) * direct * shade * grad
    irradiance = f32(env.ambient) * grad + lamp_light

    # Specular: Blinn-Phong lobes, normalised so a mirror-direction peak is (n+2)/2.
    vx, vy, vz = f32(center3[0]) - Xs, f32(center3[1]) - Ys, f32(center3[2])
    vn = np.sqrt(vx * vx + vy * vy + vz * vz)
    hx, hy, hz = lx / r + vx / vn, ly / r + vy / vn, lz / r + vz / vn
    nh = hz / np.sqrt(hx * hx + hy * hy + hz * hz)

    size = (X.shape[1], X.shape[0])

    def up(field):
        return cv2.resize(field.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)

    radiance = albedo * up(irradiance)
    for gloss, n in ((pla_gloss, env.pla_shininess), (table_gloss, 400.0), (metal_gloss, 60.0)):
        if gloss is not None:
            radiance += gloss * up(lamp_light * f32((n + 2) / 2) * np.power(nh, f32(n)))
    return radiance.astype(np.float32)


# ---------------------------------------------------------------------------
# Camera and sensor

def _disc_kernel(diameter: float) -> np.ndarray:
    if diameter < 0.5:
        return np.ones((1, 1), np.float32)
    ss = 8
    n = int(math.ceil(diameter)) + 2
    n += 1 - n % 2
    c = (n * ss - 1) / 2
    yy, xx = np.mgrid[:n * ss, :n * ss]
    fine = ((xx - c) ** 2 + (yy - c) ** 2 <= (diameter * ss / 2) ** 2).astype(np.float32)
    k = cv2.resize(fine, (n, n), interpolation=cv2.INTER_AREA)
    return k / k.sum()


def _line_kernel(length: float, angle_deg: float) -> np.ndarray:
    n = int(math.ceil(length)) + 3
    n += 1 - n % 2
    ss = 8
    k = np.zeros((n * ss, n * ss), np.float32)
    c = n * ss / 2
    dx, dy = math.cos(math.radians(angle_deg)) * length * ss / 2, math.sin(math.radians(angle_deg)) * length * ss / 2
    cv2.line(k, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1.0, ss, cv2.LINE_AA)
    k = cv2.resize(k, (n, n), interpolation=cv2.INTER_AREA)
    return k / k.sum()


def _optics(radiance, depth, env: Environment, focal, cam: Camera, pins_mm, rng) -> np.ndarray:
    img = radiance
    if env.lens_blur_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), env.lens_blur_px)
    if env.motion_blur_px > 0.5:
        img = cv2.filter2D(img, -1, _line_kernel(env.motion_blur_px, env.motion_angle_deg),
                           borderType=cv2.BORDER_REFLECT)
    if env.aperture_mm > 0:
        z_chain = float(np.mean((cam.R @ np.c_[pins_mm, np.zeros(len(pins_mm))].T + cam.t[:, None])[2]))
        z_focus = z_chain * (1 + env.focus_error)
        blur = np.minimum(18.0, env.aperture_mm * focal * np.abs(depth - z_focus) / (depth * z_focus))
        img = _defocus(img, blur.astype(np.float32), cam, pins_mm, env)
    if env.vignetting > 0:
        h, w = img.shape
        yy, xx = np.ogrid[:h, :w]
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        img = img * (1 - env.vignetting * r2).astype(np.float32)
    return img


def _defocus(img, blur, cam: Camera, pins_mm, env: Environment) -> np.ndarray:
    """Depth-varying disc blur: blend a stack of uniformly blurred copies.

    Full accuracy only around the chain (clutter included); farther out a
    single blur at the median level is used.
    """
    h, w = img.shape
    px = cam.project(np.vstack([pins_mm.min(axis=0) - 80, pins_mm.max(axis=0) + 80,
                                [pins_mm.min(axis=0)[0] - 80, pins_mm.max(axis=0)[1] + 80],
                                [pins_mm.max(axis=0)[0] + 80, pins_mm.min(axis=0)[1] - 80]]))
    pad = 24
    x0, y0 = np.clip(np.floor(px.min(axis=0)).astype(int) - pad, 0, [w, h])
    x1, y1 = np.clip(np.ceil(px.max(axis=0)).astype(int) + pad, 0, [w, h])
    out = cv2.filter2D(img, -1, _disc_kernel(float(np.median(blur))), borderType=cv2.BORDER_REFLECT)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return out
    roi, b = img[y0:y1, x0:x1], blur[y0:y1, x0:x1]
    levels = [0.0, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0, 14.0, 18.0]
    lo, hi = float(b.min()), float(b.max())
    acc = np.zeros_like(roi)
    for k, d in enumerate(levels):
        prev = levels[k - 1] if k else None
        nxt = levels[k + 1] if k + 1 < len(levels) else None
        if (nxt is not None and nxt < lo) or (prev is not None and prev > hi):
            continue
        weight = np.ones_like(b)
        if prev is not None:
            weight = np.where(b < d, np.clip((b - prev) / (d - prev), 0, 1), weight)
        if nxt is not None:
            weight = np.where(b >= d, np.clip((nxt - b) / (nxt - d), 0, 1), weight)
        if prev is None:
            weight = np.where(b < d, 1.0, weight)
        if not weight.any():
            continue
        acc += weight * cv2.filter2D(roi, -1, _disc_kernel(d), borderType=cv2.BORDER_REFLECT)
    out[y0 + pad // 2:y1 - pad // 2, x0 + pad // 2:x1 - pad // 2] = \
        acc[pad // 2:acc.shape[0] - pad // 2, pad // 2:acc.shape[1] - pad // 2]
    return out


def _srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _sensor(radiance, env: Environment, rng) -> np.ndarray:
    # Auto-exposure: centre-weighted average to a mid tone, backed off if more
    # than ~0.5% of the frame would blow out. Highlights then roll off on a
    # tone shoulder (phones' HDR); the sensor itself saturates at 1.6x.
    h, w = radiance.shape
    centre = radiance[h // 4:3 * h // 4:4, w // 4:3 * w // 4:4]
    scale = 0.3 / max(1e-6, float(centre.mean()))
    scale = min(scale, 2.0 / max(1e-6, float(np.percentile(radiance[::4, ::4], 99.5))))
    x = radiance * np.float32(scale * 2 ** env.exposure_ev)
    # Photon + read noise, in units of full scale; gain multiplies both.
    g = env.noise
    sigma = np.sqrt(np.maximum(0.0, x) * g / 4000.0 + (g * 3.0 / 4000.0) ** 2)
    x = np.clip(x + sigma * rng.standard_normal(x.shape, dtype=np.float32), 0.0, 1.6)
    knee = np.float32(0.6)
    x = np.where(x < knee, x, knee + (1 - knee) * (1 - np.exp(-(x - knee) / (1 - knee))))
    y = (_srgb(x) * 255.0).astype(np.float32)
    if env.denoise > 0:
        noise_dn = float(np.median(sigma)) * 255.0 * 3.0 + 1.0
        y = cv2.bilateralFilter(y, 0, sigmaColor=noise_dn * (1 + 2 * env.denoise),
                                sigmaSpace=0.8 + 1.2 * env.denoise)
    if env.sharpen > 0:
        y = y + env.sharpen * (y - cv2.GaussianBlur(y, (0, 0), 1.2))
    img = np.clip(np.round(y), 0, 255).astype(np.uint8)
    if env.jpeg_quality < 100:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(env.jpeg_quality)])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img
