"""Command-line interface.

    splinewire measure PHOTO           photo of the chain -> curve points + SVG
    splinewire synth [--env PRESET]    synthetic chain photo with known shape
    splinewire test-part               printable chain drawing with known shape
    splinewire test-plaque             3D-printable chain plaque (STL) with known shape
    splinewire print-chain             3D-printable working chain (print-in-place STL)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from splinewire.chain import ChainSpec, default_chain_path, load_chain_spec
from splinewire.process import process_photo
from splinewire.synthetic import (
    circle_wrap_pins,
    s_curve_pins,
    save_photo,
    write_synthetic_photo,
    write_truth,
)
from splinewire.testpart import test_part_svg

DEFAULT_CHAIN = default_chain_path()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="splinewire", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("measure", help="measure the chain in a photo")
    p.add_argument("photo", type=Path)
    p.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--focal-35mm", type=float,
                   help="35 mm-equivalent focal length; overrides EXIF")
    p.add_argument("--side", choices=["inside", "outside"], default="inside",
                   help="object on the inside of the chain's bend (wrapped around it, default) "
                        "or the outside (chain pressed into a hollow)")
    p.add_argument("--truth", type=Path, help="truth JSON from synth/test-part; reports error")

    p = sub.add_parser("synth", help="render a synthetic chain photo with known geometry")
    p.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    p.add_argument("--out", type=Path, default=Path("out/synth"))
    _add_shape_args(p)
    p.add_argument("--tilt", type=float, default=25.0, help="camera tilt from straight-on, degrees")
    p.add_argument("--distance", type=float, default=200.0, help="camera distance, mm")
    p.add_argument("--focal-35mm", type=float, default=26.0)
    p.add_argument("--size", default="4000x3000", help="image size WxH")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", choices=sorted(_presets()), metavar="PRESET",
                   help="realistic photo instead: a printed plaque in one of the simulated "
                        "environments (e.g. shadow, glare, clutter; see splinewire/scene.py). "
                        "Sets camera, light and print itself; --tilt/--distance/--size are ignored")

    p = sub.add_parser("test-part", help="write a printable chain drawing with known geometry")
    p.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    p.add_argument("--out", type=Path, default=Path("out/test-part"))
    _add_shape_args(p)

    p = sub.add_parser("test-plaque", help="write a 3D-printable chain plaque with known geometry")
    p.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    p.add_argument("--out", type=Path, default=Path("out/test-plaque"))
    _add_shape_args(p)

    p = sub.add_parser("print-chain", help="write a print-in-place chain with working joints (STL)")
    p.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    p.add_argument("--out", type=Path, default=Path("out/print-chain"))
    p.add_argument("--pins", type=int, help="number of pins (default: n_pins from the chain file); "
                                             "3 prints a single test joint")
    p.add_argument("--clearance", type=float, default=0.3,
                   help="gap between parts as printed, mm; raise it if joints fuse")
    p.add_argument("--preload", type=float, default=0.15,
                   help="how far each pin bends its spring once set, mm; sets the joint friction")

    args = parser.parse_args(argv)
    spec = load_chain_spec(args.chain)
    commands = {"measure": _measure, "synth": _synth, "test-part": _test_part, "test-plaque": _test_plaque,
                "print-chain": _print_chain}
    return commands[args.command](args, spec)


def _presets() -> dict:
    from splinewire.scene import PRESETS
    return PRESETS


def _add_shape_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shape", choices=["s-curve", "pipe", "cove"], default="s-curve")
    p.add_argument("--radius", type=float, default=30.0, help="radius for pipe/cove shapes, mm")


def _shape(args, spec: ChainSpec) -> np.ndarray:
    if args.shape == "s-curve":
        return s_curve_pins(spec)
    return circle_wrap_pins(spec, args.radius, concave=args.shape == "cove")


def _measure(args, spec: ChainSpec) -> int:
    result = process_photo(args.photo, spec, args.out, focal_35mm=args.focal_35mm,
                           side=args.side, truth_path=args.truth)
    m = result.measurement
    r = m.rectification
    print(f"{len(m.order.indices)} pins found, tilt {r.tilt_deg:.1f} deg, "
          f"focal {r.focal_px:.0f} px{' (estimated)' if r.focal_estimated else ''}, "
          f"link residual rms {r.residual_rms_mm:.3f} mm")
    if result.truth_comparison:
        t = result.truth_comparison
        print(f"vs truth: max error {t['max_error_mm']:.3f} mm, rms {t['rms_error_mm']:.3f} mm; "
              f"after scale fit {t['max_error_scaled_mm']:.3f} mm "
              f"(measured/true scale {t['scale']:.4f})")
    for w in m.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"wrote {args.out}/{args.photo.stem}.json, -curve.csv, -curve.svg, -preview.jpg")
    return 0


def _synth(args, spec: ChainSpec) -> int:
    w, h = (int(v) for v in args.size.lower().split("x"))
    pins = _shape(args, spec)
    truth = args.out / f"{args.shape}-truth.json"
    if args.env:
        from splinewire.scene import render_scene
        env = _presets()[args.env]
        scene = render_scene(pins, spec, env, seed=args.seed)
        photo = args.out / f"{args.shape}-{args.env}.jpg"
        save_photo(photo, scene.image, env.focal_35mm, quality=97)   # the scene already went through JPEG
    else:
        photo = args.out / f"{args.shape}.jpg"
        write_synthetic_photo(photo, pins, spec, (w, h), focal_35mm=args.focal_35mm,
                              distance_mm=args.distance, tilt_deg=args.tilt, seed=args.seed)
    write_truth(truth, pins)
    print(f"wrote {photo} and {truth.name}")
    print(f"try: splinewire measure {photo} --truth {truth}")
    return 0


def _test_part(args, spec: ChainSpec) -> int:
    pins = _shape(args, spec)
    args.out.mkdir(parents=True, exist_ok=True)
    svg = args.out / f"{args.shape}.svg"
    svg.write_text(test_part_svg(pins, spec), encoding="utf-8")
    write_truth(args.out / f"{args.shape}-truth.json", pins)
    print(f"wrote {svg} (print at 100% scale) and {args.shape}-truth.json")
    return 0


def _test_plaque(args, spec: ChainSpec) -> int:
    try:
        from splinewire.plaque import write_plaque
    except ImportError as err:
        print(f"test-plaque needs the dev dependencies (uv sync): {err}", file=sys.stderr)
        return 1
    pins = _shape(args, spec)
    paths = write_plaque(args.out, args.shape, pins, spec)
    write_truth(args.out / f"{args.shape}-truth.json", pins)
    print(paths["instructions"].read_text(encoding="utf-8"))
    print("wrote " + ", ".join(p.name for p in paths.values()) + f", {args.shape}-truth.json in {args.out}")
    return 0


def _print_chain(args, spec: ChainSpec) -> int:
    try:
        from splinewire.printed_chain import JointParams, write_printed_chain
    except ImportError as err:
        print(f"print-chain needs the dev dependencies (uv sync): {err}", file=sys.stderr)
        return 1
    params = JointParams(clearance_mm=args.clearance, preload_mm=args.preload)
    name = "chain" if args.pins is None else f"chain-{args.pins}pins"
    paths = write_printed_chain(args.out, spec, params, n_pins=args.pins, name=name)
    print(paths["instructions"].read_text(encoding="utf-8"))
    print("wrote " + ", ".join(p.name for p in paths.values()) + f" in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
