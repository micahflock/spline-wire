"""Command-line interface.

    splinewire measure PHOTO           photo of the chain -> curve points + SVG
    splinewire synth                   synthetic chain photo with known shape
    splinewire test-part               printable chain drawing with known shape
    splinewire test-plaque             3D-printable chain plaque (STL) with known shape
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from splinewire.chain import ChainSpec, default_chain_path, load_chain_spec
from splinewire.process import process_photo
from splinewire.synthetic import circle_wrap_pins, s_curve_pins, write_synthetic_photo, write_truth
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

    p = sub.add_parser("test-part", help="write a printable chain drawing with known geometry")
    p.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    p.add_argument("--out", type=Path, default=Path("out/test-part"))
    _add_shape_args(p)

    p = sub.add_parser("test-plaque", help="write a 3D-printable chain plaque with known geometry")
    p.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    p.add_argument("--out", type=Path, default=Path("out/test-plaque"))
    _add_shape_args(p)

    args = parser.parse_args(argv)
    spec = load_chain_spec(args.chain)
    commands = {"measure": _measure, "synth": _synth, "test-part": _test_part, "test-plaque": _test_plaque}
    return commands[args.command](args, spec)


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
    photo = args.out / f"{args.shape}.jpg"
    truth = args.out / f"{args.shape}-truth.json"
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


if __name__ == "__main__":
    raise SystemExit(main())
