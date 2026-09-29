"""Command-line interface.

    splinewire measure PHOTO           photo of the chain -> curve points + SVG
    splinewire synth                   synthetic chain photo with known shape
    splinewire test-part               printable chain drawing with known shape
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.chain import ChainSpec, load_chain_spec
from splinewire.output import write_csv, write_json, write_preview, write_svg
from splinewire.pipeline import compare_to_truth, load_photo, measure
from splinewire.synthetic import circle_wrap_pins, render_photo, s_curve_pins
from splinewire.testpart import test_part_svg

DEFAULT_CHAIN = Path("data/chain.yaml")
TRUTH_SCHEMA = "spline-wire/truth@1"
_EXIF_IFD, _TAG_FOCAL_LENGTH_35MM = 0x8769, 0xA405


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

    args = parser.parse_args(argv)
    spec = load_chain_spec(args.chain)
    return {"measure": _measure, "synth": _synth, "test-part": _test_part}[args.command](args, spec)


def _add_shape_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shape", choices=["s-curve", "pipe", "cove"], default="s-curve")
    p.add_argument("--radius", type=float, default=30.0, help="radius for pipe/cove shapes, mm")


def _shape(args, spec: ChainSpec) -> np.ndarray:
    if args.shape == "s-curve":
        return s_curve_pins(spec)
    return circle_wrap_pins(spec, args.radius, concave=args.shape == "cove")


def _measure(args, spec: ChainSpec) -> int:
    image, focal = load_photo(args.photo)
    if args.focal_35mm is not None:
        focal = focal_px_from_35mm(args.focal_35mm, (image.shape[1], image.shape[0]))
    m = measure(image, spec, focal, side=args.side)

    extra = {"photo": args.photo.name}
    if args.truth:
        truth = np.array(json.loads(args.truth.read_text(encoding="utf-8"))["pin_points"])
        extra["truth_comparison"] = compare_to_truth(m.pins_mm, truth)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.photo.stem
    write_json(args.out / f"{stem}.json", m, extra)
    write_csv(args.out / f"{stem}-curve.csv", m.contacts_mm)
    write_svg(args.out / f"{stem}-curve.svg", m)
    write_preview(args.out / f"{stem}-preview.jpg", image, m)

    r = m.rectification
    print(f"{len(m.order.indices)} pins found, tilt {r.tilt_deg:.1f} deg, "
          f"focal {r.focal_px:.0f} px{' (estimated)' if r.focal_estimated else ''}, "
          f"link residual rms {r.residual_rms_mm:.3f} mm")
    if "truth_comparison" in extra:
        t = extra["truth_comparison"]
        print(f"vs truth: max error {t['max_error_mm']:.3f} mm, rms {t['rms_error_mm']:.3f} mm")
    for w in m.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"wrote {args.out}/{stem}.json, -curve.csv, -curve.svg, -preview.jpg")
    return 0


def _synth(args, spec: ChainSpec) -> int:
    w, h = (int(v) for v in args.size.lower().split("x"))
    pins = _shape(args, spec)
    focal = focal_px_from_35mm(args.focal_35mm, (w, h))
    cam = look_at_plane(focal, (w, h), args.distance, tilt_deg=args.tilt,
                        tilt_direction_deg=35.0, roll_deg=10.0, target_mm=tuple(pins.mean(axis=0)))
    img = render_photo(pins, spec, cam, rng=np.random.default_rng(args.seed))

    args.out.mkdir(parents=True, exist_ok=True)
    photo = args.out / f"{args.shape}.jpg"
    exif = Image.Exif()
    exif.get_ifd(_EXIF_IFD)[_TAG_FOCAL_LENGTH_35MM] = int(round(args.focal_35mm))
    Image.fromarray(img).save(photo, quality=92, exif=exif)
    _write_truth(args.out / f"{args.shape}-truth.json", pins)
    print(f"wrote {photo} and {args.shape}-truth.json")
    print(f"try: splinewire measure {photo} --truth {args.out / f'{args.shape}-truth.json'}")
    return 0


def _test_part(args, spec: ChainSpec) -> int:
    pins = _shape(args, spec)
    args.out.mkdir(parents=True, exist_ok=True)
    svg = args.out / f"{args.shape}.svg"
    svg.write_text(test_part_svg(pins, spec), encoding="utf-8")
    _write_truth(args.out / f"{args.shape}-truth.json", pins)
    print(f"wrote {svg} (print at 100% scale) and {args.shape}-truth.json")
    return 0


def _write_truth(path: Path, pins: np.ndarray) -> None:
    doc = {"schema": TRUTH_SCHEMA, "units": "mm",
           "pin_points": [[round(float(x), 6), round(float(y), 6)] for x, y in pins]}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
