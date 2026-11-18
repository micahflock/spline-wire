"""Command-line interface.

    splinewire measure PHOTO           photo of the chain -> curve points + SVG
    splinewire synth [--env PRESET]    synthetic chain photo with known shape
    splinewire test-part               printable chain drawing with known shape
    splinewire test-plaque             3D-printable chain plaque (STL) with known shape
    splinewire test-gauges             3D-printable contact gauges (STL) with known shapes
    splinewire fixed-chain             the snap-fit chain posed and fused into one print (STL), with its truth
    splinewire score PHOTO... --target T  test photos scored against a gauge or plaque (docs/real-world-tests.md)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from splinewire.chain import ChainSpec, default_chain_path, load_chain_spec, oring_chain_path
from splinewire.process import process_photo
from splinewire.synthetic import (
    circle_wrap_pins,
    s_curve_pins,
    save_photo,
    write_synthetic_photo,
    write_truth,
)
from splinewire.targets import GAUGES, PLAQUE_POSES, target_names
from splinewire.testpart import test_part_svg

# measure and synth start with the default chain (the snap-fit chain); the
# commands that draw or build hardware, and synth's realistic scenes, with the
# O-ring dot chain they are made for. --chain picks another for any of them.
DEFAULT_CHAIN = default_chain_path()
ORING_CHAIN = oring_chain_path()


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
    p.add_argument("--pins", type=int, help="number of pins on the chain in this photo "
                                            "(default: n_pins from the chain file)")
    p.add_argument("--truth", type=Path, help="truth JSON from synth/test-part; reports error")
    _add_shade_arg(p)

    p = sub.add_parser("synth", help="render a synthetic chain photo with known geometry")
    p.add_argument("--chain", type=Path, help=f"chain file (default: {DEFAULT_CHAIN.name}, the snap-fit "
                                                  f"chain; with --env, {ORING_CHAIN.name}, the O-ring chain)")
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
    _add_shade_arg(p)

    p = sub.add_parser("test-part", help="write a printable chain drawing with known geometry")
    p.add_argument("--chain", type=Path, default=ORING_CHAIN)
    p.add_argument("--out", type=Path, default=Path("out/test-part"))
    _add_shape_args(p)

    p = sub.add_parser("test-plaque", help="write a 3D-printable chain plaque with known geometry")
    p.add_argument("--chain", type=Path, default=ORING_CHAIN)
    p.add_argument("--out", type=Path, default=Path("out/test-plaque"))
    _add_shape_args(p)

    p = sub.add_parser("test-gauges", help="write the 3D-printable contact gauges for the real-world tests")
    p.add_argument("names", nargs="*", help="which gauges (default: all): " + ", ".join(GAUGES))
    p.add_argument("--out", type=Path, default=Path("out/test-gauges"))
    p.add_argument("--chain", type=Path, default=DEFAULT_CHAIN, help=argparse.SUPPRESS)

    p = sub.add_parser("fixed-chain", help="write the fixed-chain test plaque: the snap-fit link posed and fused (STL)")
    p.add_argument("--pose", choices=list(PLAQUE_POSES), default="plaque-s")
    p.add_argument("--links", type=int, help="number of links (default: n_pins from the chain file)")
    p.add_argument("--out", type=Path, default=Path("out/fixed-chain"))
    p.add_argument("--chain", type=Path, default=DEFAULT_CHAIN, help=argparse.SUPPRESS)

    p = sub.add_parser("score", help="score test photos against their gauge or fixed-chain plaque")
    p.add_argument("photos", type=Path, nargs="*", help="photos of one target (or use --report)")
    p.add_argument("--target", help="the target's name: " + ", ".join(target_names()))
    p.add_argument("--report", type=Path, help="the test page's report (JSON): each test's photo, chain, "
                                               "pin count and target")
    p.add_argument("--photo-dir", type=Path, help="where the report's photos are (default: beside it)")
    p.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    p.add_argument("--pins", type=int, help="number of pins (default: n_pins from the chain file)")
    p.add_argument("--saved-focal-35mm", type=float, help="the phone's focal length, for photos without EXIF")
    p.add_argument("--fit-focal", action="store_true",
                   help="also find the focal length that fits the target best (slower)")
    p.add_argument("--json", type=Path, help="write every score to this JSON file")
    _add_shade_arg(p)

    args = parser.parse_args(argv)
    if args.command == "synth" and args.chain is None:
        args.chain = ORING_CHAIN if args.env else DEFAULT_CHAIN
    commands = {"measure": _measure, "synth": _synth, "test-part": _test_part, "test-plaque": _test_plaque,
                "score": _score, "test-gauges": _test_gauges,
                "fixed-chain": _fixed_chain}
    # ValueErrors carry the program's own messages for people (a photo with no
    # chain in it, a setting out of range): show the message, not a traceback.
    try:
        spec = load_chain_spec(args.chain)
        if getattr(args, "dark_fiducials", False):
            spec = replace(spec, fiducial_shade="dark")
        return commands[args.command](args, spec)
    except KeyError as err:
        print(f"error: {args.chain} has no {err.args[0]}", file=sys.stderr)
    except (ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
    return 1


def _presets() -> dict:
    from splinewire.scene import PRESETS
    return PRESETS


def _add_shade_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dark-fiducials", action="store_true",
                   help="the chain's fiducials are darker than its links (black on white links); "
                        "the default is the chain file's fiducial_shade, normally light on dark")


def _add_shape_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shape", choices=["s-curve", "pipe", "cove"], default="s-curve")
    p.add_argument("--radius", type=float, default=30.0, help="radius for pipe/cove shapes, mm")


def _shape(args, spec: ChainSpec) -> np.ndarray:
    if args.shape == "s-curve":
        return s_curve_pins(spec)
    return circle_wrap_pins(spec, args.radius, concave=args.shape == "cove")


def _measure(args, spec: ChainSpec) -> int:
    if args.pins is not None:
        spec = replace(spec, n_pins=args.pins)
    result = process_photo(args.photo, spec, args.out, focal_35mm=args.focal_35mm,
                           side=args.side, truth_path=args.truth, compare_truth=True)
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
    for n in m.notes:
        print(f"note: {n}")
    for w in m.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print("wrote " + ", ".join(p.name for p in result.outputs.values()) + f" in {args.out}")
    return 0


def _synth(args, spec: ChainSpec) -> int:
    w, h = (int(v) for v in args.size.lower().split("x"))
    pins = _shape(args, spec)
    truth = args.out / f"{args.shape}-truth.json"
    if args.env:
        if spec.contact == "round" or (spec.fiducial == "ring" and spec.ring_inner_mm / spec.fiducial_mm >= 0.6):
            raise ValueError("the realistic scenes (--env) draw the O-ring dot chain; "
                             f"leave out --chain or give {ORING_CHAIN.name}")
        if spec.fiducial_shade == "dark":
            raise ValueError("the realistic scenes (--env) draw the printed plaque, white on black; "
                             "leave out --dark-fiducials")
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


def _test_gauges(args, spec: ChainSpec) -> int:
    try:
        from splinewire.gauges import write_gauges
    except ImportError as err:
        print(f"test-gauges needs the dev dependencies (uv sync): {err}", file=sys.stderr)
        return 1
    unknown = [n for n in args.names if n not in GAUGES]
    if unknown:
        raise ValueError(f"no gauge {unknown[0]!r}; there are {', '.join(GAUGES)}")
    paths = write_gauges(args.out, args.names or None)
    print(paths["instructions"].read_text(encoding="utf-8"))
    print("wrote " + ", ".join(p.name for p in paths.values()) + f" in {args.out}")
    return 0


def _fixed_chain(args, spec: ChainSpec) -> int:
    try:
        from splinewire.fixed_chain import write_fixed_chain
    except ImportError as err:
        print(f"fixed-chain needs the dev dependencies (uv sync): {err}", file=sys.stderr)
        return 1
    if spec.contact != "round":
        raise ValueError("the fixed-chain plaque is made of the snap-fit link; leave out --chain")
    if args.links is not None and not 6 <= args.links <= 40:
        raise ValueError(f"--links must be between 6 and 40, got {args.links}")
    paths = write_fixed_chain(args.out, args.pose, spec, args.links)
    print(paths["instructions"].read_text(encoding="utf-8"))
    print("wrote " + ", ".join(p.name for p in paths.values()) + f" in {args.out}")
    return 0


def _score(args, spec: ChainSpec) -> int:
    import json

    from splinewire.chain import chain_presets
    from splinewire.scoring import REPORT_SCHEMA, headline, score_photo

    jobs = []           # (label, path, target, spec, saved focal)
    if args.report:
        doc = json.loads(args.report.read_text(encoding="utf-8"))
        if doc.get("schema") != REPORT_SCHEMA:
            raise ValueError(f"{args.report.name} is not a test page report ({REPORT_SCHEMA})")
        folder = args.photo_dir or args.report.parent
        presets = chain_presets()
        for t in doc["tests"]:
            if t.get("target") in (None, "object"):
                continue
            s = load_chain_spec(presets[t.get("chain") or "snap"])
            s = replace(s, n_pins=int(t["pins"]), fiducial_shade=t.get("shade") or s.fiducial_shade)
            jobs.append((t["id"], folder / t["photo"], t["target"], s, t.get("saved_focal_35mm")))
    else:
        if not args.photos or not args.target:
            raise ValueError("give photos and --target, or --report")
        if args.target not in target_names():
            raise ValueError(f"no target {args.target!r}; there are {', '.join(target_names())}")
        if args.pins is not None:
            spec = replace(spec, n_pins=args.pins)
        jobs = [(p.stem, p, args.target, spec, args.saved_focal_35mm) for p in args.photos]

    results = []
    print(f"{'test':14} {'target':13} {'pins':>4} {'tilt':>5}  {'EXIF/saved':>16}  {'estimated':>16}"
          + (f"  {'best fit':>16}" if args.fit_focal else "") + "  worst mm (focal mm)")
    for label, path, target, s, saved in jobs:
        r = score_photo(path.read_bytes(), path.name, target, s, saved_focal_35mm=saved, fit_focal=args.fit_focal)
        r["id"] = label
        results.append(r)
        if "error" in r:
            print(f"{label:14} {target:13} error: {r['error']}")
            continue
        cells = []
        for key in ("exif", "estimated", "fitted") if args.fit_focal else ("exif", "estimated"):
            e = r["focal"].get(key)
            cells.append(f"{'-':>16}" if e is None else f"{'failed':>16}" if "error" in e
                         else f"{headline(e['score']):7.3f} ({e['focal_35mm']:5.1f})")
        print(f"{label:14} {target:13} {r['pins_found']:4d} {r['tilt_deg']:5.1f}  " + "  ".join(cells))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
