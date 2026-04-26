from pathlib import Path

from validation.fiducial_geometry import strip_primitives
from validation.ground_truth import build_strip_geometry, load_strip_spec
from validation.synthetic import render_png, render_svg


def main() -> None:
    spec = load_strip_spec(Path("data/test-strip-spec.yaml"))
    geom = build_strip_geometry(spec)
    prims = strip_primitives(geom)

    out_dir = Path("data/photos")
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "strip.png"
    svg_path = out_dir / "strip.svg"

    render_png(prims, png_path, spec, px_per_mm=40)
    render_svg(prims, svg_path, spec)

    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
