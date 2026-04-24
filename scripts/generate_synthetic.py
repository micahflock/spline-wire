from pathlib import Path

from validation.ground_truth import load_strip_spec
from validation.synthetic import render_set


def main() -> None:
    spec = load_strip_spec(Path("data/test-strip-spec.yaml"))
    out = Path("data/photos")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("A", "B"):  # skip C — synthetic ArUco is a placeholder
        img_path = out / f"{name}_top_day_01.png"
        render_set(spec.sets[name], img_path, px_per_mm=40)
        print(f"wrote {img_path}")


if __name__ == "__main__":
    main()
