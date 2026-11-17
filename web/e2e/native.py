"""The native side of the end-to-end test: make the synthetic photos and say
what Python measures in them, for e2e/core.spec.ts to compare the browser with.

    uv run python web/e2e/native.py OUT_DIR      -> OUT_DIR/native.json

For each photo: the pins in photo pixels and the clipboard table as
PhotoSession measures them natively, and the truth comparison as
`splinewire measure --truth` reports it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.cli import main as cli
from splinewire.session import PhotoSession

# name -> synth arguments: a clean 12-megapixel photo of the pipe shape, and the
# printed plaque in the "shadow" scene (4032x3024, uneven light, JPEG).
PHOTOS = {"pipe": ["--shape", "pipe"], "shadow": ["--env", "shadow"]}


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    spec = load_chain_spec(default_chain_path())
    result = {"chain": asdict(spec), "photos": {}}
    for name, args in PHOTOS.items():
        folder = out / name
        if cli(["synth", *args, "--out", str(folder)]) != 0:
            raise SystemExit(f"synth {name} failed")
        photo = next(p for p in folder.glob("*.jpg"))
        truth = next(folder.glob("*-truth.json"))
        if cli(["measure", str(photo), "--truth", str(truth), "--out", str(folder / "cli")]) != 0:
            raise SystemExit(f"measure {name} failed")
        cli_doc = json.loads((folder / "cli" / f"{photo.stem}.json").read_text())
        s = PhotoSession.open(photo.read_bytes(), photo.name)
        doc = s.measure({"chain": asdict(spec)})
        result["photos"][name] = {
            "photo": str(photo), "truth": json.loads(truth.read_text()),
            "cli_truth": cli_doc["truth_comparison"],
            "pins_px": [[p["x"], p["y"]] for p in doc["view"]["pins"]],
            "focal_px": s.exif_focal_px,
            "pins_mm": doc["curve"]["pins"],
            "tsv": s.outputs()["tsv"],
        }
    (out / "native.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
