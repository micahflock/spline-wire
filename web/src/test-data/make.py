"""Write measurement.json: a Python measurement for the TypeScript tests (see README.md)."""
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from splinewire.chain import load_chain_spec
from splinewire.session import PhotoSession
from splinewire.synthetic import s_curve_pins, write_synthetic_photo

HERE = Path(__file__).parent
REPO = HERE.parent.parent.parent


def main() -> None:
    spec = load_chain_spec(REPO / "data" / "chain.yaml")
    photo = Path(tempfile.mkdtemp()) / "IMG_0001.jpg"
    write_synthetic_photo(photo, s_curve_pins(spec), spec, (2000, 1500))
    s = PhotoSession.open(photo.read_bytes(), photo.name)
    doc = s.measure({"chain": asdict(spec)})
    out = {"document": doc, "outputs": s.outputs(), "chain": asdict(spec)}
    (HERE / "measurement.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
