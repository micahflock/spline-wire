"""A HEIC photo and what Python makes of it, for the TypeScript HEIF decoder's
tests (web/src/pipeline/heif.test.ts).

    uv run python web/scripts/golden_heif.py

A 1200x900 colour render of the snap-fit chain, stored as an iPhone stores a
photo held upright: the pixels turned a quarter turn, an irot box in the
container that turns them back (added here by hand: pillow-heif writes none),
and EXIF orientation 6 as well, which a reader must not apply on top
(pillow-heif resets it to 1); Make, Model and the focal lengths in the EXIF.
Writes
golden/photo-snap.heic, golden/photo-snap-heic.png (Python's decode, 8-bit
gray) and golden/heif.json.gz (photo_info and the measurement document).
"""
from __future__ import annotations

import gzip
import io
import json
import struct
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pillow_heif import register_heif_opener  # noqa: E402

from splinewire.camera import focal_px_from_35mm, look_at_plane  # noqa: E402
from splinewire.chain import chain_presets, load_chain_spec  # noqa: E402
from splinewire.pipeline import load_photo_bytes  # noqa: E402
from splinewire.session import PhotoSession, photo_info  # noqa: E402
from splinewire.synthetic import circle_wrap_pins, render_snap_photo  # noqa: E402

register_heif_opener()

DIR = REPO / "web" / "src" / "pipeline" / "golden"
SIZE = (1200, 900)


def _boxes(d: bytes, off: int, end: int):
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", d[off:off + 8])
        yield off, size, typ.decode("latin1")
        off += size


def add_irot(d: bytes, angle: int) -> bytes:
    """The HEIF file with an irot property (anticlockwise quarter turns) on its
    primary item: added to ipco and ipma, the boxes holding them grown, and
    iloc's offsets moved past the larger meta box."""
    meta = next(b for b in _boxes(d, 0, len(d)) if b[2] == "meta")
    inner = {t: (o, s) for o, s, t in _boxes(d, meta[0] + 12, meta[0] + meta[1])}
    iprp = inner["iprp"]
    ip = {t: (o, s) for o, s, t in _boxes(d, iprp[0] + 8, iprp[0] + iprp[1])}
    ipco, ipma = ip["ipco"], ip["ipma"]
    n_props = sum(1 for _ in _boxes(d, ipco[0] + 8, ipco[0] + ipco[1]))
    irot = struct.pack(">I4sB", 9, b"irot", angle)
    # ipma: version 0, flags 0 (one-byte associations), one entry for the primary item.
    o = ipma[0] + 8
    version, flags = d[o], int.from_bytes(d[o + 1:o + 4], "big")
    assert version == 0 and flags & 1 == 0, (version, flags)
    count = struct.unpack(">I", d[o + 4:o + 8])[0]
    assert count == 1, count
    item, n = struct.unpack(">HB", d[o + 8:o + 11])
    assocs = d[o + 11:o + 11 + n] + bytes([0x80 | (n_props + 1)])          # essential
    body = d[o:o + 8] + struct.pack(">HB", item, n + 1) + assocs
    new_ipma = struct.pack(">I4s", 8 + len(body), b"ipma") + body
    new_ipco = struct.pack(">I4s", ipco[1] + len(irot), b"ipco") + d[ipco[0] + 8:ipco[0] + ipco[1]] + irot
    rest = d[iprp[0] + 8:iprp[0] + iprp[1]]
    rest = rest.replace(d[ipco[0]:ipco[0] + ipco[1]], new_ipco).replace(d[ipma[0]:ipma[0] + ipma[1]], new_ipma)
    new_iprp = struct.pack(">I4s", 8 + len(rest), b"iprp") + rest
    grow = len(new_iprp) - iprp[1]
    # iloc (version 0 or 1): move every extent offset by `grow`.
    il = bytearray(d[inner["iloc"][0]:inner["iloc"][0] + inner["iloc"][1]])
    v = il[8]
    off_size, len_size, base_size = il[12] >> 4, il[12] & 15, il[13] >> 4
    assert v == 0 and off_size == 4 and base_size in (0, 4), (v, off_size, base_size)

    def shift(at: int) -> None:
        il[at:at + 4] = struct.pack(">I", struct.unpack(">I", il[at:at + 4])[0] + grow)

    p, items = 16, struct.unpack(">H", il[14:16])[0]
    for _ in range(items):
        p += 2 + 2                                      # item id, data reference index
        if base_size:                                   # offsets are base + extent: move the base
            shift(p)
        p += base_size
        extents = struct.unpack(">H", il[p:p + 2])[0]
        p += 2
        for _ in range(extents):
            if not base_size:
                shift(p)
            p += 4 + len_size
    meta_body = b""
    for o2, s2, t2 in _boxes(d, meta[0] + 12, meta[0] + meta[1]):
        meta_body += new_iprp if t2 == "iprp" else bytes(il) if t2 == "iloc" else d[o2:o2 + s2]
    new_meta = struct.pack(">I4s", 12 + len(meta_body), b"meta") + d[meta[0] + 8:meta[0] + 12] + meta_body
    assert len(new_meta) == meta[1] + grow
    return d[:meta[0]] + new_meta + d[meta[0] + meta[1]:]


def main() -> None:
    snap = load_chain_spec(chain_presets()["snap"])
    pins = circle_wrap_pins(snap, 40.0)
    cam = look_at_plane(focal_px_from_35mm(26, SIZE), SIZE, 260.0, tilt_deg=20.0, tilt_direction_deg=60.0,
                        target_mm=tuple(pins.mean(axis=0)))
    gray = render_snap_photo(pins, snap, cam, rng=np.random.default_rng(3), table_gray=200).astype(float)
    # A colour photo, so the decoder's YCbCr to RGB matters: a warm table and chain.
    rgb = np.clip(np.stack([gray * 1.05 + 4, gray * 0.95, gray * 0.8 + 12], axis=-1), 0, 255).astype(np.uint8)
    # Stored a quarter turn clockwise; irot 1 (a quarter turn anticlockwise) turns it back.
    im = Image.fromarray(rgb, "RGB").transpose(Image.Transpose.ROTATE_270)
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x0110] = "iPhone 15"
    exif[0x0112] = 6                                  # as an iPhone writes it; irot is what turns
    sub = exif.get_ifd(0x8769)
    sub[0x920A] = IFDRational(61, 10)                 # FocalLength, mm (a RATIONAL, as cameras write it)
    sub[0xA405] = 26                                  # FocalLengthIn35mmFilm
    buf = io.BytesIO()
    im.save(buf, format="HEIF", exif=exif, quality=50)
    data = add_irot(buf.getvalue(), 1)
    (DIR / "photo-snap.heic").write_bytes(data)

    decoded, _ = load_photo_bytes(data, "photo-snap.heic")
    Image.fromarray(decoded, "L").save(DIR / "photo-snap-heic.png", optimize=True)
    with Image.open(io.BytesIO(data)) as check:
        assert check.size == SIZE, check.size      # stored turned, read upright
    assert decoded.shape == (SIZE[1], SIZE[0]), decoded.shape
    session = PhotoSession.open(data, "photo-snap.heic")
    options = {"chain": asdict(snap)}
    doc = session.measure(options)
    assert doc["ok"], doc.get("error")
    out = {"info": photo_info(io.BytesIO(data)), "options": options, "document": doc}
    with gzip.open(DIR / "heif.json.gz", "wt", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"photo-snap.heic: {len(data)} bytes, {doc['result']['pins']} pins, focal {doc['result']['focal_source']}")


if __name__ == "__main__":
    main()
