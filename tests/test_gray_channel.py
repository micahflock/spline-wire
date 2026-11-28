"""The colour channel detection reads (ChainSpec gray_channel): the luma, as
always, or one of R, G, B alone, for coloured rims that read poorly in gray
(orange on off-white links: blue, with fiducial_shade "dark"). The figures
behind the choice: experiments/colour_channel/."""
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from PIL import Image

from splinewire.chain import GRAY_CHANNELS, chain_presets, load_chain_spec, spec_from_dict
from splinewire.cli import main as cli
from splinewire.pipeline import gray_from_image, load_photo
from splinewire.session import PhotoSession
from splinewire.synthetic import s_curve_pins, write_synthetic_photo

SNAP = load_chain_spec(chain_presets()["snap"])
ORANGE, OFF_WHITE, DARK_TABLE = (240, 122, 35), (235, 230, 214), (60, 60, 60)
COLOURS = {"rim": ORANGE, "link": OFF_WHITE, "table": DARK_TABLE}


def test_presets_read_the_luma_and_old_settings_read_as_luma():
    assert {load_chain_spec(p).gray_channel for p in chain_presets().values()} == {"luma"}
    raw = {k: v for k, v in asdict(SNAP).items() if k != "gray_channel"}
    assert spec_from_dict(raw).gray_channel == "luma"              # saved before the setting existed
    for ch in GRAY_CHANNELS:
        assert spec_from_dict({**raw, "gray_channel": ch}).gray_channel == ch
    with pytest.raises(ValueError, match="gray_channel must be one of"):
        spec_from_dict({**raw, "gray_channel": "blue"})


def test_each_channel_is_pillows():
    rng = np.random.default_rng(1)
    rgb = Image.fromarray(rng.integers(0, 256, (37, 53, 3), dtype=np.uint8))
    assert np.array_equal(gray_from_image(rgb, "luma"), np.asarray(rgb.convert("L")))
    for k, ch in enumerate("rgb"):
        assert np.array_equal(gray_from_image(rgb, ch), np.asarray(rgb)[:, :, k])
        assert np.array_equal(gray_from_image(rgb, ch), np.asarray(rgb.getchannel(k)))
    gray = rgb.convert("L")                                         # a gray photo: each channel is its gray
    for ch in GRAY_CHANNELS:
        assert np.array_equal(gray_from_image(gray, ch), np.asarray(gray))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(7)
    assert np.array_equal(gray_from_image(rgba, "b"), np.asarray(rgb)[:, :, 2])
    with pytest.raises(ValueError, match="gray_channel"):
        gray_from_image(rgb, "x")


def test_load_photo_reads_the_channel_upright(tmp_path):
    rgb = np.zeros((20, 30, 3), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 30
    rgb[0, 0] = (200, 100, 50)
    exif = Image.Exif()
    exif[0x0112] = 6                                                # rotate 90 degrees to stand upright
    path = tmp_path / "turned.png"
    Image.fromarray(rgb).save(path, exif=exif)
    for ch, at, rest in (("r", 200, 10), ("g", 100, 20), ("b", 50, 30)):
        img, _ = load_photo(path, ch)
        assert img.shape == (30, 20) and img[0, -1] == at and img[5, 5] == rest


@pytest.fixture(scope="module")
def orange(tmp_path_factory):
    """Orange rims on off-white links on a dark table, as a phone's JPEG."""
    path = tmp_path_factory.mktemp("orange") / "orange.jpg"
    write_synthetic_photo(path, s_curve_pins(SNAP), SNAP, (2000, 1500), distance_mm=260.0, tilt_deg=15.0,
                          seed=4, colours=COLOURS)
    return path


def test_orange_rims_are_found_in_blue_only(orange):
    """Not in gray, either way round; blue, with Colours dark, finds every pin
    (experiments/colour_channel/: 13 of 13 on six tables, worst pin 0.004 to
    0.006 mm, where gray found none on a dark table or a green mat)."""
    s = PhotoSession.open(orange.read_bytes(), orange.name)
    dark = replace(SNAP, fiducial_shade="dark")
    truth = {"name": "truth", "pin_points": s_curve_pins(SNAP).tolist()}
    for spec in (SNAP, dark, replace(SNAP, gray_channel="b"), replace(dark, gray_channel="r")):
        doc = s.measure({"chain": asdict(spec)})
        assert not doc["ok"] and "need the whole chain in view" in doc["error"], spec
    doc = s.measure({"chain": asdict(replace(dark, gray_channel="b")), "truth": truth})
    json.dumps(doc)
    assert doc["ok"] and doc["result"]["pins"] == SNAP.n_pins and doc["result"]["warnings"] == []
    assert doc["result"]["truth"]["max_error_mm"] < 0.02
    assert s.channel == "b"
    # Back to gray: read again, and not found again.
    assert not s.measure({"chain": asdict(dark)})["ok"] and s.channel == "luma"


def test_a_session_given_only_gray_says_so():
    s = PhotoSession("gray.png", np.zeros((50, 60), np.uint8), None, {})
    doc = s.measure({"chain": asdict(replace(SNAP, gray_channel="b"))})
    assert not doc["ok"] and "open it again to read its b channel" in doc["error"]


def test_the_cli_measures_with_a_channel(orange, tmp_path, capsys):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"pin_points": s_curve_pins(SNAP).tolist()}))
    assert cli(["measure", str(orange), "--dark-fiducials", "--out", str(tmp_path / "gray")]) == 1
    assert "need the whole chain in view" in capsys.readouterr().err
    assert cli(["measure", str(orange), "--dark-fiducials", "--channel", "b", "--truth", str(truth),
                "--out", str(tmp_path / "blue")]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"{SNAP.n_pins} pins found")
    doc = json.loads((tmp_path / "blue" / "orange.json").read_text())
    assert doc["truth_comparison"]["max_error_mm"] < 0.02


def test_synth_draws_a_coloured_chain(tmp_path, capsys):
    out = tmp_path / "synth"
    assert cli(["synth", "--shape", "pipe", "--size", "800x600", "--rim-rgb", "240,122,35", "--link-rgb", "235,230,214",
                "--out", str(out)]) == 0
    with Image.open(out / "pipe.jpg") as im:
        assert im.mode == "RGB"
        rgb = np.asarray(im)
    assert np.abs(rgb[..., 0].astype(int) - rgb[..., 2]).max() > 100       # coloured, not gray
    assert cli(["synth", "--table-rgb", "60,60,60", "--out", str(out)]) == 1
    with pytest.raises(SystemExit):                                         # argparse: not R,G,B
        cli(["synth", "--rim-rgb", "1,2", "--out", str(out)])
    assert "expected R,G,B" in capsys.readouterr().err
