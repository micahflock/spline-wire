"""A chain card's link (splinewire/chainlink.py, docs/chain-link.md) against the
shared test vectors that web/src/pipeline/chainlink.test.ts reads too."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.chainlink import (
    ALPHABET,
    ChainLink,
    ChainLinkError,
    apply_link,
    card_svg,
    check_character,
    check_ok,
    fmt_mm,
    link_url,
    link_value,
    parse_link,
    preset_specs,
    qr_matrix,
)
from tests.conftest import REPO

VECTORS = json.loads((REPO / "web" / "src" / "pipeline" / "chainlink-vectors.json").read_text(encoding="utf-8"))
PRESETS = {name: SimpleNamespace(**v) for name, v in VECTORS["presets"].items()}


def test_vectors_presets_are_the_real_ones():
    """A preset named on a printed card is never changed in place (D30)."""
    real = preset_specs()
    for name, p in PRESETS.items():
        assert (real[name].pitch_mm, real[name].half_width_mm, real[name].n_pins) == (p.pitch_mm, p.half_width_mm, p.n_pins)


@pytest.mark.parametrize("case", VECTORS["check_characters"], ids=lambda c: c["text"] or "empty")
def test_check_characters(case):
    alphabet = case.get("alphabet", ALPHABET)
    assert check_character(case["text"], alphabet) == case["k"]
    assert check_ok(case["text"], case["k"], alphabet)


@pytest.mark.parametrize("case", VECTORS["good"], ids=lambda c: c["link"][-40:])
def test_good_links(case):
    assert parse_link(case["link"], PRESETS).to_json() == case["parsed"]


@pytest.mark.parametrize("case", VECTORS["bad"], ids=lambda c: c["error"] + ":" + c["link"][-30:])
def test_bad_links(case):
    with pytest.raises(ChainLinkError) as err:
        parse_link(case["link"], PRESETS)
    assert (err.value.code, str(err.value)) == (case["error"], case["message"])
    assert isinstance(err.value, ValueError)            # shown to the person as is


def test_vectors_cover_the_cases_the_spec_lists():
    codes = {c["error"] for c in VECTORS["bad"]}
    assert {"check", "pitch", "version", "cut_short", "unknown_field"} <= codes
    assert len(VECTORS["good"]) >= 10


@pytest.mark.parametrize("case", VECTORS["build"], ids=lambda c: c["value"])
def test_build(case):
    assert link_value(ChainLink(**case["fields"]), PRESETS) == case["value"]
    assert parse_link(case["value"], PRESETS).base == case["fields"]["base"]


@pytest.mark.parametrize("case", VECTORS["build_bad"], ids=lambda c: c["error"])
def test_build_refuses(case):
    with pytest.raises(ChainLinkError) as err:
        link_value(ChainLink(**case["fields"]), PRESETS)
    assert (err.value.code, str(err.value)) == (case["error"], case["message"])


def test_check_catches_one_changed_character_and_swaps_but_0_z():
    body = "1;base=snap;p=12.987;h=4.39"
    k = check_character(body)
    changed = [body[:i] + r + body[i + 1:] for i, c in enumerate(body) if c in ALPHABET for r in ALPHABET if r != c]
    assert len(changed) == 19 * 35 and not any(check_ok(t, k) for t in changed)
    n = 36
    f = lambda x: (2 * x) // n + (2 * x) % n             # a code doubled, its digits summed
    missed = [ALPHABET[a] + ALPHABET[b] for a in range(n) for b in range(a + 1, n) if (f(a) + b - f(b) - a) % n == 0]
    assert missed == ["0z"]
    assert not check_ok(body.replace("12.987", "12.978"), k)


def test_numbers_are_written_as_the_port_writes_them():
    assert [fmt_mm(x) for x in (13, 12.987, 12.9875, 4.39, 0.0004, 3.951)] == ["13", "12.987", "12.988", "4.39", "0", "3.951"]


def test_url_and_apply():
    url = link_url(ChainLink("snap", 12.987, 4.39), PRESETS)
    assert url == "https://micahflock.github.io/spline-wire/phone/?debug#cal=1;base=snap;p=12.987;h=4.39;k=k"
    assert link_url(ChainLink("snap", 13.0), PRESETS, "https://example.org/phone/#pair=x") == \
        "https://example.org/phone/#cal=1;base=snap;p=13;k=1"
    snap = load_chain_spec(default_chain_path())
    assert apply_link(snap, ChainLink("snap", 12.9)) == replace(snap, pitch_mm=12.9)
    assert apply_link(snap, ChainLink("snap", 13.1, 4.2, "dark", "b", 20)) == \
        replace(snap, pitch_mm=13.1, half_width_mm=4.2, fiducial_shade="dark", n_pins=20)


def test_card_qr_code_decodes_to_the_link():
    import cv2
    url = link_url(ChainLink("snap", 12.987, 4.39, "dark", "b"), PRESETS)
    m = np.array(qr_matrix(url), dtype=bool)
    image = np.pad(np.where(m, 0, 255).astype(np.uint8), 4, constant_values=255)
    image = np.kron(image, np.ones((8, 8), np.uint8))
    assert cv2.QRCodeDetector().detectAndDecode(image)[0] == url
    svg = card_svg(parse_link(url, PRESETS), url, PRESETS)
    assert svg.count("<rect") == 1 + int(m.sum())       # the card's outline and every dark module
    assert 'width="85.6mm"' in svg and "pitch 12.987 mm" in svg and "half width 4.39 mm" in svg
    assert "cal=1;base=snap;p=12.987;h=4.39;shade=dark;ch=b;k=" in svg
