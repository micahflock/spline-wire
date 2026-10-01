import re

from splinewire.synthetic import s_curve_pins
from splinewire import testpart


def _circles(svg: str) -> list[tuple[float, str]]:
    return [(float(r), fill) for r, fill in re.findall(r'<circle [^>]*r="([\d.]+)" fill="(#\w+)"', svg)]


def test_test_part_draws_the_chains_fiducial(spec):
    svg = testpart.test_part_svg(s_curve_pins(spec), spec)
    circles = _circles(svg)
    outer = spec.fiducial_mm / 2
    if spec.fiducial == "dot":
        assert circles == [(outer, "#fff")] * spec.n_pins            # a solid light disc per pin
    else:
        inner = spec.ring_inner_mm / 2
        assert circles == [(outer, "#fff"), (inner, "#222")] * spec.n_pins   # disc with a link-coloured hole
