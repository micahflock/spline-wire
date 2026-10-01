"""Printable test part: a chain drawing with exactly known pin positions.

Print the SVG at 100% scale (check the 50 mm scale bar), photograph it and
run `splinewire measure --truth` to measure the pipeline's real-world error
before any chain hardware exists.
"""
from __future__ import annotations

import numpy as np

from splinewire.chain import ChainSpec


def test_part_svg(pins_mm: np.ndarray, spec: ChainSpec, margin_mm: float = 12.0) -> str:
    lo = pins_mm.min(axis=0) - margin_mm
    hi = pins_mm.max(axis=0) + margin_mm
    hi[1] += 12.0  # room for the scale bar and label
    w, h = hi - lo

    def xy(p) -> tuple[float, float]:
        return float(p[0] - lo[0]), float(hi[1] - p[1])

    chain_d = "M " + " L ".join(f"{x:.4f} {y:.4f}" for x, y in map(xy, pins_mm))
    ro, ri = spec.fiducial_mm / 2, spec.ring_inner_mm / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.3f}mm" height="{h:.3f}mm" '
        f'viewBox="0 0 {w:.3f} {h:.3f}">',
        f'<rect width="{w:.3f}" height="{h:.3f}" fill="#fff"/>',
        # Link bodies: a stroke of the chain's full width with round ends.
        f'<path d="{chain_d}" fill="none" stroke="#222" stroke-width="{2 * spec.half_width_mm}" '
        'stroke-linecap="round" stroke-linejoin="round"/>',
    ]
    for x, y in map(xy, pins_mm):
        parts.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="{ro}" fill="#fff"/>')
        if spec.fiducial == "ring":
            parts.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="{ri}" fill="#222"/>')
    parts += [
        '<path d="M 6 6 L 56 6" stroke="#000" stroke-width="0.5"/>',
        '<path d="M 6 4 L 6 8 M 56 4 L 56 8" stroke="#000" stroke-width="0.3"/>',
        '<text x="6" y="12" font-family="sans-serif" font-size="3">'
        f'50 mm - print at 100% | spline-wire test part, pitch {spec.pitch_mm} mm</text>',
        "</svg>",
    ]
    return "\n".join(parts) + "\n"
