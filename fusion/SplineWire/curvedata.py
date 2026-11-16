"""Clipboard reading and point parsing for the Spline Wire Fusion add-in.

Pure Python with no Fusion imports, so it can be unit-tested outside Fusion.
"""
from __future__ import annotations

import re
import subprocess
import sys

MM_PER_UNIT = {"mm": 1.0, "cm": 10.0, "in": 25.4}
_SEPARATORS = re.compile(r"[\t,; ]+")
_UNIT = re.compile(r"(?<![a-z])(mm|cm|in|inch|inches)(?![a-z])")


def parse_points(text: str) -> list[tuple[float, float]]:
    """Points in mm from a pasted table.

    One point per line: two or three numbers separated by tabs, commas,
    semicolons or spaces (a z value is ignored). Other lines, like a header,
    are skipped. Units default to mm; a header mentioning cm or in (e.g.
    "x_cm", "X (in)", "inches") switches them.
    """
    scale = 1.0
    points: list[tuple[float, float]] = []
    for line in (text or "").splitlines():
        fields = [f for f in _SEPARATORS.split(line.strip()) if f]
        values = _floats(fields)
        if values is None or not 2 <= len(values) <= 3:
            unit = _unit_scale(line)
            if unit is not None and not points:
                scale = unit
            continue
        points.append((values[0] * scale, values[1] * scale))
    if len(points) < 2:
        raise ValueError(
            "The clipboard doesn't hold a list of points. In Spline Wire, select a "
            "processed photo and click \u201cCopy points\u201d, then try again."
        )
    return points


def _unit_scale(line: str) -> float | None:
    """Millimetres per unit, if a header line names a unit. mm or cm win over
    "in", which is also a word: "points in mm" is not inches."""
    named = set(_UNIT.findall(line.lower()))
    for unit in ("mm", "cm"):
        if unit in named:
            return MM_PER_UNIT[unit]
    return MM_PER_UNIT["in"] if named else None


def _floats(fields: list[str]) -> list[float] | None:
    try:
        return [float(f) for f in fields]
    except ValueError:
        return None


def read_clipboard() -> str:
    """Text on the system clipboard ("" if there is none)."""
    if sys.platform == "win32":
        return _read_clipboard_windows()
    if sys.platform == "darwin":
        return subprocess.run(["pbpaste"], capture_output=True, text=True, check=False).stdout
    return ""


def _read_clipboard_windows() -> str:
    import ctypes
    from ctypes import wintypes

    CF_UNICODETEXT = 13
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    if not user32.OpenClipboard(None):
        return ""
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        ptr = kernel32.GlobalLock(handle)
        try:
            return ctypes.wstring_at(ptr) if ptr else ""
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()
