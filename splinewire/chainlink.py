"""A chain card's link: one chain's calibrated numbers in the phone page's address.

    https://micahflock.github.io/spline-wire/phone/#cal=1;base=snap;p=12.987;h=4.39;k=7

The grammar, the check character and worked examples are in docs/chain-link.md
(decision D30 in docs/web-build.md); this module and its TypeScript port,
web/src/pipeline/chainlink.ts, follow it, and both are checked against the same
test vectors (web/src/pipeline/chainlink-vectors.json). A change here is made
there too, with the same messages.

A link names its preset (`base`) and carries what differs from chain to chain
within a run: the pitch, and optionally the half width, the fiducials' shade,
the colour channel and the pin count. Everything else comes from the preset.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from urllib.parse import unquote

from splinewire.chain import SHADES, ChainSpec

VERSION = 1
PARAM = "cal"                 # the fragment parameter: #cal=... (beside #pair=...)
PITCH_BAND = 0.02             # a link's pitch is within 2% of its preset's
HALF_WIDTH_BAND = 0.10        # and its half width within 10%
MIN_PINS, MAX_PINS = 3, 200
# The colour channel the photo's gray image is made from (item 13's
# gray_channel; its names are shared with it).
CHANNELS = ("luma", "r", "g", "b")
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
# The phone page a card opens. ?debug while chain cards are being tested
# (work queue item 14): drop it once the feature is promoted.
PHONE_PAGE = "https://micahflock.github.io/spline-wire/phone/?debug"

# Field names in the order the builder writes them, between the version and k.
FIELDS = ("base", "p", "h", "shade", "ch", "n")
_NUMBER = re.compile(r"(0|[1-9][0-9]*)(\.[0-9]{1,3})?")
_NAME = re.compile(r"[a-z]+")
_DIGITS = re.compile(r"[0-9]+")
_BAD_ESCAPE = re.compile(r"%(?![0-9a-fA-F]{2})")
_RETRY = "scan it again or type the pitch in Settings"


class ChainLinkError(ValueError):
    """A link that cannot be used, with a message for a person. `code` says
    which check failed (the test vectors name it): not_a_link, damaged,
    version, cut_short, check, unknown_field, missing, unknown_preset,
    number, pitch, half_width, value."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ChainLink:
    """What a link says. None: not given, the preset's own value is used."""
    base: str
    pitch_mm: float
    half_width_mm: float | None = None
    fiducial_shade: str | None = None
    gray_channel: str | None = None
    n_pins: int | None = None

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# --- the check character -----------------------------------------------------

def _counted(text: str) -> str:
    """The characters the check covers: digits and letters, case-folded."""
    return "".join(c for c in text.lower() if c in ALPHABET)


def _luhn_sum(codes: list[int], n: int, double_last: bool) -> int:
    total, double = 0, double_last
    for code in reversed(codes):
        a = code * 2 if double else code
        total += a // n + a % n
        double = not double
    return total


def check_character(text: str, alphabet: str = ALPHABET) -> str:
    """Luhn mod N (N = len(alphabet), 36 by default) over the digits and
    letters of `text`; anything else is skipped."""
    n = len(alphabet)
    codes = [alphabet.index(c) for c in text.lower() if c in alphabet]
    return alphabet[(n - _luhn_sum(codes, n, double_last=True) % n) % n]


def check_ok(text: str, k: str, alphabet: str = ALPHABET) -> bool:
    """Whether `k` is the check character of `text`."""
    n = len(alphabet)
    codes = [alphabet.index(c) for c in (text + k).lower() if c in alphabet]
    return len(k) == 1 and k.lower() in alphabet and _luhn_sum(codes, n, double_last=False) % n == 0


# --- numbers -----------------------------------------------------------------

def _thousandths(x: float) -> int:
    """x in thousandths, half up (math.floor(x * 1000 + 0.5), as Math.round)."""
    return math.floor(x * 1000 + 0.5)


def fmt_mm(x: float) -> str:
    """Up to 3 decimals, trailing zeros dropped: 13.0 -> "13", 12.9870 -> "12.987"."""
    m = _thousandths(x)
    sign, m = ("-", -m) if m < 0 else ("", m)
    text = f"{sign}{m // 1000}.{m % 1000:03d}".rstrip("0").rstrip(".")
    return text


def _number(field: str, what: str, text: str) -> float:
    if not _NUMBER.fullmatch(text):
        raise ChainLinkError("number", f"The {what} ({field}={text}) is not a number of millimetres "
                                       "with up to 3 decimals, such as 12.987.")
    return float(text)


# --- checking what a link says ------------------------------------------------

def _nominal(presets: Mapping[str, ChainSpec], base: str) -> ChainSpec:
    if base not in presets:
        raise ChainLinkError("unknown_preset", f'Spline Wire has no chain preset named "{base}" '
                                               f"(there are {', '.join(presets)}): if the card is new, "
                                               "reload the page to update Spline Wire.")
    return presets[base]


def _in_band(what: str, value: float, base: str, nominal: float, band: float) -> None:
    lo, hi = nominal * (1 - band), nominal * (1 + band)
    if not abs(value - nominal) <= band * nominal + 1e-9:
        raise ChainLinkError(
            "pitch" if what == "pitch" else "half_width",
            f"The {what}, {fmt_mm(value)} mm, is more than "
            f"{round(band * 100)}% from the {base} chain's {fmt_mm(nominal)} mm: it must be between "
            f"{fmt_mm(lo)} and {fmt_mm(hi)} mm. Check the number, or measure the chain again.")


def check_link(link: ChainLink, presets: Mapping[str, ChainSpec]) -> ChainLink:
    """The link if every value is one a card may carry, else ChainLinkError."""
    nominal = _nominal(presets, link.base)
    _in_band("pitch", link.pitch_mm, link.base, nominal.pitch_mm, PITCH_BAND)
    if link.half_width_mm is not None:
        _in_band("half width", link.half_width_mm, link.base, nominal.half_width_mm, HALF_WIDTH_BAND)
    if link.fiducial_shade is not None and link.fiducial_shade not in SHADES:
        raise ChainLinkError("value", f"The colours (shade={link.fiducial_shade}) must be light or dark.")
    if link.gray_channel is not None and link.gray_channel not in CHANNELS:
        raise ChainLinkError("value", f"The colour channel (ch={link.gray_channel}) must be luma, r, g or b.")
    if link.n_pins is not None and not (isinstance(link.n_pins, int) and MIN_PINS <= link.n_pins <= MAX_PINS):
        raise ChainLinkError("value", f"The number of pins (n={link.n_pins}) must be a whole number "
                                      f"from {MIN_PINS} to {MAX_PINS}.")
    return link


# --- building ----------------------------------------------------------------

def link_value(link: ChainLink, presets: Mapping[str, ChainSpec]) -> str:
    """The `cal` parameter's value: "1;base=snap;p=12.987;h=4.39;k=7".
    Numbers are rounded to 3 decimals (half up) first, then checked."""
    for what, x in (("pitch", link.pitch_mm), ("half width", link.half_width_mm)):
        if x is not None and not math.isfinite(x):
            raise ChainLinkError("number", f"The {what} must be a number of millimetres, such as 12.987.")
    link = replace(link, pitch_mm=_thousandths(link.pitch_mm) / 1000)
    if link.half_width_mm is not None:
        link = replace(link, half_width_mm=_thousandths(link.half_width_mm) / 1000)
    check_link(link, presets)
    fields = [str(VERSION), f"base={link.base}", f"p={fmt_mm(link.pitch_mm)}"]
    if link.half_width_mm is not None:
        fields.append(f"h={fmt_mm(link.half_width_mm)}")
    if link.fiducial_shade is not None:
        fields.append(f"shade={link.fiducial_shade}")
    if link.gray_channel is not None:
        fields.append(f"ch={link.gray_channel}")
    if link.n_pins is not None:
        fields.append(f"n={link.n_pins}")
    body = ";".join(fields)
    return f"{body};k={check_character(body)}"


def link_url(link: ChainLink, presets: Mapping[str, ChainSpec], page: str = PHONE_PAGE) -> str:
    """The card's address: the phone page with #cal=<value>."""
    return f"{page.split('#')[0]}#{PARAM}={link_value(link, presets)}"


# --- parsing -----------------------------------------------------------------

def _decode(text: str) -> str:
    """Percent-decoded as decodeURIComponent decodes it: a % not followed by
    two hex digits, or escapes that are not UTF-8, raise UnicodeDecodeError."""
    if _BAD_ESCAPE.search(text):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "a bad % escape")
    return unquote(text, errors="strict")


def link_text(text: str) -> str:
    """The `cal` value in a link, a fragment or the value alone, percent-decoded."""
    text = text.strip()
    if "#" in text or text.lower().startswith(PARAM + "="):
        fragment = text.split("#", 1)[1] if "#" in text else text
        for part in fragment.split("&"):
            name, eq, value = part.partition("=")
            if name.lower() == PARAM and eq:
                return _decode(value).strip()
    elif re.match(r"[0-9]", text):
        return _decode(text)
    raise ChainLinkError("not_a_link", "This is not a chain card's link (it has no cal=): "
                                       "scan the card's QR code again or type the pitch in Settings.")


def parse_link(text: str, presets: Mapping[str, ChainSpec]) -> ChainLink:
    """What a chain card's link says, checked; ChainLinkError if it cannot be used.
    `text` is the card's address, its fragment, or the cal value alone."""
    try:
        value = link_text(text).lower()
    except UnicodeDecodeError:
        raise ChainLinkError("damaged", f"This chain card's link is damaged (a bad % escape): {_RETRY}.") from None
    fields = value.split(";")
    if not _DIGITS.fullmatch(fields[0]):
        raise ChainLinkError("damaged", f"This chain card's link is damaged (it does not start with "
                                        f"a version number): {_RETRY}.")
    version = int(fields[0])
    if version > VERSION:
        raise ChainLinkError("version", "This chain card was made for a newer version of Spline Wire "
                                        f"(link version {fields[0].lstrip('0')}): reload the page to update it.")
    if fields[0] != str(VERSION):
        raise ChainLinkError("damaged", f"This chain card's link is damaged (version {fields[0]}): {_RETRY}.")
    last = fields[-1] if len(fields) > 1 else ""
    if not (last.startswith("k=") and len(last) == 3 and last[2] in ALPHABET):
        raise ChainLinkError("cut_short", "This chain card's link is cut short (it does not end in "
                                          "a check character, ;k=): scan it again or copy the whole link.")
    if not check_ok(";".join(fields[:-1]), last[2]):
        raise ChainLinkError("check", f"This chain card's code is damaged (check character): {_RETRY}.")

    given: dict[str, str] = {}
    for f in fields[1:-1]:
        name, eq, v = f.partition("=")
        if not f:
            raise ChainLinkError("damaged", f"This chain card's link is damaged (an empty field): {_RETRY}.")
        if eq and _NAME.fullmatch(name) and not v:
            raise ChainLinkError("damaged", f"This chain card's link is damaged ({name} has no value): {_RETRY}.")
        if not (eq and _NAME.fullmatch(name)):
            raise ChainLinkError("damaged", f'This chain card\'s link is damaged (a field "{f}"): {_RETRY}.')
        if name in given:
            raise ChainLinkError("damaged", f"This chain card's link is damaged ({name} is given twice): {_RETRY}.")
        given[name] = v
    unknown = [n for n in given if n not in FIELDS]
    if unknown:
        raise ChainLinkError("unknown_field", "This chain card's link has a field this version of Spline Wire "
                                              f"does not know ({unknown[0]}={given[unknown[0]]}): reload the page "
                                              "to update it, or type the pitch in Settings.")
    if "base" not in given:
        raise ChainLinkError("missing", f"This chain card's link has no chain preset (base=): {_RETRY}.")
    if "p" not in given:
        raise ChainLinkError("missing", f"This chain card's link has no pitch (p=): {_RETRY}.")
    _nominal(presets, given["base"])
    n = given.get("n")
    if n is not None and not re.fullmatch(r"[1-9][0-9]*", n):
        raise ChainLinkError("value", f"The number of pins (n={n}) must be a whole number "
                                      f"from {MIN_PINS} to {MAX_PINS}.")
    link = ChainLink(
        base=given["base"],
        pitch_mm=_number("p", "pitch", given["p"]),
        half_width_mm=_number("h", "half width", given["h"]) if "h" in given else None,
        fiducial_shade=given.get("shade"),
        gray_channel=given.get("ch"),
        n_pins=int(n) if n is not None else None,
    )
    return check_link(link, presets)


# --- using it ----------------------------------------------------------------

def apply_link(spec: ChainSpec, link: ChainLink) -> ChainSpec:
    """The preset's spec with the link's numbers over it. The colour channel is
    only carried (link.gray_channel) until the spec has one (work queue item 13)."""
    return replace(
        spec,
        pitch_mm=link.pitch_mm,
        half_width_mm=spec.half_width_mm if link.half_width_mm is None else link.half_width_mm,
        fiducial_shade=link.fiducial_shade or spec.fiducial_shade,
        n_pins=spec.n_pins if link.n_pins is None else link.n_pins,
    )


def preset_specs() -> dict[str, ChainSpec]:
    """The presets that ship (data/chain*.yaml), by name."""
    from splinewire.chain import chain_presets, load_chain_spec
    return {name: load_chain_spec(path) for name, path in chain_presets().items()}


# --- the card ----------------------------------------------------------------

def qr_matrix(text: str) -> list[list[bool]]:
    """The QR code of `text` (error correction M or better), dark modules True,
    without its quiet zone."""
    import segno
    q = segno.make(text, error="m", micro=False)
    return [[bool(v) for v in row] for row in q.matrix]


def card_svg(link: ChainLink, url: str, presets: Mapping[str, ChainSpec]) -> str:
    """A card 85.6 x 54 mm (a bank card) with the link's QR code, its numbers
    in words and the link itself, for a person to type if the code will not scan."""
    from xml.sax.saxutils import escape
    m = qr_matrix(url)
    size, quiet = len(m), 4                     # the quiet zone the standard asks for, in modules
    qr_mm = 40.0                                # with its quiet zone
    cell = qr_mm / (size + 2 * quiet)
    x0, y0 = 1.0, 1.0
    rects = "".join(f'<rect x="{x0 + (c + quiet) * cell:.4f}" y="{y0 + (r + quiet) * cell:.4f}" '
                    f'width="{cell:.4f}" height="{cell:.4f}"/>'
                    for r, row in enumerate(m) for c, v in enumerate(row) if v)
    nominal = presets[link.base]
    lines = [("Spline Wire", 4.2, True), (f"{link.base} chain", 3.4, True), ("", 2.0, False),
             (f"pitch {fmt_mm(link.pitch_mm)} mm", 3.4, False)]
    if link.half_width_mm is not None:
        lines.append((f"half width {fmt_mm(link.half_width_mm)} mm", 3.4, False))
    if link.fiducial_shade is not None:
        lines.append((f"{link.fiducial_shade} fiducials", 3.4, False))
    if link.gray_channel is not None:
        lines.append((f"channel {link.gray_channel}", 3.4, False))
    lines.append((f"{link.n_pins if link.n_pins is not None else nominal.n_pins} pins", 3.4, False))
    words, y = [], 6.0
    for text, size_mm, bold in lines:
        if text:
            weight = ' font-weight="bold"' if bold else ""
            words.append(f'<text x="43" y="{y:.1f}" font-size="{size_mm}"{weight}>{escape(text)}</text>')
        y += size_mm * 1.45
    # The link in two lines, the page and the cal value, for a person to type.
    page, _, value = url.partition("#")
    small = (f'<text x="4" y="46" font-size="2.1" font-family="monospace">{escape(page)}#</text>'
             f'<text x="4" y="49.5" font-size="2.1" font-family="monospace">{escape(value)}</text>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" width="85.6mm" height="54mm" viewBox="0 0 85.6 54">\n'
            '<rect x="0.1" y="0.1" width="85.4" height="53.8" rx="3" fill="#fff" stroke="#999" stroke-width="0.2"/>\n'
            f'<g fill="#000">{rects}</g>\n'
            f'<g font-family="sans-serif" fill="#000">{"".join(words)}</g>\n'
            f'<g fill="#000">{small}</g>\n'
            '</svg>\n')
