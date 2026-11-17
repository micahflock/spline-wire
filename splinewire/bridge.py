"""What the browser's core worker calls: one function per command of
web/src/core/protocol.ts, over the photos it holds open.

The worker (web/src/core/worker.ts) calls one function, `call`:

    call(cmd, args_json, data=None) -> str

- `cmd` is the protocol's command name: "open", "measure", "outputs",
  "close", "analyse" (and "stats").
- `args_json` is the command's arguments as a JSON string, except a photo's
  bytes. JSON both ways because the documents are small (a few kB) and a
  string crosses the JS/Python boundary in one copy, with no proxies to
  destroy afterwards.
- `data`, for "open" only, is the photo's bytes: bytes, a memoryview, or the
  JsProxy of a Uint8Array (anything with `to_bytes()`), so the worker can
  hand over the transferred buffer as it is.
- The return value is a JSON string, {"ok": true, "result": ...} or
  {"ok": false, "error": {"kind", "message", "type"?}}: the protocol's
  Response without the id. The error kinds are decided here, where the
  exception's class is known: ValueError is "person" (its message is for the
  person, as is), NotOpen is "closed" (open the photo again and retry), and
  anything else is "bug", with its type.

Memory decides the rest (web-build S1: a 12-megapixel photo takes the wasm
heap to ~580 MB, and the heap never shrinks). At most MAX_OPEN_PHOTOS
sessions are kept; opening one more closes the least recently used first, so
its memory can be reused for the new one.

ezdxf (only for the DXF output, 1.4 s to import under Pyodide) is not
imported here: `output.dxf_text` imports it when asked, and the worker loads
the package before the first "outputs".
"""
from __future__ import annotations

import gc
import json
import sys
from collections import OrderedDict
from pathlib import PurePosixPath, PureWindowsPath

from splinewire import __version__
from splinewire.chain import spec_from_dict
from splinewire.live import analyse_json
from splinewire.session import OUTPUT_NAMES, PhotoSession

# web/src/core/protocol.ts MAX_OPEN_PHOTOS: change both together.
MAX_OPEN_PHOTOS = 2


class NotOpen(Exception):
    """The photo is not open (never was, or was closed to make room). Not a
    ValueError: the page reopens it from its own copy, the person sees nothing."""


_open: OrderedDict[str, PhotoSession] = OrderedDict()     # least recently used first


def _session(photo: str) -> PhotoSession:
    s = _open.get(photo)
    if s is None:
        raise NotOpen(f"photo {photo!r} is not open")
    _open.move_to_end(photo)
    return s


def _close(photo: str) -> None:
    if _open.pop(photo, None) is not None:
        gc.collect()        # give the image's memory back to the allocator before the next one


def open_photo(photo: str, name: str, data) -> dict:
    """Decode a photo and keep it open as `photo`. ValueError if it is not one."""
    if hasattr(data, "to_bytes"):
        data = data.to_bytes()
    _close(photo)                       # opened again: the new bytes win
    while len(_open) >= MAX_OPEN_PHOTOS:
        _close(next(iter(_open)))       # before decoding, so the memory is free for it
    s = PhotoSession.open(bytes(data), name)
    _open[photo] = s
    return {"info": s.info}


def measure(photo: str, options: dict) -> dict:
    return _session(photo).measure(options)


def outputs(photo: str) -> dict:
    """The latest measurement's files, and the names to download them as.
    "tsv" (the clipboard table) has no file name."""
    s = _session(photo)
    files = s.outputs()
    stem = _stem(s.name)
    return {"files": files, "names": {k: v.format(stem=stem) for k, v in OUTPUT_NAMES.items()}}


def close(photo: str) -> None:
    _close(photo)


def analyse(frame: dict) -> dict:
    if not isinstance(frame, dict) or not isinstance(frame.get("chain"), dict):
        raise ValueError("a frame must be an object with points, diameters, width, height and a chain")
    try:
        spec_from_dict(frame["chain"])
        return analyse_json(frame)
    except KeyError as err:
        raise ValueError(f"the frame or its chain is missing {err.args[0]}") from None


def stats() -> dict:
    """Which photos are open, least recently used first."""
    return {"open": list(_open)}


def versions() -> dict:
    """Versions of what the measurement runs on, for the page's About and bug reports."""
    import cv2
    import numpy
    import PIL
    import scipy
    return {"splinewire": __version__, "python": sys.version.split()[0], "numpy": numpy.__version__,
            "scipy": scipy.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__}


def _stem(name: str) -> str:
    # A name from a Windows browser can carry a path; keep the file's own stem.
    return PurePosixPath(PureWindowsPath(name).name).stem or "photo"


_COMMANDS = {
    "open": lambda a, data: open_photo(str(a["photo"]), str(a.get("name") or a["photo"]), data),
    "measure": lambda a, _: measure(str(a["photo"]), a.get("options")),
    "outputs": lambda a, _: outputs(str(a["photo"])),
    "close": lambda a, _: close(str(a["photo"])),
    "analyse": lambda a, _: analyse(a),
    "stats": lambda a, _: stats(),
    "versions": lambda a, _: versions(),
}


def call(cmd: str, args_json: str = "{}", data=None) -> str:
    """Run one command; see the module's docstring. Never raises."""
    try:
        fn = _COMMANDS.get(cmd)
        if fn is None:
            raise LookupError(f"unknown command {cmd!r}")
        try:
            args = json.loads(args_json) if args_json else {}
        except ValueError as err:       # the worker's fault, not the person's
            raise TypeError(f"arguments are not JSON: {err}") from None
        result = fn(args, data)
    except NotOpen as err:
        return _error("closed", str(err))
    except ValueError as err:
        return _error("person", str(err))
    except Exception as err:            # a bug: say what it was, so it can be reported
        return _error("bug", str(err), type(err).__name__)
    try:
        return json.dumps({"ok": True, "result": result}, allow_nan=False)
    except (TypeError, ValueError) as err:   # NaN or a numpy type in a result: a bug here, not the person's
        return _error("bug", str(err), type(err).__name__)


def _error(kind: str, message: str, type_: str | None = None) -> str:
    error = {"kind": kind, "message": message}
    if type_:
        error["type"] = type_
    return json.dumps({"ok": False, "error": error})
