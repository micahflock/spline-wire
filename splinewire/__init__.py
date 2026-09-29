"""Spline Wire: measure real-world curves with a fiducial chain and a phone photo."""

__version__ = "0.4.0"


def version_string() -> str:
    """Version plus the CI build it came from, e.g. "0.4.0 (build 12, 8bdccf7)".

    CI writes splinewire/_build.py before packaging; local runs say "dev".
    """
    try:
        from splinewire._build import BUILD
    except ImportError:
        return f"{__version__} (dev)"
    return f"{__version__} (build {BUILD})"
