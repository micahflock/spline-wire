Fixtures written by Python, so TypeScript tests check against the reference implementation.

- `measurement.json`: a `PhotoSession.measure` document, `outputs()` and the chain, for a synthetic
  13-pin s-curve photo (2000×1500). Regenerate with `uv run python web/src/test-data/make.py`.
