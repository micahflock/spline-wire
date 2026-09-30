"""Write splinewire/_build.py with the CI run number and commit.

splinewire.version_string() shows it in the app's title bar so it's easy to
tell which build is running. Run by the Windows workflow before packaging.
"""
import os
from pathlib import Path

run = os.environ.get("GITHUB_RUN_NUMBER", "local")
sha = os.environ.get("GITHUB_SHA", "")[:7] or "unknown"
target = Path(__file__).resolve().parent.parent / "splinewire" / "_build.py"
target.write_text(f'BUILD = "{run}, {sha}"\n', encoding="utf-8")
print(f"wrote {target}: build {run}, {sha}")
