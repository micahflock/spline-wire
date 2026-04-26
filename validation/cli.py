from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

from anthropic import Anthropic

from validation.extraction import ClaudeExtractor
from validation.ground_truth import load_strip_spec
from validation.schemas import ImageDetection, ScoringResult
from validation.scoring import score_detection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM fiducial extraction on a batch of images"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--photos", type=Path, required=True,
        help="directory of images; every image is treated as a photo of the canonical strip",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="claude-opus-4-7")
    parser.add_argument("--prompt-version", default="v2")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_strip_spec(args.spec)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    extractor = ClaudeExtractor(
        client=client, model=args.model, prompt_version=args.prompt_version
    )

    rows: list[dict] = []
    for img_path in sorted(args.photos.glob("*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".heic"}:
            continue

        print(f"extracting {img_path.name}...")
        detection = extractor.extract(img_path)
        _dump_detection(args.out, detection)

        try:
            result = score_detection(detection, spec)
            rows.append(_flatten_result(detection, result))
            print(
                f"  {len(detection.detections)} tiles detected, "
                f"mean error {result.mean_error_mm:.3f} mm"
            )
        except ValueError as err:
            print(f"  scoring failed: {err}")
            rows.append({
                "image": img_path.name,
                "model": args.model,
                "prompt_version": args.prompt_version,
                "n_detected": len(detection.detections),
                "mean_error_mm": None,
                "error_message": str(err),
            })

    _write_results_csv(args.out / "results.csv", rows)
    print(f"\nresults written to {args.out / 'results.csv'}")


def _dump_detection(out_dir: Path, detection: ImageDetection) -> None:
    stem = Path(detection.image_path).stem
    with (out_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(detection), f, indent=2)


def _flatten_result(d: ImageDetection, r: ScoringResult) -> dict:
    return {
        "image": Path(d.image_path).name,
        "model": d.model,
        "prompt_version": d.prompt_version,
        "n_detected": len(d.detections),
        "mean_error_mm": round(r.mean_error_mm, 4),
        "homography_rmse_px": round(r.homography_rmse_px, 3),
        "latency_seconds": round(d.latency_seconds, 2),
    }


def _write_results_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
