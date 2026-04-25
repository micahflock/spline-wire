from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Protocol

from validation.prompts import prompt_for_strip
from validation.schemas import DetectedTile, ImageDetection


class Extractor(Protocol):
    def extract(self, image_path: Path) -> ImageDetection: ...


class ClaudeExtractor:
    def __init__(
        self,
        client,
        model: str = "claude-opus-4-7",
        prompt_version: str = "v2",
    ) -> None:
        self.client = client
        self.model = model
        self.prompt_version = prompt_version

    def extract(self, image_path: Path) -> ImageDetection:
        prompt = prompt_for_strip()
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        media_type = _media_type_for(image_path)

        t0 = time.monotonic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                    ],
                }
            ],
        )
        latency = time.monotonic() - t0

        raw = response.content[0].text
        detections = parse_extraction_json(raw)
        return ImageDetection(
            image_path=str(image_path),
            model=self.model,
            prompt_version=self.prompt_version,
            detections=detections,
            raw_response=raw,
            latency_seconds=latency,
        )


def parse_extraction_json(raw: str) -> list[DetectedTile]:
    cleaned = _strip_code_fences(raw).strip()
    data = json.loads(cleaned)
    return [
        DetectedTile(
            tile_id=item["tile_id"],
            circle_xy_px=tuple(item["circle_xy_px"]),
            glyph_xy_px=(
                tuple(item["glyph_xy_px"])
                if item.get("glyph_xy_px") is not None
                else None
            ),
            confidence=float(item["confidence"]),
        )
        for item in data
    ]


def _strip_code_fences(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1)
    return raw


def _media_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".heic": "image/heic",
    }.get(ext, "image/jpeg")
