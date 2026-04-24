import json
from pathlib import Path
from unittest.mock import MagicMock

from validation.extraction import ClaudeExtractor, parse_extraction_json
from validation.schemas import DetectedTile


def test_parse_extraction_json_returns_tile_list():
    raw = json.dumps([
        {"tile_id": "0", "circle_xy_px": [120.5, 200.0],
         "glyph_xy_px": [160.0, 200.0], "confidence": 0.92},
        {"tile_id": "1", "circle_xy_px": [220.0, 200.0],
         "glyph_xy_px": [260.0, 200.0], "confidence": 0.88},
    ])
    tiles = parse_extraction_json(raw)
    assert len(tiles) == 2
    assert tiles[0].tile_id == "0"
    assert tiles[0].circle_xy_px == (120.5, 200.0)
    assert tiles[1].confidence == 0.88


def test_parse_extraction_json_strips_code_fences():
    raw = "```json\n" + json.dumps([
        {"tile_id": "0", "circle_xy_px": [1, 2],
         "glyph_xy_px": None, "confidence": 1.0},
    ]) + "\n```"
    tiles = parse_extraction_json(raw)
    assert tiles[0].glyph_xy_px is None


def test_claude_extractor_builds_request_with_image_and_prompt(tmp_path: Path):
    # prepare a 1x1 PNG
    from PIL import Image
    img_path = tmp_path / "test.png"
    Image.new("RGB", (1, 1), (255, 255, 255)).save(img_path)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps([
        {"tile_id": "0", "circle_xy_px": [0, 0],
         "glyph_xy_px": None, "confidence": 1.0}
    ]))]
    mock_client.messages.create.return_value = mock_response

    extractor = ClaudeExtractor(
        client=mock_client, model="claude-opus-4-7", prompt_version="v1"
    )
    result = extractor.extract(img_path, set_name="A")

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert any(
        isinstance(block, dict) and block.get("type") == "image"
        for msg in kwargs["messages"]
        for block in (msg["content"] if isinstance(msg["content"], list) else [])
    )
    assert result.model == "claude-opus-4-7"
    assert result.prompt_version == "v1"
    assert result.set_name == "A"
    assert len(result.detections) == 1
    assert result.detections[0].tile_id == "0"
