import pytest
from unittest.mock import patch, MagicMock
from pipeline.analyze import parse_vision_response, ACTION_MAP
from pipeline.models import PagePlan


def test_parse_vision_response_with_character():
    raw = '{"character": true, "activity": "hopping", "description": "rabbit hopping through garden"}'
    result = parse_vision_response("page01.jpg", raw)
    assert result.character is True
    assert result.action == "jump"   # "hopping" maps to "jump"
    assert "rabbit" in result.description


def test_parse_vision_response_no_character():
    raw = '{"character": false, "activity": null, "description": "garden with flowers"}'
    result = parse_vision_response("page01.jpg", raw)
    assert result.character is False
    assert result.action is None


def test_parse_vision_response_handles_bad_json():
    raw = "No character detected in this image."
    result = parse_vision_response("page01.jpg", raw)
    assert result.character is False
    assert result.action is None


def test_action_map_covers_common_activities():
    assert "walk" in ACTION_MAP.values()
    assert "jump" in ACTION_MAP.values()
    assert "wave" in ACTION_MAP.values()
