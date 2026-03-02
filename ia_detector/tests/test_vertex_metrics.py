
import pytest
from ia_detector.vertex_metrics import aletheia_ai_score

def test_aletheia_ai_score_basic():
    # Test with a simple string
    text = "This is a simple test sentence."
    score = aletheia_ai_score({"response": text})
    assert isinstance(score, float)
    assert 0 <= score <= 100

def test_aletheia_ai_score_empty():
    # Test with empty input
    score = aletheia_ai_score({"response": ""})
    assert score == 0.0

def test_aletheia_ai_score_direct_string():
    # Test passing string directly (handling potential SDK quirk)
    text = "Direct string input."
    score = aletheia_ai_score(text)
    assert isinstance(score, float)
