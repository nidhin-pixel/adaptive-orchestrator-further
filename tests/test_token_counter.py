"""
Unit tests for token counter utility.
Tests both tiktoken tokenizer and the documented fallback heuristic under offline conditions.
"""

import pytest
from app.utils.token_counter import (
    count_prompt_and_completion_tokens,
    count_tokens,
    estimate_fallback_tokens,
    is_tiktoken_available,
)
import app.utils.token_counter as tc_module


def test_token_counter_empty_and_invalid_inputs():
    """Verify safe handling of empty, whitespace, None, and non-string types."""
    assert count_tokens(None) == 0
    assert count_tokens("") == 0
    assert count_tokens("   ") == 0
    assert count_tokens(12345) == 0  # type: ignore
    assert count_tokens({"key": "val"}) == 0  # type: ignore


def test_token_counter_basic_text():
    """Verify token counting returns positive integer for normal English sentence."""
    text = "Hello world! This is a test prompt for the adaptive orchestrator."
    count = count_tokens(text)
    assert isinstance(count, int)
    assert count > 5


def test_token_counter_prompt_and_completion():
    """Verify prompt and completion counting helper."""
    prompt = "What is the capital of France?"
    completion = "The capital of France is Paris."

    p_tokens, c_tokens = count_prompt_and_completion_tokens(prompt, completion)
    assert p_tokens > 0
    assert c_tokens > 0
    assert isinstance(p_tokens, int)
    assert isinstance(c_tokens, int)


def test_fallback_estimator_direct():
    """Verify documented fallback heuristic specifically."""
    assert estimate_fallback_tokens("") == 0
    assert estimate_fallback_tokens(None) == 0  # type: ignore
    assert estimate_fallback_tokens("   ") == 0

    # Short single word: length 4 -> ceil(4/4) = 1 token
    assert estimate_fallback_tokens("chat") >= 1

    # Text with 40 characters
    text_40 = "1234567890123456789012345678901234567890"
    estimated = estimate_fallback_tokens(text_40)
    assert estimated >= 10


def test_token_counter_with_fallback_forced(monkeypatch):
    """Test counting when tiktoken is unavailable or disabled."""
    monkeypatch.setattr(tc_module, "_TIKTOKEN_AVAILABLE", False)

    text = "This should use the fallback heuristic estimate."
    count = tc_module.count_tokens(text)
    assert count > 0
    assert count == estimate_fallback_tokens(text)


def test_token_counter_unknown_model_fallback():
    """Test passing an unusual or unrecognized model name falls back cleanly."""
    text = "Testing unfamiliar model architecture."
    count = count_tokens(text, model_name="unknown-custom-model-v9")
    assert count > 0
    assert isinstance(count, int)
