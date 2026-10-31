"""
Unit tests for Provider adapters (Provider A & Provider B) and Base contracts.
Uses httpx.MockTransport for 100% offline, isolated network testing.
"""

import json
import pytest
import httpx

from app.providers.base import (
    ExecutionResult,
    ModelDescriptor,
    ModelProvider,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConfigError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.providers.provider_a import OpenAICompatibleProvider, ProviderA
from app.providers.provider_b import AnthropicProvider, ProviderB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gpt4_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        model_id="gpt-4o",
        provider="openai",
        tier="flagship",
        supported_capabilities=["chat", "tools", "vision"],
        context_window=128000,
        input_price_per_million=2.50,
        output_price_per_million=10.00,
        enabled=True,
    )


@pytest.fixture
def claude_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        model_id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        tier="flagship",
        supported_capabilities=["chat", "artifacts"],
        context_window=200000,
        input_price_per_million=3.00,
        output_price_per_million=15.00,
        enabled=True,
    )


# ---------------------------------------------------------------------------
# Contract & Base Tests
# ---------------------------------------------------------------------------

def test_model_descriptor_contract(gpt4_descriptor: ModelDescriptor):
    assert gpt4_descriptor.model_id == "gpt-4o"
    assert gpt4_descriptor.provider == "openai"
    assert gpt4_descriptor.enabled is True
    assert gpt4_descriptor.input_price_per_million == 2.50
    assert gpt4_descriptor.output_price_per_million == 10.00
    assert "chat" in gpt4_descriptor.supported_capabilities


def test_model_descriptor_allows_extra_fields():
    model = ModelDescriptor(
        model_id="test-model",
        provider="test-provider",
        custom_orchestrator_meta="extra_value",
    )
    assert model.model_id == "test-model"
    assert getattr(model, "custom_orchestrator_meta", None) == "extra_value"


def test_execution_result_helpers():
    success = ExecutionResult.create_success(
        content="Hello, world!",
        model_id="gpt-4o",
        latency_ms=123.456,
        prompt_tokens=10,
        completion_tokens=5,
        raw_response={"test": "data"},
    )
    assert success.success is True
    assert success.content == "Hello, world!"
    assert success.model_id == "gpt-4o"
    assert success.error is None
    assert success.prompt_tokens == 10
    assert success.completion_tokens == 5
    assert success.latency_ms == 123.46

    failure = ExecutionResult.create_failure(
        model_id="gpt-4o",
        error="Network error",
        latency_ms=50.0,
    )
    assert failure.success is False
    assert failure.content == ""
    assert failure.error == "Network error"
    assert failure.latency_ms == 50.0
    assert failure.prompt_tokens == 0
    assert failure.completion_tokens == 0


# ---------------------------------------------------------------------------
# Provider A (OpenAI Compatible) Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provider_a_successful_call(gpt4_descriptor: ModelDescriptor):
    """Test successful completion with full response parsing and token tracking."""
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        data = json.loads(request.content.decode("utf-8"))
        assert data["model"] == "gpt-4o"
        assert data["messages"][0]["role"] == "system"
        assert data["messages"][0]["content"] == "You are a helpful assistant"
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"] == "Explain recursion"
        assert data["temperature"] == 0.2
        assert data["max_tokens"] == 150

        response_body = {
            "id": "chatcmpl-test12345",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Recursion is when a function calls itself.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 14,
                "completion_tokens": 8,
                "total_tokens": 22,
            },
        }
        return httpx.Response(200, json=response_body)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        api_key="test-api-key",
        base_url="https://api.openai.com/v1",
        client=mock_client,
    )

    context = {
        "system_prompt": "You are a helpful assistant",
        "temperature": 0.2,
        "max_tokens": 150,
    }

    result = await provider.call(
        prompt="Explain recursion",
        model=gpt4_descriptor,
        context=context,
    )

    assert result.success is True
    assert result.content == "Recursion is when a function calls itself."
    assert result.model_id == "gpt-4o"
    assert result.prompt_tokens == 14
    assert result.completion_tokens == 8
    assert result.latency_ms >= 0.0
    assert result.error is None
    assert result.raw_response is not None
    assert result.raw_response["id"] == "chatcmpl-test12345"
    assert len(recorded_requests) == 1
    assert "Bearer test-api-key" in recorded_requests[0].headers["Authorization"]


@pytest.mark.asyncio
async def test_provider_a_missing_api_key(gpt4_descriptor: ModelDescriptor, monkeypatch):
    """Test safe handling when API key is not configured."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = OpenAICompatibleProvider(api_key=None)
    result = await provider.call(prompt="Hello", model=gpt4_descriptor)

    assert result.success is False
    assert "API key not configured" in (result.error or "")
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


@pytest.mark.asyncio
async def test_provider_a_disabled_model(gpt4_descriptor: ModelDescriptor):
    """Test that disabled model descriptor is rejected safely."""
    gpt4_descriptor.enabled = False
    provider = OpenAICompatibleProvider(api_key="sk-test")
    result = await provider.call(prompt="Hello", model=gpt4_descriptor)

    assert result.success is False
    assert "disabled" in (result.error or "")


@pytest.mark.asyncio
async def test_provider_a_http_error(gpt4_descriptor: ModelDescriptor):
    """Test handling of HTTP 401 and 500 error responses."""
    def handler_401(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    client_401 = httpx.AsyncClient(transport=httpx.MockTransport(handler_401))
    provider = OpenAICompatibleProvider(api_key="bad-key", client=client_401)
    result = await provider.call(prompt="Hello", model=gpt4_descriptor)

    assert result.success is False
    assert "Authentication failed" in (result.error or "")
    assert "401" in (result.error or "")

    # Test 500 server error
    def handler_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client_500 = httpx.AsyncClient(transport=httpx.MockTransport(handler_500))
    provider_500 = OpenAICompatibleProvider(api_key="valid-key", client=client_500)
    result_500 = await provider_500.call(prompt="Hello", model=gpt4_descriptor)

    assert result_500.success is False
    assert "HTTP 500" in (result_500.error or "")


@pytest.mark.asyncio
async def test_provider_a_rate_limit_error(gpt4_descriptor: ModelDescriptor):
    """Test handling of HTTP 429 rate limit error."""
    def handler_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})

    client_429 = httpx.AsyncClient(transport=httpx.MockTransport(handler_429))
    provider = OpenAICompatibleProvider(api_key="sk-key", client=client_429)
    result = await provider.call(prompt="Hello", model=gpt4_descriptor)

    assert result.success is False
    assert "Rate limit exceeded" in (result.error or "")


@pytest.mark.asyncio
async def test_provider_a_timeout_handling(gpt4_descriptor: ModelDescriptor):
    """Test handling of request timeouts without crashing."""
    def handler_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read operation timed out")

    client_timeout = httpx.AsyncClient(transport=httpx.MockTransport(handler_timeout))
    provider = OpenAICompatibleProvider(api_key="sk-key", timeout=5.0, client=client_timeout)
    result = await provider.call(prompt="Hello", model=gpt4_descriptor)

    assert result.success is False
    assert "timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_provider_a_malformed_response(gpt4_descriptor: ModelDescriptor):
    """Test response missing choices or invalid JSON."""
    def handler_empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_empty))
    provider = OpenAICompatibleProvider(api_key="sk-key", client=client)
    result = await provider.call(prompt="Hello", model=gpt4_descriptor)

    assert result.success is False
    assert "missing valid 'choices'" in (result.error or "")


@pytest.mark.asyncio
async def test_provider_a_raise_on_error(gpt4_descriptor: ModelDescriptor):
    """Test opt-in exception raising when raise_on_error=True."""
    def handler_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timeout")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_timeout))
    provider = OpenAICompatibleProvider(api_key="sk-key", client=client, raise_on_error=True)

    with pytest.raises(ProviderTimeoutError):
        await provider.call(prompt="Hello", model=gpt4_descriptor)


# ---------------------------------------------------------------------------
# Provider B (Anthropic Messages API) Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provider_b_successful_call(claude_descriptor: ModelDescriptor):
    """Test successful Anthropic Messages API completion."""
    recorded_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        data = json.loads(request.content.decode("utf-8"))
        assert data["model"] == "claude-3-5-sonnet-20241022"
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello Claude"
        assert data["system"] == "Be concise."
        assert data["max_tokens"] == 512

        response_body = {
            "id": "msg_01X9DYP",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! How can I help you today?",
                }
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 9,
            },
        }
        return httpx.Response(200, json=response_body)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(
        api_key="ant-api-key",
        client=mock_client,
    )

    context = {
        "system_prompt": "Be concise.",
        "max_tokens": 512,
    }

    result = await provider.call(
        prompt="Hello Claude",
        model=claude_descriptor,
        context=context,
    )

    assert result.success is True
    assert result.content == "Hello! How can I help you today?"
    assert result.model_id == "claude-3-5-sonnet-20241022"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 9
    assert result.latency_ms >= 0.0
    assert result.error is None
    assert len(recorded_requests) == 1
    assert recorded_requests[0].headers["x-api-key"] == "ant-api-key"
    assert recorded_requests[0].headers["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_provider_b_missing_key(claude_descriptor: ModelDescriptor, monkeypatch):
    """Test Anthropic provider without API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PROVIDER_B_API_KEY", raising=False)

    provider = AnthropicProvider(api_key=None)
    result = await provider.call(prompt="Hi", model=claude_descriptor)

    assert result.success is False
    assert "ANTHROPIC_API_KEY" in (result.error or "")


@pytest.mark.asyncio
async def test_provider_b_http_error(claude_descriptor: ModelDescriptor):
    """Test Anthropic provider HTTP error response."""
    def handler_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler_500))
    provider = AnthropicProvider(api_key="test-key", client=client)
    result = await provider.call(prompt="Hi", model=claude_descriptor)

    assert result.success is False
    assert "Anthropic API returned HTTP 500" in (result.error or "")
