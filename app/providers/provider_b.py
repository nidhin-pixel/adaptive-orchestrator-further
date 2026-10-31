"""
Anthropic Claude Messages API provider adapter using httpx.
Serves as Provider B adapter for multi-provider routing in the Adaptive Orchestrator.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
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


class AnthropicProvider(ModelProvider):
    """
    Asynchronous provider adapter for Anthropic Claude Messages API.
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com"
    DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
    DEFAULT_TIMEOUT = 30.0
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        anthropic_version: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        raise_on_error: bool = False,
    ) -> None:
        """
        Initialize the Anthropic provider adapter.

        Args:
            api_key: Anthropic API key. Reads from ANTHROPIC_API_KEY or PROVIDER_B_API_KEY if omitted.
            base_url: Base endpoint URL. Reads from ANTHROPIC_BASE_URL or PROVIDER_B_BASE_URL if omitted.
            timeout: Request timeout in seconds. Reads from PROVIDER_B_TIMEOUT if omitted.
            anthropic_version: Anthropic API version header.
            client: Optional pre-configured httpx.AsyncClient for dependency injection (e.g. testing).
            raise_on_error: If True, raises exceptions instead of returning a failed ExecutionResult.
        """
        self.api_key = (
            api_key
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("PROVIDER_B_API_KEY")
        )

        env_base = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("PROVIDER_B_BASE_URL")
        self.base_url = (base_url or env_base or self.DEFAULT_BASE_URL).rstrip("/")

        env_timeout = os.getenv("PROVIDER_B_TIMEOUT")
        if timeout is not None:
            self.timeout = float(timeout)
        elif env_timeout:
            try:
                self.timeout = float(env_timeout)
            except ValueError:
                self.timeout = self.DEFAULT_TIMEOUT
        else:
            self.timeout = self.DEFAULT_TIMEOUT

        self.anthropic_version = anthropic_version or self.DEFAULT_ANTHROPIC_VERSION
        self._injected_client = client
        self.raise_on_error = raise_on_error

    def _build_endpoint_url(self) -> str:
        """Construct the Anthropic messages endpoint URL."""
        if self.base_url.endswith("/v1/messages"):
            return self.base_url
        return f"{self.base_url}/v1/messages"

    def _build_messages(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Construct messages list for Anthropic Messages API."""
        messages: List[Dict[str, str]] = []
        ctx = context or {}

        history = ctx.get("messages") or ctx.get("history")
        if isinstance(history, list):
            for turn in history:
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    role = str(turn["role"]).lower()
                    # Anthropic supports 'user' and 'assistant' in messages
                    if role in ("user", "assistant"):
                        messages.append({"role": role, "content": str(turn["content"])})

        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_payload(
        self,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct Anthropic Messages payload."""
        ctx = context or {}
        messages = self._build_messages(prompt, ctx)

        max_tokens = int(ctx.get("max_tokens") or self.DEFAULT_MAX_TOKENS)

        payload: Dict[str, Any] = {
            "model": model.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": float(ctx.get("temperature", 0.7)),
        }

        # System prompt is a top-level parameter in Anthropic Messages API
        system_prompt = ctx.get("system_prompt") or ctx.get("system_instruction")
        if system_prompt:
            payload["system"] = str(system_prompt)

        return payload

    async def call(
        self,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute an asynchronous completion call to the Anthropic Messages API.

        Returns:
            ExecutionResult matching the orchestrator specification.
        """
        start_time = time.perf_counter()

        # Step 1: Input & Configuration Validation
        try:
            self.validate_inputs(prompt, model)
            if not self.api_key:
                raise ProviderConfigError("API key not configured: ANTHROPIC_API_KEY environment variable is required")
        except ProviderConfigError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            if self.raise_on_error:
                raise
            return ExecutionResult.create_failure(
                model_id=getattr(model, "model_id", "unknown"),
                error=str(e),
                latency_ms=latency_ms,
            )

        endpoint = self._build_endpoint_url()
        payload = self._build_payload(prompt, model, context)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Step 2: HTTP Request Execution
        try:
            if self._injected_client is not None:
                response = await self._injected_client.post(
                    endpoint, json=payload, headers=headers, timeout=self.timeout
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        endpoint, json=payload, headers=headers
                    )

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Step 3: Handle Status Codes
            if response.status_code in (401, 403):
                raise ProviderAuthError(
                    f"Anthropic authentication failed (HTTP {response.status_code}): {response.text}",
                    status_code=response.status_code,
                )

            if response.status_code == 429:
                raise ProviderRateLimitError(
                    f"Anthropic rate limit exceeded (HTTP 429): {response.text}",
                    status_code=429,
                )

            if response.status_code >= 400:
                raise ProviderAPIError(
                    f"Anthropic API returned HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )

            # Step 4: Parse Successful Response
            try:
                data = response.json()
            except Exception as json_err:
                raise ProviderAPIError(f"Malformed JSON response from Anthropic: {json_err}")

            if not isinstance(data, dict):
                raise ProviderAPIError("Anthropic response is not a valid JSON dictionary")

            # Anthropic response: content is a list of blocks, e.g. [{"type": "text", "text": "..."}]
            content_blocks = data.get("content", [])
            extracted_text = ""
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        extracted_text += block.get("text", "")

            # Usage in Anthropic: {"input_tokens": X, "output_tokens": Y}
            usage = data.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

            return ExecutionResult.create_success(
                content=extracted_text,
                model_id=model.model_id,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                raw_response=data,
            )

        except httpx.TimeoutException as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Request timed out after {self.timeout}s: {e}"
            if self.raise_on_error:
                raise ProviderTimeoutError(err_msg) from e
            return ExecutionResult.create_failure(
                model_id=model.model_id,
                error=err_msg,
                latency_ms=latency_ms,
            )

        except httpx.RequestError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Anthropic network error: {e}"
            if self.raise_on_error:
                raise ProviderAPIError(err_msg) from e
            return ExecutionResult.create_failure(
                model_id=model.model_id,
                error=err_msg,
                latency_ms=latency_ms,
            )

        except ProviderError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            if self.raise_on_error:
                raise
            return ExecutionResult.create_failure(
                model_id=model.model_id,
                error=str(e),
                latency_ms=latency_ms,
                raw_response=getattr(e, "raw_response", None),
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Unexpected Anthropic provider error: {str(e)}"
            if self.raise_on_error:
                raise ProviderError(err_msg) from e
            return ExecutionResult.create_failure(
                model_id=model.model_id,
                error=err_msg,
                latency_ms=latency_ms,
            )


# Compatibility alias
ProviderB = AnthropicProvider
