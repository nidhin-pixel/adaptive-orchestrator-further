"""
OpenAI-compatible LLM Provider implementation using httpx.
Supports any OpenAI-compatible endpoint (OpenAI, DeepSeek, Groq, Ollama, vLLM, etc.)
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


class OpenAICompatibleProvider(ModelProvider):
    """
    Asynchronous provider adapter for OpenAI-compatible chat completion APIs.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        raise_on_error: bool = False,
    ) -> None:
        """
        Initialize the provider adapter.

        Args:
            api_key: API key. If omitted, reads from LLM_API_KEY (or OPENAI_API_KEY).
            base_url: Base endpoint URL. If omitted, reads from LLM_BASE_URL.
            timeout: Request timeout in seconds. If omitted, reads from LLM_TIMEOUT.
            client: Optional pre-configured httpx.AsyncClient for dependency injection (e.g. testing).
            raise_on_error: If True, raises exceptions instead of returning a failed ExecutionResult.
        """
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        env_base_url = os.getenv("LLM_BASE_URL")
        self.base_url = (base_url or env_base_url or self.DEFAULT_BASE_URL).rstrip("/")

        env_timeout = os.getenv("LLM_TIMEOUT")
        if timeout is not None:
            self.timeout = float(timeout)
        elif env_timeout:
            try:
                self.timeout = float(env_timeout)
            except ValueError:
                self.timeout = self.DEFAULT_TIMEOUT
        else:
            self.timeout = self.DEFAULT_TIMEOUT

        self._injected_client = client
        self.raise_on_error = raise_on_error

    def _build_endpoint_url(self) -> str:
        """Construct the chat completion URL."""
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _build_messages(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Construct the messages list from prompt and context."""
        messages: List[Dict[str, str]] = []
        ctx = context or {}

        # System prompt / instructions
        system_prompt = ctx.get("system_prompt") or ctx.get("system_instruction")
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})

        # History / prior turns if provided
        history = ctx.get("messages") or ctx.get("history")
        if isinstance(history, list):
            for turn in history:
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    messages.append({"role": str(turn["role"]), "content": str(turn["content"])})

        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_payload(
        self,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct the request payload for chat completions."""
        ctx = context or {}
        messages = self._build_messages(prompt, ctx)

        payload: Dict[str, Any] = {
            "model": model.model_id,
            "messages": messages,
            "temperature": ctx.get("temperature", 0.7),
        }

        if "max_tokens" in ctx and ctx["max_tokens"] is not None:
            payload["max_tokens"] = int(ctx["max_tokens"])

        if "top_p" in ctx and ctx["top_p"] is not None:
            payload["top_p"] = float(ctx["top_p"])

        if "stop" in ctx and ctx["stop"] is not None:
            payload["stop"] = ctx["stop"]

        return payload

    async def call(
        self,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute an asynchronous completion call to the OpenAI-compatible endpoint.

        Returns:
            ExecutionResult matching the orchestrator specification.
        """
        start_time = time.perf_counter()

        # Step 1: Input & Configuration Validation
        try:
            self.validate_inputs(prompt, model)
            if not self.api_key:
                raise ProviderConfigError("API key not configured: LLM_API_KEY environment variable is required")
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
            "Authorization": f"Bearer {self.api_key}",
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

            # Step 3: Handle HTTP Status Codes
            if response.status_code == 401 or response.status_code == 403:
                raise ProviderAuthError(
                    f"Authentication failed (HTTP {response.status_code}): {response.text}",
                    status_code=response.status_code,
                )

            if response.status_code == 429:
                raise ProviderRateLimitError(
                    f"Rate limit exceeded (HTTP 429): {response.text}",
                    status_code=429,
                )

            if response.status_code >= 400:
                raise ProviderAPIError(
                    f"Upstream provider returned HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )

            # Step 4: Parse Successful Response
            try:
                data = response.json()
            except Exception as json_err:
                raise ProviderAPIError(f"Malformed JSON response from provider: {json_err}")

            if not isinstance(data, dict):
                raise ProviderAPIError("Provider response payload is not a valid JSON object")

            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                raise ProviderAPIError("Provider response missing valid 'choices' list")

            first_choice = choices[0]
            if not isinstance(first_choice, dict) or "message" not in first_choice:
                raise ProviderAPIError("Provider response missing 'message' in first choice")

            message = first_choice.get("message", {})
            content = message.get("content") or ""

            # Extract token counts
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            return ExecutionResult.create_success(
                content=content,
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
            err_msg = f"Network communication error: {e}"
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
            err_msg = f"Unexpected provider error: {str(e)}"
            if self.raise_on_error:
                raise ProviderError(err_msg) from e
            return ExecutionResult.create_failure(
                model_id=model.model_id,
                error=err_msg,
                latency_ms=latency_ms,
            )


# Compatibility alias
ProviderA = OpenAICompatibleProvider
