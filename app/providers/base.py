"""
Base provider abstraction and data contracts for the Adaptive LLM Orchestrator.

Defines:
- ModelDescriptor: Metadata and pricing configuration for an LLM model.
- ExecutionResult: Standardized execution output contract returned to orchestrator.
- ModelProvider / BaseProvider: Abstract asynchronous interface for provider adapters.
- Provider exception hierarchy for granular error handling.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Data Contracts
# ---------------------------------------------------------------------------

class ModelDescriptor(BaseModel):
    """
    Metadata describing an individual LLM model, its capabilities, and pricing.
    Compatible with the main orchestrator specification.
    """
    model_config = ConfigDict(extra="allow")

    model_id: str = Field(..., description="Unique model identifier, e.g. 'gpt-4o' or 'claude-3-5-sonnet'")
    provider: str = Field(..., description="Provider name, e.g. 'openai', 'anthropic'")
    tier: str = Field(default="standard", description="Model tier, e.g. 'budget', 'standard', 'premium'")
    supported_capabilities: List[str] = Field(default_factory=list, description="Capabilities like ['chat', 'vision', 'tools']")
    context_window: int = Field(default=4096, description="Maximum context window in tokens")
    input_price_per_million: Optional[float] = Field(default=None, description="Input cost per 1M tokens in USD")
    output_price_per_million: Optional[float] = Field(default=None, description="Output cost per 1M tokens in USD")
    enabled: bool = Field(default=True, description="Whether the model is active and routable")


class ExecutionResult(BaseModel):
    """
    Standardized result contract returned by any provider after executing a prompt.
    Compatible with the main orchestrator specification.
    """
    model_config = ConfigDict(extra="allow")

    content: str = Field(default="", description="The generated response content text")
    model_id: str = Field(..., description="Model identifier used for the execution")
    success: bool = Field(..., description="Whether execution completed without failure")
    latency_ms: float = Field(..., description="End-to-end request latency in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if execution failed, else None")
    prompt_tokens: int = Field(default=0, description="Tokens consumed by prompt")
    completion_tokens: int = Field(default=0, description="Tokens consumed by completion")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Raw provider payload if available")

    @classmethod
    def create_success(
        cls,
        content: str,
        model_id: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        raw_response: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Helper to construct a successful ExecutionResult."""
        return cls(
            content=content,
            model_id=model_id,
            success=True,
            latency_ms=round(latency_ms, 2),
            error=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_response=raw_response,
        )

    @classmethod
    def create_failure(
        cls,
        model_id: str,
        error: str,
        latency_ms: float = 0.0,
        raw_response: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Helper to construct a failed ExecutionResult."""
        return cls(
            content="",
            model_id=model_id,
            success=False,
            latency_ms=round(latency_ms, 2),
            error=error,
            prompt_tokens=0,
            completion_tokens=0,
            raw_response=raw_response,
        )


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Base exception for all provider-related errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, raw_response: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.raw_response = raw_response


class ProviderConfigError(ProviderError):
    """Raised when provider configuration or credentials are missing/invalid."""
    pass


class ProviderAuthError(ProviderConfigError):
    """Raised when authentication against the provider fails (e.g. 401/403)."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when an upstream request times out."""
    pass


class ProviderRateLimitError(ProviderError):
    """Raised when the provider enforces rate limits (e.g. 429)."""
    pass


class ProviderAPIError(ProviderError):
    """Raised when the upstream provider returns an HTTP error or invalid payload."""
    pass


# ---------------------------------------------------------------------------
# Abstract Provider Interface
# ---------------------------------------------------------------------------

class ModelProvider(ABC):
    """
    Abstract base class for all LLM provider adapters.
    Subclasses must implement the async `call` method.
    """

    @abstractmethod
    async def call(
        self,
        prompt: str,
        model: ModelDescriptor,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute an LLM completion call asynchronously.

        Args:
            prompt: The user input text.
            model: Descriptor defining model identity and settings.
            context: Optional dictionary with extra configurations:
                     - system_prompt / system_instruction: str
                     - temperature: float (default 0.7)
                     - max_tokens: int
                     - messages / history: list of prior conversation turns

        Returns:
            ExecutionResult: Standardized execution result with latency, token usage, and status.
        """
        raise NotImplementedError

    def validate_inputs(self, prompt: str, model: ModelDescriptor) -> None:
        """
        Validate input arguments before sending to the provider.

        Raises:
            ProviderConfigError: If arguments or model descriptor are invalid or model disabled.
        """
        if not isinstance(model, ModelDescriptor):
            raise ProviderConfigError("Invalid model parameter: must be an instance of ModelDescriptor")

        if not model.enabled:
            raise ProviderConfigError(f"Model '{model.model_id}' is disabled in ModelDescriptor")

        if not isinstance(prompt, str):
            raise ProviderConfigError(f"Prompt must be a string, got {type(prompt).__name__}")


# Backward-compatibility alias
BaseProvider = ModelProvider
