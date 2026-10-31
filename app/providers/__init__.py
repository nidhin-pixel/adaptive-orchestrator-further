"""
LLM Providers package for Adaptive LLM Orchestrator.
Exposes ModelDescriptor, ExecutionResult, base interfaces, and provider implementations.
"""

from app.providers.base import (
    BaseProvider,
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

__all__ = [
    "ModelDescriptor",
    "ExecutionResult",
    "ModelProvider",
    "BaseProvider",
    "ProviderError",
    "ProviderConfigError",
    "ProviderAuthError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    "ProviderAPIError",
    "OpenAICompatibleProvider",
    "ProviderA",
    "AnthropicProvider",
    "ProviderB",
]
