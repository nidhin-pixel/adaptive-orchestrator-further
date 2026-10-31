"""
Cost tracking and calculation module for Adaptive LLM Orchestrator.
Accepts configurable token pricing per million tokens.
Never hardcodes arbitrary pricing; cleanly handles missing pricing.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.providers.base import ExecutionResult, ModelDescriptor


class ModelPricing(BaseModel):
    """Pricing configuration per million tokens for a model."""
    model_id: str
    input_price_per_million: Optional[float] = None
    output_price_per_million: Optional[float] = None


class CostBreakdown(BaseModel):
    """Calculated cost breakdown for an LLM call."""
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    input_price_per_million: Optional[float] = None
    output_price_per_million: Optional[float] = None
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None
    total_cost: Optional[float] = None
    pricing_available: bool = False


class CostSummary(BaseModel):
    """Aggregated cost summary across recorded calls."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    priced_requests: int = 0
    unpriced_requests: int = 0
    model_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_million: Optional[float],
    output_price_per_million: Optional[float],
    model_id: str = "unknown",
) -> CostBreakdown:
    """
    Calculate dollar cost given token counts and prices per million tokens.

    Args:
        prompt_tokens: Number of prompt/input tokens.
        completion_tokens: Number of completion/output tokens.
        input_price_per_million: Price per 1M prompt tokens (USD), or None if unpriced.
        output_price_per_million: Price per 1M completion tokens (USD), or None if unpriced.
        model_id: Model identifier for logging and reporting.

    Returns:
        CostBreakdown: Granular breakdown of input, output, and total costs.
                       If pricing is None, cost values are None and pricing_available is False.
    """
    prompt_tokens = max(0, int(prompt_tokens or 0))
    completion_tokens = max(0, int(completion_tokens or 0))

    has_input_price = input_price_per_million is not None and input_price_per_million >= 0
    has_output_price = output_price_per_million is not None and output_price_per_million >= 0

    if not has_input_price and not has_output_price:
        return CostBreakdown(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            input_cost=None,
            output_cost=None,
            total_cost=None,
            pricing_available=False,
        )

    input_cost = (
        (prompt_tokens / 1_000_000.0) * input_price_per_million
        if has_input_price
        else 0.0
    )
    output_cost = (
        (completion_tokens / 1_000_000.0) * output_price_per_million
        if has_output_price
        else 0.0
    )

    total_cost = (input_cost or 0.0) + (output_cost or 0.0)

    return CostBreakdown(
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        input_cost=round(input_cost, 6) if has_input_price else None,
        output_cost=round(output_cost, 6) if has_output_price else None,
        total_cost=round(total_cost, 6),
        pricing_available=True,
    )


class CostTracker:
    """
    Thread-safe cost tracking engine.
    Maintains a model pricing registry and tracks aggregate usage costs.
    """

    def __init__(self, pricing_registry: Optional[Dict[str, ModelPricing]] = None) -> None:
        self._pricing: Dict[str, ModelPricing] = pricing_registry or {}
        self._lock = threading.Lock()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0
        self._priced_requests = 0
        self._unpriced_requests = 0
        self._model_stats: Dict[str, Dict[str, Any]] = {}

    def register_pricing(
        self,
        model_id: str,
        input_price_per_million: Optional[float],
        output_price_per_million: Optional[float],
    ) -> None:
        """Register or update pricing for a given model ID."""
        with self._lock:
            self._pricing[model_id] = ModelPricing(
                model_id=model_id,
                input_price_per_million=input_price_per_million,
                output_price_per_million=output_price_per_million,
            )

    def register_model_descriptor(self, model: ModelDescriptor) -> None:
        """Register pricing directly from a ModelDescriptor."""
        self.register_pricing(
            model_id=model.model_id,
            input_price_per_million=model.input_price_per_million,
            output_price_per_million=model.output_price_per_million,
        )

    def get_pricing(self, model_id: str) -> Optional[ModelPricing]:
        """Retrieve registered pricing for a model."""
        with self._lock:
            return self._pricing.get(model_id)

    def record_usage(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        input_price_per_million: Optional[float] = None,
        output_price_per_million: Optional[float] = None,
    ) -> CostBreakdown:
        """
        Record token usage, calculate cost, and update aggregate summaries.

        If explicit prices are not provided, uses registered model pricing.
        """
        # Determine applicable pricing
        if input_price_per_million is None and output_price_per_million is None:
            reg = self.get_pricing(model_id)
            if reg:
                input_price_per_million = reg.input_price_per_million
                output_price_per_million = reg.output_price_per_million

        breakdown = calculate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            model_id=model_id,
        )

        with self._lock:
            self._total_prompt_tokens += breakdown.prompt_tokens
            self._total_completion_tokens += breakdown.completion_tokens

            if model_id not in self._model_stats:
                self._model_stats[model_id] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "request_count": 0,
                }

            stats = self._model_stats[model_id]
            stats["prompt_tokens"] += breakdown.prompt_tokens
            stats["completion_tokens"] += breakdown.completion_tokens
            stats["total_tokens"] += breakdown.prompt_tokens + breakdown.completion_tokens
            stats["request_count"] += 1

            if breakdown.pricing_available and breakdown.total_cost is not None:
                self._total_cost += breakdown.total_cost
                stats["cost"] = round(stats["cost"] + breakdown.total_cost, 6)
                self._priced_requests += 1
            else:
                self._unpriced_requests += 1

        return breakdown

    def record_execution(
        self,
        result: ExecutionResult,
        model: Optional[ModelDescriptor] = None,
    ) -> CostBreakdown:
        """
        Record cost directly from an ExecutionResult and optional ModelDescriptor.
        """
        in_price = model.input_price_per_million if model else None
        out_price = model.output_price_per_million if model else None

        return self.record_usage(
            model_id=result.model_id,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            input_price_per_million=in_price,
            output_price_per_million=out_price,
        )

    def get_summary(self) -> CostSummary:
        """Return the current aggregated cost summary."""
        with self._lock:
            total_tokens = self._total_prompt_tokens + self._total_completion_tokens
            return CostSummary(
                total_prompt_tokens=self._total_prompt_tokens,
                total_completion_tokens=self._total_completion_tokens,
                total_tokens=total_tokens,
                total_cost=round(self._total_cost, 6),
                priced_requests=self._priced_requests,
                unpriced_requests=self._unpriced_requests,
                model_breakdown={k: dict(v) for k, v in self._model_stats.items()},
            )

    def reset(self) -> None:
        """Reset all tracked usage while keeping registered pricing."""
        with self._lock:
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_cost = 0.0
            self._priced_requests = 0
            self._unpriced_requests = 0
            self._model_stats.clear()
