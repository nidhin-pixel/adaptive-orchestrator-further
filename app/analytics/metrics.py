"""
Analytics and metrics collector for Adaptive LLM Orchestrator.
Tracks requests, success/failure counts, rates, latencies, tokens, and per-model statistics.
Thread-safe in-memory design prepared for future Prometheus or persistent telemetry exports.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.providers.base import ExecutionResult


class ModelMetrics(BaseModel):
    """Metrics tracked per individual model."""
    requests: int = 0
    successful: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return round(self.total_latency_ms / self.requests, 2) if self.requests > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return round(self.successful / self.requests, 4) if self.requests > 0 else 0.0


class MetricsSnapshot(BaseModel):
    """Snapshot of overall system metrics at a point in time."""
    total_requests: int = Field(default=0, description="Total number of requests initiated")
    successful_requests: int = Field(default=0, description="Total requests that completed successfully")
    failed_requests: int = Field(default=0, description="Total requests that encountered errors")
    success_rate: float = Field(default=0.0, description="Ratio of successful requests (0.0 to 1.0)")
    avg_latency_ms: float = Field(default=0.0, description="Average latency across all requests in ms")
    total_prompt_tokens: int = Field(default=0, description="Sum of prompt tokens consumed")
    total_completion_tokens: int = Field(default=0, description="Sum of completion tokens generated")
    total_tokens: int = Field(default=0, description="Sum of all tokens processed")
    per_model: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-model breakdown")
    last_error: Optional[str] = Field(default=None, description="Most recent error message observed")


class MetricsCollector:
    """
    In-memory metrics collector for tracking LLM orchestrator performance and reliability.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_latency_ms = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._last_error: Optional[str] = None
        self._models: Dict[str, ModelMetrics] = {}

    def record_request(
        self,
        model_id: str,
        success: bool,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """
        Record a single execution request.

        Args:
            model_id: Model used.
            success: Whether request succeeded.
            latency_ms: Execution duration in milliseconds.
            prompt_tokens: Tokens in input prompt.
            completion_tokens: Tokens in output completion.
            error: Error message if failed.
        """
        with self._lock:
            self._total_requests += 1
            self._total_latency_ms += max(0.0, float(latency_ms))
            self._total_prompt_tokens += max(0, int(prompt_tokens or 0))
            self._total_completion_tokens += max(0, int(completion_tokens or 0))

            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1
                if error:
                    self._last_error = str(error)

            # Per-model metrics
            if model_id not in self._models:
                self._models[model_id] = ModelMetrics()

            m = self._models[model_id]
            m.requests += 1
            m.total_latency_ms += max(0.0, float(latency_ms))
            m.prompt_tokens += max(0, int(prompt_tokens or 0))
            m.completion_tokens += max(0, int(completion_tokens or 0))
            m.total_tokens += max(0, int(prompt_tokens or 0)) + max(0, int(completion_tokens or 0))

            if success:
                m.successful += 1
            else:
                m.failed += 1

    def record_execution(self, result: ExecutionResult) -> None:
        """
        Convenience method to record metrics directly from an ExecutionResult object.
        """
        self.record_request(
            model_id=result.model_id,
            success=result.success,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            error=result.error,
        )

    def get_metrics(self) -> MetricsSnapshot:
        """
        Return a thread-safe snapshot of all collected metrics.
        """
        with self._lock:
            avg_lat = (
                round(self._total_latency_ms / self._total_requests, 2)
                if self._total_requests > 0
                else 0.0
            )
            rate = (
                round(self._successful_requests / self._total_requests, 4)
                if self._total_requests > 0
                else 0.0
            )
            total_toks = self._total_prompt_tokens + self._total_completion_tokens

            per_model_data: Dict[str, Dict[str, Any]] = {}
            for mid, m in self._models.items():
                per_model_data[mid] = {
                    "requests": m.requests,
                    "successful": m.successful,
                    "failed": m.failed,
                    "success_rate": m.success_rate,
                    "avg_latency_ms": m.avg_latency_ms,
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "total_tokens": m.total_tokens,
                }

            return MetricsSnapshot(
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                success_rate=rate,
                avg_latency_ms=avg_lat,
                total_prompt_tokens=self._total_prompt_tokens,
                total_completion_tokens=self._total_completion_tokens,
                total_tokens=total_toks,
                per_model=per_model_data,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """Reset all counters and model statistics."""
        with self._lock:
            self._total_requests = 0
            self._successful_requests = 0
            self._failed_requests = 0
            self._total_latency_ms = 0.0
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._last_error = None
            self._models.clear()
