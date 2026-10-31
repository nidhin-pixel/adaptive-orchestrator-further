"""
Unit tests for Analytics modules: CostTracker and MetricsCollector.
Verifies cost arithmetic, missing pricing tolerance, and telemetry counters.
"""

import pytest

from app.analytics.cost_tracker import (
    CostBreakdown,
    CostSummary,
    CostTracker,
    calculate_cost,
)
from app.analytics.metrics import (
    MetricsCollector,
    MetricsSnapshot,
)
from app.providers.base import ExecutionResult, ModelDescriptor


# ---------------------------------------------------------------------------
# Cost Calculation & Tracker Tests
# ---------------------------------------------------------------------------

def test_calculate_cost_standard():
    """Verify standard dollar cost calculation per million tokens."""
    # 1,000,000 prompt tokens @ $2.50/M = $2.50
    # 500,000 completion tokens @ $10.00/M = $5.00
    # Total = $7.50
    breakdown = calculate_cost(
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        input_price_per_million=2.50,
        output_price_per_million=10.00,
        model_id="gpt-4o",
    )
    assert breakdown.pricing_available is True
    assert breakdown.input_cost == 2.50
    assert breakdown.output_cost == 5.00
    assert breakdown.total_cost == 7.50
    assert breakdown.prompt_tokens == 1_000_000
    assert breakdown.completion_tokens == 500_000


def test_calculate_cost_missing_pricing():
    """Verify safe handling when pricing is None or missing (do not invent prices)."""
    breakdown = calculate_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        input_price_per_million=None,
        output_price_per_million=None,
        model_id="unpriced-model",
    )
    assert breakdown.pricing_available is False
    assert breakdown.input_cost is None
    assert breakdown.output_cost is None
    assert breakdown.total_cost is None


def test_calculate_cost_partial_pricing():
    """Verify handling when only input price is known."""
    breakdown = calculate_cost(
        prompt_tokens=1_000_000,
        completion_tokens=500,
        input_price_per_million=1.00,
        output_price_per_million=None,
        model_id="partial-price-model",
    )
    assert breakdown.pricing_available is True
    assert breakdown.input_cost == 1.00
    assert breakdown.output_cost is None
    assert breakdown.total_cost == 1.00


def test_cost_tracker_lifecycle():
    """Verify CostTracker registering models, recording usage, and summarizing."""
    tracker = CostTracker()

    model = ModelDescriptor(
        model_id="claude-3-5-sonnet",
        provider="anthropic",
        input_price_per_million=3.00,
        output_price_per_million=15.00,
    )
    tracker.register_model_descriptor(model)

    # Record 100,000 prompt tokens and 50,000 completion tokens
    # 100,000 / 1M * 3.00 = 0.30
    # 50,000 / 1M * 15.00 = 0.75
    # Total = 1.05
    breakdown = tracker.record_usage(
        model_id="claude-3-5-sonnet",
        prompt_tokens=100_000,
        completion_tokens=50_000,
    )
    assert breakdown.pricing_available is True
    assert breakdown.total_cost == 1.05

    # Record an unpriced model usage
    unpriced_breakdown = tracker.record_usage(
        model_id="unknown-model",
        prompt_tokens=10_000,
        completion_tokens=5_000,
    )
    assert unpriced_breakdown.pricing_available is False

    summary = tracker.get_summary()
    assert summary.total_prompt_tokens == 110_000
    assert summary.total_completion_tokens == 55_000
    assert summary.total_tokens == 165_000
    assert summary.total_cost == 1.05
    assert summary.priced_requests == 1
    assert summary.unpriced_requests == 1
    assert "claude-3-5-sonnet" in summary.model_breakdown
    assert summary.model_breakdown["claude-3-5-sonnet"]["cost"] == 1.05

    # Reset
    tracker.reset()
    clean_summary = tracker.get_summary()
    assert clean_summary.total_tokens == 0
    assert clean_summary.total_cost == 0.0


def test_cost_tracker_record_execution():
    """Verify recording directly from an ExecutionResult object."""
    tracker = CostTracker()
    model = ModelDescriptor(
        model_id="gpt-4o",
        provider="openai",
        input_price_per_million=5.00,
        output_price_per_million=15.00,
    )

    result = ExecutionResult.create_success(
        content="Response text",
        model_id="gpt-4o",
        latency_ms=100.0,
        prompt_tokens=200_000,
        completion_tokens=100_000,
    )

    breakdown = tracker.record_execution(result, model=model)
    assert breakdown.pricing_available is True
    # 200k/1M * 5 = 1.0, 100k/1M * 15 = 1.5 -> total 2.5
    assert breakdown.total_cost == 2.50


# ---------------------------------------------------------------------------
# MetricsCollector Tests
# ---------------------------------------------------------------------------

def test_metrics_empty_state():
    """Verify metrics collector initial state."""
    collector = MetricsCollector()
    metrics = collector.get_metrics()

    assert metrics.total_requests == 0
    assert metrics.successful_requests == 0
    assert metrics.failed_requests == 0
    assert metrics.success_rate == 0.0
    assert metrics.avg_latency_ms == 0.0
    assert metrics.total_tokens == 0
    assert len(metrics.per_model) == 0


def test_metrics_record_requests():
    """Verify metrics calculation across multiple requests."""
    collector = MetricsCollector()

    # Request 1: Success on gpt-4o (100ms, 10 prompt, 20 completion)
    collector.record_request(
        model_id="gpt-4o",
        success=True,
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=20,
    )

    # Request 2: Success on gpt-4o (200ms, 20 prompt, 40 completion)
    collector.record_request(
        model_id="gpt-4o",
        success=True,
        latency_ms=200.0,
        prompt_tokens=20,
        completion_tokens=40,
    )

    # Request 3: Failure on claude-3-5 (300ms, 5 prompt, 0 completion)
    collector.record_request(
        model_id="claude-3-5",
        success=False,
        latency_ms=300.0,
        prompt_tokens=5,
        completion_tokens=0,
        error="API Rate limit",
    )

    snapshot = collector.get_metrics()
    assert snapshot.total_requests == 3
    assert snapshot.successful_requests == 2
    assert snapshot.failed_requests == 1
    # Success rate: 2/3 = 0.6667
    assert snapshot.success_rate == 0.6667
    # Avg latency: (100 + 200 + 300) / 3 = 200.0ms
    assert snapshot.avg_latency_ms == 200.0
    assert snapshot.total_prompt_tokens == 35
    assert snapshot.total_completion_tokens == 60
    assert snapshot.total_tokens == 95
    assert snapshot.last_error == "API Rate limit"

    # Per-model metrics
    assert "gpt-4o" in snapshot.per_model
    assert snapshot.per_model["gpt-4o"]["requests"] == 2
    assert snapshot.per_model["gpt-4o"]["successful"] == 2
    assert snapshot.per_model["gpt-4o"]["avg_latency_ms"] == 150.0

    assert "claude-3-5" in snapshot.per_model
    assert snapshot.per_model["claude-3-5"]["requests"] == 1
    assert snapshot.per_model["claude-3-5"]["failed"] == 1


def test_metrics_record_execution():
    """Verify recording metrics directly from ExecutionResult."""
    collector = MetricsCollector()
    exec_res = ExecutionResult.create_success(
        content="Success content",
        model_id="gpt-4o-mini",
        latency_ms=85.5,
        prompt_tokens=15,
        completion_tokens=25,
    )

    collector.record_execution(exec_res)
    snapshot = collector.get_metrics()

    assert snapshot.total_requests == 1
    assert snapshot.successful_requests == 1
    assert snapshot.total_tokens == 40
    assert snapshot.avg_latency_ms == 85.5


def test_metrics_reset():
    """Verify metrics reset clears all tracking."""
    collector = MetricsCollector()
    collector.record_request("model-a", True, 50.0, 10, 10)
    assert collector.get_metrics().total_requests == 1

    collector.reset()
    assert collector.get_metrics().total_requests == 0
    assert len(collector.get_metrics().per_model) == 0


def test_calculate_cost_zero_tokens():
    """Verify safe calculation when token counts are zero."""
    breakdown = calculate_cost(
        prompt_tokens=0,
        completion_tokens=0,
        input_price_per_million=2.50,
        output_price_per_million=10.00,
        model_id="gpt-4o",
    )
    assert breakdown.pricing_available is True
    assert breakdown.input_cost == 0.0
    assert breakdown.output_cost == 0.0
    assert breakdown.total_cost == 0.0


def test_concurrent_analytics_safety():
    """Verify thread-safety of CostTracker and MetricsCollector under concurrent load."""
    import concurrent.futures

    collector = MetricsCollector()
    tracker = CostTracker()
    tracker.register_pricing("gpt-4o", input_price_per_million=2.00, output_price_per_million=4.00)

    num_threads = 10
    requests_per_thread = 50

    def worker():
        for _ in range(requests_per_thread):
            collector.record_request(
                model_id="gpt-4o",
                success=True,
                latency_ms=10.0,
                prompt_tokens=100,
                completion_tokens=50,
            )
            tracker.record_usage(
                model_id="gpt-4o",
                prompt_tokens=100,
                completion_tokens=50,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        concurrent.futures.wait(futures)

    # Verification
    expected_total_requests = num_threads * requests_per_thread
    metrics_snapshot = collector.get_metrics()
    assert metrics_snapshot.total_requests == expected_total_requests
    assert metrics_snapshot.successful_requests == expected_total_requests
    assert metrics_snapshot.failed_requests == 0
    assert metrics_snapshot.total_prompt_tokens == expected_total_requests * 100
    assert metrics_snapshot.total_completion_tokens == expected_total_requests * 50

    cost_summary = tracker.get_summary()
    assert cost_summary.priced_requests == expected_total_requests
    assert cost_summary.total_prompt_tokens == expected_total_requests * 100
    assert cost_summary.total_completion_tokens == expected_total_requests * 50
    assert cost_summary.total_cost > 0.0

