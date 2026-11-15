"""Permanent regression coverage for benchmark timing and throughput validity."""

import math

import pytest

from edgecostbench.metrics import percentile, summarize_latencies


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_percentile_rejects_nonfinite_values(value):
    with pytest.raises(ValueError):
        percentile([1.0, value], 0.5)


@pytest.mark.parametrize("fraction", [math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_percentile_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], fraction)


@pytest.mark.parametrize("latency", [math.nan, math.inf, -math.inf, -1.0])
def test_summary_rejects_invalid_latency(latency):
    with pytest.raises(ValueError):
        summarize_latencies([latency], 1.0, 1)


@pytest.mark.parametrize("elapsed", [0.0, -1.0, math.nan, math.inf, -math.inf, 1e-320])
def test_summary_rejects_invalid_elapsed(elapsed):
    with pytest.raises(ValueError):
        summarize_latencies([1.0], elapsed, 1)


@pytest.mark.parametrize("items", [0, -1, True, False, 1.5, "2", 10**400])
def test_summary_rejects_invalid_batch_size(items):
    with pytest.raises(ValueError):
        summarize_latencies([1.0], 1.0, items)


def test_summary_rejects_overflowed_item_throughput():
    with pytest.raises(ValueError):
        summarize_latencies([1.0], 1e-200, 10**200)


def test_summary_reports_batch_aware_throughput():
    result = summarize_latencies([10.0, 20.0, 30.0], 0.6, 4)
    assert result["calls_per_second"] == pytest.approx(5.0)
    assert result["items_per_second"] == pytest.approx(20.0)
    assert result["latency_ms"]["p50"] == 20.0
    assert result["latency_ms"]["p95"] == pytest.approx(29.0)
