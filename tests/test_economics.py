import math

import pytest

from edgecostbench.economics import estimate_total_cost, sensitivity_grid


BASE = dict(items_per_second=100, device_price=1000, lifespan_years=2,
            maintenance_per_year=100, idle_watts=10, load_watts=60,
            electricity_price_per_kwh=0.2, utilization=0.5, useful_fraction=0.8,
            currency="USD", cloud_price_per_million_calls=2)


def test_yearly_energy_and_useful_per_million_cost():
    r = estimate_total_cost(**BASE)
    expected_kwh = (10 * 0.5 + 60 * 0.5) * 8760 / 1000
    annual_cost = 500 + 100 + expected_kwh * 0.2
    useful_items = 100 * 0.5 * 8760 * 3600 * 0.8
    assert r["annual"]["electricity_kwh"] == pytest.approx(expected_kwh)
    assert r["annual"]["total_cost"] == pytest.approx(annual_cost)
    assert r["annual"]["useful_items"] == pytest.approx(useful_items)
    assert r["edge_cost_per_million_useful_items"] == pytest.approx(annual_cost * 1e6 / useful_items)
    assert r["cloud_cost_per_million_useful_items"] == pytest.approx(2.5)
    assert r["break_even_utilization"] is not None
    assert r["edge_cheaper_at_assumed_utilization"] is True
    assert r["assumptions"]["cloud_quality_assumed_equal_to_edge"] is True


def test_zero_utilization_is_not_infinite_cost_or_free_inference():
    r = estimate_total_cost(**{**BASE, "utilization": 0})
    assert r["annual"]["useful_items"] == 0
    assert r["edge_cost_per_million_useful_items"] is None
    assert r["edge_cheaper_at_assumed_utilization"] is None
    assert r["annual"]["electricity_kwh"] == pytest.approx(10 * 8760 / 1000)


def test_cloud_break_even_not_possible_when_cloud_free():
    r = estimate_total_cost(**{**BASE, "cloud_price_per_million_calls": 0})
    assert r["break_even_utilization"] is None
    assert r["edge_cheaper_at_assumed_utilization"] is False


def test_cloud_success_fraction_can_differ_from_edge():
    r = estimate_total_cost(**{**BASE, "cloud_useful_fraction": 1.0})
    assert r["cloud_cost_per_million_useful_items"] == 2
    assert r["assumptions"]["cloud_quality_assumed_equal_to_edge"] is False


def test_sensitivity_changes_utilization_and_tariff():
    results = sensitivity_grid(BASE, [0.25, 0.75], [0.1, 0.3])
    assert len(results) == 4
    assert results[0]["edge_cost_per_million_useful_items"] > results[2]["edge_cost_per_million_useful_items"]
    assert results[0]["edge_cost_per_million_useful_items"] < results[1]["edge_cost_per_million_useful_items"]
    with pytest.raises(ValueError, match="empty"):
        sensitivity_grid(BASE, [], [0.1])


@pytest.mark.parametrize("field,value", [
    ("items_per_second", 0), ("items_per_second", math.nan),
    ("device_price", -1), ("lifespan_years", 0), ("maintenance_per_year", -1),
    ("idle_watts", -1), ("load_watts", -1),
    ("electricity_price_per_kwh", math.inf), ("utilization", -0.01),
    ("utilization", 1.01), ("useful_fraction", 0), ("useful_fraction", 1.01),
    ("cloud_price_per_million_calls", -1), ("cloud_useful_fraction", 0),
])
def test_rejects_unrealistic_unknown_or_invalid_inputs(field, value):
    payload = {**BASE, field: value}
    with pytest.raises(ValueError):
        estimate_total_cost(**payload)


def test_rejects_load_below_idle_and_missing_cloud_pricing():
    with pytest.raises(ValueError, match="cannot be less"):
        estimate_total_cost(**{**BASE, "idle_watts": 100})
    with pytest.raises(ValueError, match="requires cloud pricing"):
        estimate_total_cost(**{**BASE, "cloud_price_per_million_calls": None,
                               "cloud_useful_fraction": 0.9})


def test_derived_cost_overflow_is_rejected():
    with pytest.raises(ValueError, match="overflow"):
        estimate_total_cost(**{**BASE, "device_price": 1e308, "lifespan_years": 1e-308})
