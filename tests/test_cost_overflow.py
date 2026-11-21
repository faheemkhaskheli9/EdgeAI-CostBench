import math

import pytest

from edgecostbench.cost import estimate_cost


@pytest.mark.parametrize("latency,watts,tariff", [
    (1e308, 1e308, None),  # overflow in energy
    (1e308, 1e100, 1e200),  # overflow in price
    (1e100, 1e100, 1e115),  # overflow only after per-million scaling
])
def test_rejects_derived_overflow(latency, watts, tariff):
    with pytest.raises(ValueError, match="overflow"):
        estimate_cost(latency, watts, tariff, "USD")


def test_large_finite_inputs_can_still_produce_finite_estimate():
    result = estimate_cost(1e150, 1e150, 1, "USD")
    assert math.isfinite(result["estimated_electricity_cost_per_million_calls"])


def test_zero_latency_has_zero_energy_even_with_large_power():
    result = estimate_cost(0, 1e308, 1e308, "USD")
    assert result["estimated_kwh_per_call"] == 0
    assert result["estimated_electricity_cost_per_million_calls"] == 0
