"""Estimated inference electricity costs, never presented as measured energy."""

from __future__ import annotations

from math import isfinite


def estimate_cost(mean_latency_ms: float, power_watts: float | None,
                  electricity_price_per_kwh: float | None, currency: str) -> dict:
    if not isfinite(mean_latency_ms) or mean_latency_ms < 0:
        raise ValueError("latency cannot be negative")
    if power_watts is not None and (not isfinite(power_watts) or power_watts <= 0):
        raise ValueError("power_watts must be positive")
    if electricity_price_per_kwh is not None and (not isfinite(electricity_price_per_kwh) or electricity_price_per_kwh < 0):
        raise ValueError("electricity price cannot be negative")
    if electricity_price_per_kwh is not None and power_watts is None:
        raise ValueError("electricity price requires --power-watts")
    # kWh = watts * milliseconds / (1000 ms/s * 3600 s/h * 1000 W/kW)
    energy = (power_watts * mean_latency_ms / 3_600_000_000) if power_watts is not None else None
    price = (energy * electricity_price_per_kwh) if energy is not None and electricity_price_per_kwh is not None else None
    return {
        "energy_measurement": None,
        "assumed_average_power_watts": power_watts,
        "estimated_kwh_per_call": energy,
        "electricity_price_per_kwh": electricity_price_per_kwh,
        "currency": currency,
        "estimated_electricity_cost_per_million_calls": price * 1_000_000 if price is not None else None,
        "notes": "Power is a user assumption, not sensor telemetry. Uses mean timed call latency; excludes idle, setup, cooling, hardware and cloud fees.",
    }
