"""Annualized edge inference TCO using explicit assumptions, not energy sensors."""

from __future__ import annotations

from math import isfinite

HOURS_PER_YEAR = 365 * 24
SECONDS_PER_YEAR = HOURS_PER_YEAR * 3600


def _nonnegative(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} finite number")
    return float(value)


def _fraction(name: str, value: float, *, allow_zero: bool = False) -> float:
    _nonnegative(name, value)
    if value > 1 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be in {'[0, 1]' if allow_zero else '(0, 1]'}")
    return float(value)


def estimate_total_cost(*, items_per_second: float, device_price: float, lifespan_years: float,
                        maintenance_per_year: float, idle_watts: float, load_watts: float,
                        electricity_price_per_kwh: float, utilization: float, useful_fraction: float,
                        currency: str, cloud_price_per_million_calls: float | None = None,
                        cloud_useful_fraction: float | None = None) -> dict:
    """Estimate annual 24/7 available-device cost and cost per useful item.

    Utilization is the fraction of calendar time running *sequential benchmark*
    calls at the reported items/s. Remaining time draws idle watts. Output
    quality and cloud/edge utility fractions are assumptions, not measurements.
    """
    rate = _nonnegative("items_per_second", items_per_second, positive=True)
    capex = _nonnegative("device_price", device_price)
    years = _nonnegative("lifespan_years", lifespan_years, positive=True)
    maintenance = _nonnegative("maintenance_per_year", maintenance_per_year)
    idle = _nonnegative("idle_watts", idle_watts)
    load = _nonnegative("load_watts", load_watts)
    price = _nonnegative("electricity_price_per_kwh", electricity_price_per_kwh)
    util = _fraction("utilization", utilization, allow_zero=True)
    useful = _fraction("useful_fraction", useful_fraction)
    if load < idle:
        raise ValueError("load_watts cannot be less than idle_watts")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("currency must be a non-empty label")
    if cloud_price_per_million_calls is None and cloud_useful_fraction is not None:
        raise ValueError("cloud_useful_fraction requires cloud pricing")
    cloud_rate = None if cloud_price_per_million_calls is None else _nonnegative(
        "cloud_price_per_million_calls", cloud_price_per_million_calls)
    cloud_fraction = (None if cloud_rate is None else useful if cloud_useful_fraction is None else
                      _fraction("cloud_useful_fraction", cloud_useful_fraction))

    annual_capital = capex / years
    annual_energy_kwh = (idle * (1 - util) + load * util) * HOURS_PER_YEAR / 1000
    annual_energy_cost = annual_energy_kwh * price
    annual_cost = annual_capital + maintenance + annual_energy_cost
    annual_items = rate * util * SECONDS_PER_YEAR
    annual_useful = annual_items * useful
    if not all(isfinite(value) for value in (annual_capital, annual_energy_kwh, annual_energy_cost,
                                              annual_cost, annual_items, annual_useful)):
        raise ValueError("derived annual metrics overflow; use realistic assumptions")
    edge_per_million = (annual_cost * 1_000_000 / annual_useful) if annual_useful > 0 else None
    cloud_per_million_useful = (cloud_rate / cloud_fraction) if cloud_rate is not None else None
    if (edge_per_million is not None and not isfinite(edge_per_million)) or (
            cloud_per_million_useful is not None and not isfinite(cloud_per_million_useful)):
        raise ValueError("derived cost overflows; use realistic assumptions")

    # Solve annual edge expense <= cloud cost of the same number of useful items:
    # (fixed + incremental_power*u) <= (items/u * edge_quality * cloud_price/cloud_quality) * u
    break_even = None
    if cloud_rate is not None:
        fixed = annual_capital + maintenance + idle * HOURS_PER_YEAR * price / 1000
        incremental = (load - idle) * HOURS_PER_YEAR * price / 1000
        cloud_savings_per_unit_util = rate * SECONDS_PER_YEAR * useful * (cloud_rate / cloud_fraction) / 1_000_000
        margin = cloud_savings_per_unit_util - incremental
        if margin > 0:
            threshold = fixed / margin
            if threshold <= 1:
                break_even = threshold

    return {
        "currency": currency,
        "assumptions": {
            "items_per_second": rate, "device_price": capex, "lifespan_years": years,
            "maintenance_per_year": maintenance, "idle_watts": idle, "load_watts": load,
            "electricity_price_per_kwh": price, "utilization": util,
            "useful_fraction": useful, "cloud_price_per_million_calls": cloud_rate,
            "cloud_useful_fraction": cloud_fraction,
            "cloud_quality_assumed_equal_to_edge": cloud_rate is not None and cloud_useful_fraction is None,
        },
        "annual": {
            "capital_amortization": annual_capital,
            "maintenance": maintenance,
            "electricity_kwh": annual_energy_kwh,
            "electricity_cost": annual_energy_cost,
            "total_cost": annual_cost,
            "items": annual_items,
            "useful_items": annual_useful,
        },
        "edge_cost_per_million_useful_items": edge_per_million,
        "cloud_cost_per_million_useful_items": cloud_per_million_useful,
        "edge_cheaper_at_assumed_utilization": (edge_per_million < cloud_per_million_useful
                                              if edge_per_million is not None and cloud_per_million_useful is not None
                                              else None),
        "break_even_utilization": break_even,
        "notes": "Annual 8760-hour steady-state model; user-supplied power, uptime and success fractions are assumptions. No cooling, networking, financing, replacement, tax or sensor measurements.",
    }


def sensitivity_grid(assumptions: dict, utilizations: list[float], electricity_prices: list[float]) -> list[dict]:
    """Vary utilization and electricity price without changing remaining assumptions."""
    if not utilizations or not electricity_prices:
        raise ValueError("sensitivity ranges cannot be empty")
    output = []
    for util in utilizations:
        for price in electricity_prices:
            scenario = estimate_total_cost(**{**assumptions, "utilization": util,
                                              "electricity_price_per_kwh": price})
            output.append({
                "utilization": scenario["assumptions"]["utilization"],
                "electricity_price_per_kwh": scenario["assumptions"]["electricity_price_per_kwh"],
                "edge_cost_per_million_useful_items": scenario["edge_cost_per_million_useful_items"],
                "break_even_utilization": scenario["break_even_utilization"],
            })
    return output
