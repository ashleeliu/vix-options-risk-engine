"""
Stress scenario engine: shocks spot, volatility, and rates (independently
and jointly) and fully reprices the portfolio under each shock.

This is "scenario analysis" in the risk-management sense: fixed, chosen
shocks (not a statistical distribution) used to answer "what happens to
my book if the market moves like X" - complementary to the statistical
VaR/ES measures in risk_historical.py / risk_montecarlo.py.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from portfolio import Portfolio

DEFAULT_SPOT_SHOCKS = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
DEFAULT_VOL_SHOCKS = [-0.50, -0.25, 0.0, 0.25, 0.50, 1.00]      # relative change in vol
DEFAULT_RATE_SHOCKS = [-0.010, -0.005, 0.0, 0.005, 0.010]        # absolute change in r


@dataclass
class ScenarioResult:
    spot_shock: float
    vol_shock: float
    rate_shock: float
    shocked_S: float
    shocked_sigma: float
    shocked_r: float
    portfolio_value: float
    pnl: float


def run_scenarios(
    portfolio: Portfolio,
    S: float,
    r: float,
    q: float,
    sigma: float,
    spot_shocks: list = DEFAULT_SPOT_SHOCKS,
    vol_shocks: list = DEFAULT_VOL_SHOCKS,
    rate_shocks: list = DEFAULT_RATE_SHOCKS,
) -> pd.DataFrame:
    """Full grid of independent spot x vol x rate shocks, fully repricing
    the portfolio (not a linear/Greeks approximation) at each point."""
    base_value = portfolio.value(S, r, q, sigma)
    rows = []
    for ds in spot_shocks:
        for dv in vol_shocks:
            for dr in rate_shocks:
                shocked_S = S * (1 + ds)
                shocked_sigma = max(sigma * (1 + dv), 1e-4)
                shocked_r = r + dr
                value = portfolio.value(shocked_S, shocked_r, q, shocked_sigma)
                rows.append(ScenarioResult(
                    spot_shock=ds, vol_shock=dv, rate_shock=dr,
                    shocked_S=shocked_S, shocked_sigma=shocked_sigma, shocked_r=shocked_r,
                    portfolio_value=value, pnl=value - base_value,
                ))
    return pd.DataFrame([r.__dict__ for r in rows])


def worst_best_scenarios(scenario_df: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    worst = scenario_df.nsmallest(n, "pnl")
    best = scenario_df.nlargest(n, "pnl")
    return worst, best


def spot_vol_pnl_grid(portfolio: Portfolio, S: float, r: float, q: float, sigma: float,
                       spot_shocks: list = DEFAULT_SPOT_SHOCKS, vol_shocks: list = DEFAULT_VOL_SHOCKS) -> pd.DataFrame:
    """2D P&L grid over spot x vol shocks only (rates held fixed) - the
    classic stress-test heatmap."""
    base_value = portfolio.value(S, r, q, sigma)
    grid = pd.DataFrame(index=[f"{v:+.0%}" for v in vol_shocks], columns=[f"{s:+.0%}" for s in spot_shocks], dtype=float)
    for dv in vol_shocks:
        for ds in spot_shocks:
            value = portfolio.value(S * (1 + ds), r, q, max(sigma * (1 + dv), 1e-4))
            grid.loc[f"{dv:+.0%}", f"{ds:+.0%}"] = value - base_value
    grid.index.name = "vol shock"
    grid.columns.name = "spot shock"
    return grid


if __name__ == "__main__":
    from portfolio import example_spy_portfolio

    spot, r, q, sigma = 580.0, 0.045, 0.013, 0.18
    port = example_spy_portfolio(spot)

    scenario_df = run_scenarios(port, spot, r, q, sigma)
    print(f"Ran {len(scenario_df)} scenarios")

    worst, best = worst_best_scenarios(scenario_df)
    print("\nWorst 5 scenarios (P&L):")
    print(worst[["spot_shock", "vol_shock", "rate_shock", "pnl"]].to_string(index=False))
    print("\nBest 5 scenarios (P&L):")
    print(best[["spot_shock", "vol_shock", "rate_shock", "pnl"]].to_string(index=False))

    print("\nSpot x Vol P&L grid (rates held fixed):")
    grid = spot_vol_pnl_grid(port, spot, r, q, sigma)
    print(grid.round(0).to_string())
