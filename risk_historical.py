"""
Historical VaR and Expected Shortfall (ES), via full revaluation.

Methodology: take a history of daily underlying log-returns (real ones if
you have them - e.g. from SPY price history; synthetic ones for testing),
apply each historical day's return as a shock to today's spot price, fully
reprice the portfolio at each shocked spot, and read VaR/ES off the
resulting empirical P&L distribution. This makes no distributional
assumption about returns (unlike Delta-Normal) - it just replays history.

VaR_alpha  = -(the alpha-quantile of the P&L distribution)   [a loss, so positive]
ES_alpha   = -(the mean P&L in the tail beyond VaR_alpha)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio import Portfolio


@dataclass
class VarResult:
    confidence: float
    var: float
    es: float
    n_obs: int
    n_tail: int


def historical_pnl_distribution(
    portfolio: Portfolio,
    S: float, r: float, q: float, sigma: float,
    historical_log_returns: np.ndarray,
) -> np.ndarray:
    """Full-revaluation P&L for each historical daily return, holding vol
    and rates fixed (pure spot-shock historical simulation - the most
    common baseline version of the method)."""
    base_value = portfolio.value(S, r, q, sigma)
    shocked_S = S * np.exp(historical_log_returns)
    pnl = np.array([portfolio.value(s, r, q, sigma) - base_value for s in shocked_S])
    return pnl


def compute_var_es(pnl: np.ndarray, confidence: float = 0.99) -> VarResult:
    """VaR/ES from an empirical P&L sample. `confidence=0.99` means the 1%
    worst-case loss level (99% VaR)."""
    alpha = 1 - confidence
    var_level = -np.quantile(pnl, alpha)  # a loss is reported as a positive number
    tail = pnl[pnl <= -var_level]
    es = -tail.mean() if len(tail) > 0 else var_level
    return VarResult(confidence=confidence, var=var_level, es=es, n_obs=len(pnl), n_tail=len(tail))


def generate_synthetic_returns(n_days: int = 500, daily_vol: float = 0.011, seed: int = 1) -> np.ndarray:
    """Synthetic daily log-returns (fat-tailed via Student-t, since real
    equity returns aren't Gaussian) for testing without a real price
    history. Swap this for real SPY daily log-returns in production."""
    rng = np.random.default_rng(seed)
    t_draws = rng.standard_t(df=5, size=n_days)
    t_draws /= t_draws.std()  # normalize so daily_vol is the actual realized vol
    return t_draws * daily_vol


if __name__ == "__main__":
    from portfolio import example_spy_portfolio

    spot, r, q, sigma = 580.0, 0.045, 0.013, 0.18
    port = example_spy_portfolio(spot)

    returns = generate_synthetic_returns(n_days=500)
    pnl = historical_pnl_distribution(port, spot, r, q, sigma, returns)

    for conf in (0.95, 0.99):
        result = compute_var_es(pnl, confidence=conf)
        print(f"{conf:.0%} Historical VaR: ${result.var:,.2f}   ES: ${result.es:,.2f}   (n_obs={result.n_obs}, n_tail={result.n_tail})")
