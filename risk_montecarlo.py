"""
Monte Carlo full-revaluation VaR/ES.

Unlike historical simulation (which replays actual past days),
Monte Carlo simulates a large number of *hypothetical* joint spot/vol
moves from a chosen distribution, fully reprices the portfolio under each,
and reads VaR/ES off the resulting P&L distribution. This lets you:
  - use however many scenarios you want (not limited by history length)
  - explicitly model the spot/vol correlation (the "leverage effect":
    vol tends to rise when spot falls) rather than only what happened to
    actually co-occur historically
  - stress the tails by choosing fatter-tailed shock distributions
"""

from dataclasses import dataclass

import numpy as np

from portfolio import Portfolio
from risk_historical import VarResult, compute_var_es


def simulate_joint_shocks(
    n_sims: int,
    horizon_days: int,
    daily_vol: float,
    vol_of_vol: float = 0.15,
    spot_vol_correlation: float = -0.6,
    seed: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate `n_sims` joint (log-return, relative vol change) shocks over
    `horizon_days`, correlated via spot_vol_correlation (negative, since
    vol tends to spike when spot drops - the well-documented equity
    "leverage effect"). Both legs use Student-t marginals for fat tails,
    linked via a Gaussian copula (simple and good enough for a stress
    testbed; a real desk might use a fitted joint historical distribution
    instead).
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, spot_vol_correlation], [spot_vol_correlation, 1.0]])
    L = np.linalg.cholesky(cov)

    z = rng.standard_normal((n_sims, 2))
    correlated_normals = z @ L.T
    # convert to fat-tailed marginals via the Gaussian copula -> t-distribution trick
    u = np.array([_norm_cdf(correlated_normals[:, i]) for i in range(2)]).T
    t_draws = np.array([_t_ppf(u[:, i], df=5) for i in range(2)]).T

    horizon_vol = daily_vol * np.sqrt(horizon_days)
    log_returns = t_draws[:, 0] * horizon_vol
    vol_shocks = t_draws[:, 1] * vol_of_vol * np.sqrt(horizon_days)
    return log_returns, vol_shocks


def _norm_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)


def _t_ppf(u, df):
    from scipy.stats import t
    return t.ppf(u, df=df)


def montecarlo_pnl_distribution(
    portfolio: Portfolio,
    S: float, r: float, q: float, sigma: float,
    n_sims: int = 20_000,
    horizon_days: int = 1,
    daily_vol: float = 0.011,
    vol_of_vol: float = 0.15,
    spot_vol_correlation: float = -0.6,
    seed: int = 2,
) -> np.ndarray:
    base_value = portfolio.value(S, r, q, sigma)
    log_returns, vol_shocks = simulate_joint_shocks(
        n_sims, horizon_days, daily_vol, vol_of_vol, spot_vol_correlation, seed
    )
    shocked_S = S * np.exp(log_returns)
    shocked_sigma = np.clip(sigma * (1 + vol_shocks), 1e-4, None)

    pnl = np.array([
        portfolio.value(s, r, q, sig) - base_value
        for s, sig in zip(shocked_S, shocked_sigma)
    ])
    return pnl


if __name__ == "__main__":
    from portfolio import example_spy_portfolio

    spot, r, q, sigma = 580.0, 0.045, 0.013, 0.18
    port = example_spy_portfolio(spot)

    pnl = montecarlo_pnl_distribution(port, spot, r, q, sigma, n_sims=20_000)
    print(f"Simulated {len(pnl)} joint spot/vol scenarios")
    print(f"Mean P&L: ${pnl.mean():,.2f}  (should be near 0 - shocks are unbiased)")

    for conf in (0.95, 0.99):
        result = compute_var_es(pnl, confidence=conf)
        print(f"{conf:.0%} Monte Carlo VaR: ${result.var:,.2f}   ES: ${result.es:,.2f}   (n_obs={result.n_obs}, n_tail={result.n_tail})")
