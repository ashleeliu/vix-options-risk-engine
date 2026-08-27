"""
Delta-Normal and Delta-Gamma analytic VaR approximations, compared against
Monte Carlo full revaluation.

Delta-Normal: linearize P&L in the spot move (P&L ~= Delta * dS), assume
dS is Gaussian, and get a closed-form VaR = z_alpha * |Delta| * S * sigma_daily
* sqrt(horizon). Fast, but breaks down for portfolios with significant
gamma (e.g. anything with short-dated options or a lot of convexity),
since it can't tell a long-gamma book (safer than linear P&L suggests)
from a short-gamma book (riskier).

Delta-Gamma: adds the second-order (gamma) term to the P&L approximation:
P&L ~= Delta*dS + 0.5*Gamma*dS^2, then uses a Cornish-Fisher-style
correction (via the skewness gamma introduces into the P&L distribution)
instead of assuming pure Gaussian P&L. More accurate for convex books,
still an approximation.

The point of building both is the comparison itself: showing where each
approximation over/under-states risk relative to full repricing is more
informative for a risk-interview conversation than either number alone.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from greeks import Greeks
from portfolio import Portfolio


@dataclass
class ApproxVarResult:
    method: str
    confidence: float
    var: float


def delta_normal_var(delta: float, S: float, daily_vol: float, confidence: float = 0.99, horizon_days: int = 1) -> float:
    """VaR = z_alpha * |dollar delta| * sigma_daily * sqrt(horizon)."""
    z = norm.ppf(confidence)
    dollar_delta = delta  # `delta` here is already the position's Greek delta (shares-equivalent), so dollar delta = delta * S is NOT right if delta already includes S sensitivity in price terms
    # Portfolio.aggregate_greeks().delta is dPrice/dS summed over quantity, i.e. already a
    # "shares-equivalent" delta - dollar delta exposure is delta * S.
    dollar_exposure = delta * S
    sigma_horizon = daily_vol * np.sqrt(horizon_days)
    return z * abs(dollar_exposure) * sigma_horizon


def delta_gamma_var(
    delta: float, gamma: float, S: float, daily_vol: float,
    confidence: float = 0.99, horizon_days: int = 1,
) -> float:
    """Cornish-Fisher-adjusted Delta-Gamma VaR.

    P&L ~= delta*S*dR + 0.5*gamma*S^2*dR^2   where dR is the (Gaussian)
    return over the horizon. This P&L is a (shifted, scaled) noncentral
    chi-square-like variable - it has nonzero skewness from the gamma term.
    We approximate its quantile using a Cornish-Fisher expansion driven by
    that skewness, which is the standard practical shortcut (an exact
    closed form exists via chi-square inversion, but Cornish-Fisher is
    what's typically implemented and is accurate enough for reasonably
    small gamma exposures).
    """
    sigma_h = daily_vol * np.sqrt(horizon_days)
    dollar_delta = delta * S
    dollar_gamma = gamma * S ** 2

    pnl_mean = 0.5 * dollar_gamma * sigma_h ** 2  # E[dR^2] = sigma_h^2
    pnl_var = (dollar_delta * sigma_h) ** 2 + 0.5 * (dollar_gamma * sigma_h ** 2) ** 2
    pnl_std = np.sqrt(pnl_var)
    # skewness of delta*X + 0.5*gamma*X^2 for X ~ N(0, sigma_h^2):
    pnl_skew = (
        3 * dollar_delta ** 2 * dollar_gamma * sigma_h ** 4 + dollar_gamma ** 3 * sigma_h ** 6
    ) / (pnl_std ** 3 + 1e-12)

    z = norm.ppf(1 - confidence)  # left-tail quantile (loss side)
    # Cornish-Fisher expansion
    z_cf = z + (z ** 2 - 1) * pnl_skew / 6
    pnl_quantile = pnl_mean + pnl_std * z_cf
    return max(-pnl_quantile, 0.0)  # VaR reported as a positive loss number


def compare_var_methods(
    portfolio: Portfolio,
    S: float, r: float, q: float, sigma: float,
    daily_vol: float,
    montecarlo_pnl: np.ndarray,
    confidence: float = 0.99,
) -> dict:
    """Side-by-side: Delta-Normal, Delta-Gamma, and Monte Carlo full-reval
    VaR at the same confidence level, so the approximation errors are
    directly visible."""
    from risk_historical import compute_var_es

    g: Greeks = portfolio.aggregate_greeks(S, r, q, sigma)

    dn_var = delta_normal_var(g.delta, S, daily_vol, confidence)
    dg_var = delta_gamma_var(g.delta, g.gamma, S, daily_vol, confidence)
    mc_result = compute_var_es(montecarlo_pnl, confidence)

    return {
        "delta_normal_var": dn_var,
        "delta_gamma_var": dg_var,
        "montecarlo_var": mc_result.var,
        "delta_normal_error_pct": 100 * (dn_var - mc_result.var) / mc_result.var,
        "delta_gamma_error_pct": 100 * (dg_var - mc_result.var) / mc_result.var,
    }


if __name__ == "__main__":
    from portfolio import example_spy_portfolio
    from risk_montecarlo import montecarlo_pnl_distribution

    spot, r, q, sigma = 580.0, 0.045, 0.013, 0.18
    daily_vol = 0.011
    port = example_spy_portfolio(spot)

    mc_pnl = montecarlo_pnl_distribution(port, spot, r, q, sigma, n_sims=50_000, daily_vol=daily_vol)

    for conf in (0.95, 0.99):
        result = compare_var_methods(port, spot, r, q, sigma, daily_vol, mc_pnl, confidence=conf)
        print(f"\n{conf:.0%} VaR comparison:")
        print(f"  Delta-Normal: ${result['delta_normal_var']:,.2f}  ({result['delta_normal_error_pct']:+.1f}% vs. Monte Carlo)")
        print(f"  Delta-Gamma:  ${result['delta_gamma_var']:,.2f}  ({result['delta_gamma_error_pct']:+.1f}% vs. Monte Carlo)")
        print(f"  Monte Carlo (full reval): ${result['montecarlo_var']:,.2f}")
