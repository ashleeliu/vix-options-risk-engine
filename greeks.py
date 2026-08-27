"""
Analytical Black-Scholes Greeks, validated against central finite
differences.

All Greeks are for European options with a continuous dividend yield q.
Conventions:
  - vega, theta are per 1.0 unit of volatility / 1.0 year, respectively
    (i.e. NOT the "per 1% vol" / "per day" scaling some desks quote -
    see `vega_per_pct` / `theta_per_day` helpers below for that).
  - theta is dPrice/dt where t is calendar time (so theta is usually
    negative for long options, since value decays as time passes and T
    shrinks): theta = -dPrice/dT.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str = "call") -> float:
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def analytical_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str = "call") -> Greeks:
    """Closed-form Black-Scholes Greeks."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -(S * disc_q * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (
            -(S * disc_q * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def vega_per_pct(g: Greeks) -> float:
    """Convention some desks use: P&L per 1 percentage-point (0.01) vol move."""
    return g.vega * 0.01


def theta_per_day(g: Greeks) -> float:
    """Convention some desks use: P&L per calendar day of time decay."""
    return g.theta / 365.0


def finite_difference_greeks(S, K, T, r, q, sigma, option_type: str = "call", eps: float = 1e-4) -> Greeks:
    """Central finite-difference Greeks, used purely to validate the
    analytical formulas above - not meant for production use (much slower,
    and noisier for gamma/theta than the closed forms)."""
    price = lambda S_, T_, sigma_: bs_price(S_, K, T_, r, q, sigma_, option_type)

    delta = (price(S + eps, T, sigma) - price(S - eps, T, sigma)) / (2 * eps)
    gamma = (price(S + eps, T, sigma) - 2 * price(S, T, sigma) + price(S - eps, T, sigma)) / (eps ** 2)
    vega = (price(S, T, sigma + eps) - price(S, T, sigma - eps)) / (2 * eps)
    # theta = -dPrice/dT (time decay as calendar time passes, i.e. T shrinks)
    theta = -(price(S, T + eps, sigma) - price(S, T - eps, sigma)) / (2 * eps)
    rho_eps = 1e-4
    rho = (bs_price(S, K, T, r + rho_eps, q, sigma, option_type) - bs_price(S, K, T, r - rho_eps, q, sigma, option_type)) / (2 * rho_eps)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def validate_greeks(n_trials: int = 200, seed: int = 0, tol: float = 1e-3) -> dict:
    """Compare analytical vs. finite-difference Greeks across randomized
    (S, K, T, sigma) combinations. Returns max absolute and max relative
    error per Greek, and prints a pass/fail summary."""
    rng = np.random.default_rng(seed)
    max_abs_err = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    for _ in range(n_trials):
        S = rng.uniform(50, 500)
        K = S * rng.uniform(0.7, 1.3)
        T = rng.uniform(0.05, 2.0)
        sigma = rng.uniform(0.05, 0.8)
        r = rng.uniform(0.0, 0.06)
        q = rng.uniform(0.0, 0.04)
        option_type = rng.choice(["call", "put"])

        analytic = analytical_greeks(S, K, T, r, q, sigma, option_type)
        numeric = finite_difference_greeks(S, K, T, r, q, sigma, option_type)

        for greek_name in max_abs_err:
            err = abs(getattr(analytic, greek_name) - getattr(numeric, greek_name))
            max_abs_err[greek_name] = max(max_abs_err[greek_name], err)

    print(f"Greek validation over {n_trials} randomized trials (max abs error vs. finite differences):")
    all_pass = True
    for greek_name, err in max_abs_err.items():
        status = "PASS" if err < tol else "FAIL"
        all_pass &= err < tol
        print(f"  {greek_name:6s}: {err:.6f}  [{status}]")
    print("All Greeks validated within tolerance." if all_pass else "Some Greeks exceeded tolerance - investigate.")

    return max_abs_err


if __name__ == "__main__":
    validate_greeks()
