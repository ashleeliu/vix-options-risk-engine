"""
VaR backtesting via the Kupiec (1995) proportion-of-failures (POF) test.

A VaR model is "correctly calibrated" at confidence level c if, over many
days, the fraction of days where the realized loss exceeds the predicted
VaR is close to (1-c) - e.g. a 99% VaR should be breached about 1% of the
time. The Kupiec test formalizes "close to" via a likelihood-ratio test:

    LR_pof = -2 * ln[ (1-p)^(n-x) * p^x / (1 - x/n)^(n-x) * (x/n)^x ]

where p = 1-confidence (expected breach rate), n = number of observations,
x = number of actual breaches. Under the null hypothesis that the model is
correctly calibrated, LR_pof ~ chi-square(1); reject calibration (at 95%
test confidence) if LR_pof > 3.841.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass
class KupiecResult:
    n_obs: int
    n_breaches: int
    expected_breach_rate: float
    observed_breach_rate: float
    lr_statistic: float
    p_value: float
    reject_at_95: bool


def count_breaches(realized_pnl: np.ndarray, var_estimates: np.ndarray) -> np.ndarray:
    """A breach is a day where the realized loss exceeds the VaR prediction
    for that day (both `realized_pnl` and `var_estimates` are 1D arrays,
    same length, one entry per day). VaR is a positive loss number, so a
    breach is `-realized_pnl[i] > var_estimates[i]`, i.e. pnl[i] < -var[i]."""
    return realized_pnl < -var_estimates


def kupiec_test(realized_pnl: np.ndarray, var_estimates: np.ndarray, confidence: float = 0.99) -> KupiecResult:
    breaches = count_breaches(realized_pnl, var_estimates)
    n = len(realized_pnl)
    x = int(breaches.sum())
    p = 1 - confidence  # expected breach probability

    observed_rate = x / n
    # Guard the two edge cases (0 or all breaches) where the naive log-likelihood is undefined.
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        log_null = (n - x) * np.log(1 - p) + x * np.log(p)
        log_alt = (n - x) * np.log(1 - observed_rate) + x * np.log(observed_rate)
        lr = -2 * (log_null - log_alt)

    p_value = 1 - chi2.cdf(lr, df=1)

    return KupiecResult(
        n_obs=n, n_breaches=x,
        expected_breach_rate=p, observed_breach_rate=observed_rate,
        lr_statistic=lr, p_value=p_value, reject_at_95=lr > 3.841,
    )


def print_kupiec_result(result: KupiecResult, confidence: float):
    print(f"Kupiec POF test @ {confidence:.0%} VaR:")
    print(f"  Observations: {result.n_obs}, Breaches: {result.n_breaches}")
    print(f"  Expected breach rate: {result.expected_breach_rate:.2%}   Observed: {result.observed_breach_rate:.2%}")
    print(f"  LR statistic: {result.lr_statistic:.3f}   p-value: {result.p_value:.3f}")
    verdict = "REJECT calibration (model likely mis-specified)" if result.reject_at_95 else "Fail to reject (model calibration looks reasonable)"
    print(f"  Verdict at 95% test confidence: {verdict}")


if __name__ == "__main__":
    from portfolio import example_spy_portfolio
    from risk_historical import generate_synthetic_returns, historical_pnl_distribution
    from risk_approx import delta_normal_var
    from greeks import analytical_greeks

    spot, r, q, sigma = 580.0, 0.045, 0.013, 0.18
    daily_vol = 0.011
    confidence = 0.99
    port = example_spy_portfolio(spot)

    # Simulate a long "live" backtest window: each day, compute a static
    # Delta-Normal VaR estimate (as if computed at the start of each day)
    # and compare it against that day's realized P&L from an independent
    # daily return draw.
    n_days = 750
    daily_returns = generate_synthetic_returns(n_days=n_days, daily_vol=daily_vol, seed=99)

    delta = port.aggregate_greeks(spot, r, q, sigma).delta
    var_estimate = delta_normal_var(delta, spot, daily_vol, confidence)
    var_series = np.full(n_days, var_estimate)  # static book/vol assumption for simplicity

    realized_pnl = historical_pnl_distribution(port, spot, r, q, sigma, daily_returns)

    result = kupiec_test(realized_pnl, var_series, confidence=confidence)
    print_kupiec_result(result, confidence)
