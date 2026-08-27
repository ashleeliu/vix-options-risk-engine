"""
SVI (Stochastic Volatility Inspired) volatility smile calibration.

Fits the raw SVI parameterization to each expiry's market implied
volatilities independently:

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

where k = log-moneyness = ln(K / F) (F = forward price), and w = total
variance = iv^2 * T. This is Gatheral's standard 5-parameter form: it's
flexible enough to fit real equity/index smiles well, and (unlike raw
spline/interpolation through market points) gives you a smooth, twice-
differentiable curve you can reason about analytically and check for
arbitrage.

Parameter constraints for a well-behaved (if not fully arbitrage-free)
fit, following Gatheral & Jacquier (2014):
  b >= 0                              (variance must increase away from ATM)
  |rho| < 1                            (valid correlation-like parameter)
  sigma > 0                            (smile can't be a degenerate kink)
  a + b*sigma*sqrt(1-rho^2) >= 0        (minimum of w(k) must be non-negative)

This module fits the smile SLICE BY SLICE (each expiry independently) -
it does not yet enforce *calendar* no-arbitrage (total variance increasing
across expiries at fixed strike); see vol_surface.detect_calendar_arbitrage
for that check applied to the fitted surface.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def svi_total_variance(k, a: float, b: float, rho: float, m: float, sigma: float):
    """w(k) - vectorized over k (accepts a scalar or numpy array)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


@dataclass
class SviFit:
    T: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    rmse_iv: float
    n_points: int

    def iv(self, k):
        w = svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma)
        return np.sqrt(np.maximum(w, 1e-10) / self.T)


def _initial_guess(k: np.ndarray, w: np.ndarray) -> list:
    return [max(w.min(), 1e-4), 0.1, -0.3, float(np.median(k)), 0.1]


def _constraints_ok(a, b, rho, m, sigma) -> bool:
    if b < 0 or not (-1 < rho < 1) or sigma <= 0:
        return False
    return a + b * sigma * np.sqrt(1 - rho ** 2) >= -1e-8


def fit_svi_slice(k: np.ndarray, w: np.ndarray, T: float) -> SviFit:
    """Fit one expiry's smile. Uses SLSQP with the constraints above,
    penalizing constraint violation heavily rather than encoding it as a
    hard scipy constraint (in practice, a soft penalty converges more
    reliably from arbitrary starting points for this particular problem)."""

    def objective(params):
        a, b, rho, m, sigma = params
        penalty = 0.0
        if b < 0:
            penalty += 1e4 * b ** 2
        if abs(rho) >= 1:
            penalty += 1e4 * (abs(rho) - 0.999) ** 2
        if sigma <= 0:
            penalty += 1e4 * sigma ** 2
        min_w = a + b * max(sigma, 1e-6) * np.sqrt(max(1 - rho ** 2, 0))
        if min_w < 0:
            penalty += 1e4 * min_w ** 2

        model_w = svi_total_variance(k, a, b, rho, m, max(sigma, 1e-6))
        sse = np.sum((model_w - w) ** 2)
        return sse + penalty

    x0 = _initial_guess(k, w)
    result = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
    a, b, rho, m, sigma = result.x
    sigma = abs(sigma)

    fitted_w = svi_total_variance(k, a, b, rho, m, sigma)
    fitted_iv = np.sqrt(np.maximum(fitted_w, 1e-10) / T)
    market_iv = np.sqrt(np.maximum(w, 1e-10) / T)
    rmse_iv = float(np.sqrt(np.mean((fitted_iv - market_iv) ** 2)))

    return SviFit(T=T, a=a, b=b, rho=rho, m=m, sigma=sigma, rmse_iv=rmse_iv, n_points=len(k))


def calibrate_surface(df: pd.DataFrame, forward_fn) -> dict:
    """Fit SVI independently to every expiry in df. `forward_fn(T)` should
    return the forward price for that maturity (e.g. S*exp((r-q)*T)).
    df must have columns: strike, T, iv, expiry.
    Returns {T: SviFit}.
    """
    fits = {}
    for T in sorted(df["T"].unique()):
        subset = df[df["T"] == T]
        F = forward_fn(T)
        k = np.log(subset["strike"].values / F)
        w = (subset["iv"].values ** 2) * T
        fits[T] = fit_svi_slice(k, w, T)
    return fits


def residuals_and_rmse(df: pd.DataFrame, fits: dict, forward_fn) -> pd.DataFrame:
    """Per-contract residual (fitted_iv - market_iv), plus a moneyness
    bucket label, so RMSE can be broken out by maturity and moneyness."""
    rows = []
    for _, row in df.iterrows():
        T, K, market_iv = row["T"], row["strike"], row["iv"]
        fit = fits[T]
        F = forward_fn(T)
        k = np.log(K / F)
        fitted_iv = float(fit.iv(k))
        moneyness = K / F
        if moneyness < 0.97:
            bucket = "ITM (K/F<0.97)"
        elif moneyness > 1.03:
            bucket = "OTM (K/F>1.03)"
        else:
            bucket = "ATM (0.97-1.03)"
        rows.append({
            "T": T, "strike": K, "moneyness": moneyness, "bucket": bucket,
            "market_iv": market_iv, "fitted_iv": fitted_iv, "residual": fitted_iv - market_iv,
        })
    return pd.DataFrame(rows)


def rmse_table(residual_df: pd.DataFrame) -> pd.DataFrame:
    """RMSE of fitted-vs-market IV, broken out by maturity x moneyness
    bucket - the standard "where does the model fit poorly" diagnostic."""
    table = residual_df.groupby(["T", "bucket"])["residual"].apply(
        lambda r: np.sqrt(np.mean(r ** 2))
    ).unstack("bucket")
    return table


def plot_svi_fits(df: pd.DataFrame, fits: dict, forward_fn, out_path: str = "svi_calibration.png"):
    """One panel per expiry: raw market IV points vs. the fitted SVI
    smile curve, so fit quality is visible at a glance."""
    import matplotlib.pyplot as plt

    expiries = sorted(fits.keys())
    fig, axes = plt.subplots(1, len(expiries), figsize=(4.5 * len(expiries), 4), sharey=True)
    if len(expiries) == 1:
        axes = [axes]

    for ax, T in zip(axes, expiries):
        subset = df[df["T"] == T].sort_values("strike")
        F = forward_fn(T)
        k_market = np.log(subset["strike"].values / F)
        ax.scatter(k_market, subset["iv"].values, s=18, color="#dc2626", label="market IV", zorder=3)

        fit = fits[T]
        k_grid = np.linspace(k_market.min(), k_market.max(), 200)
        ax.plot(k_grid, fit.iv(k_grid), color="#2563eb", linewidth=1.5, label="SVI fit")

        ax.set_title(f"T={T:.2f}y\nRMSE={fit.rmse_iv:.4f}")
        ax.set_xlabel("log-moneyness  ln(K/F)")
        if T == expiries[0]:
            ax.set_ylabel("Implied Volatility")
            ax.legend(fontsize=8)

    fig.suptitle("SVI Smile Calibration: Market IV vs. Fitted Curve", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    from synthetic_data import generate_synthetic_chain

    df, true_params, market = generate_synthetic_chain()
    forward_fn = lambda T: market.spot * np.exp((market.r - market.q) * T)

    fits = calibrate_surface(df, forward_fn)

    print("SVI calibration results by expiry:")
    for T, fit in sorted(fits.items()):
        print(f"  T={T:.3f}: a={fit.a:.4f} b={fit.b:.4f} rho={fit.rho:+.4f} m={fit.m:+.4f} sigma={fit.sigma:.4f}  RMSE(iv)={fit.rmse_iv:.4f}  n={fit.n_points}")

    residual_df = residuals_and_rmse(df, fits, forward_fn)
    print("\nRMSE by maturity x moneyness bucket:")
    print(rmse_table(residual_df).round(4).to_string())

    print(f"\nOverall RMSE (fitted vs. market IV): {np.sqrt(np.mean(residual_df['residual']**2)):.4f}")

    plot_svi_fits(df, fits, forward_fn)
