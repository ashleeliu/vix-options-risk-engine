"""
Run the SVI calibration on real, live VIX data.

This runs your actual vol_surface.py (pulling real option chains via
yfinance) and feeds its output directly into svi.py's calibration -
no synthetic data involved. Requires vol_surface.py, svi.py, and greeks.py
to all be in the same folder, and requires network access (this fetches
live data).
"""

import runpy

import numpy as np

from svi import calibrate_surface, residuals_and_rmse, rmse_table, plot_svi_fits


def main():
    # Runs vol_surface.py top to bottom (it isn't wrapped in
    # if __name__ == "__main__", so this also prints its own IV extraction
    # summary, arbitrage results, and saves vol_surface.png as a side effect)
    # and captures its variables.
    ns = runpy.run_path("vol_surface.py")

    df = ns["df"]        # columns: strike, expiry, T, mid, iv
    spot = ns["spot"]
    r = ns["r"]
    q = ns["q"]

    forward_fn = lambda T: spot * np.exp((r - q) * T)

    print("\n" + "=" * 60)
    print("SVI CALIBRATION ON LIVE DATA")
    print("=" * 60)

    fits = calibrate_surface(df, forward_fn)

    print("\nSVI calibration results by expiry:")
    for T, fit in sorted(fits.items()):
        print(f"  T={T:.3f}: a={fit.a:.4f} b={fit.b:.4f} rho={fit.rho:+.4f} "
              f"m={fit.m:+.4f} sigma={fit.sigma:.4f}  RMSE(iv)={fit.rmse_iv:.4f}  n={fit.n_points}")

    residual_df = residuals_and_rmse(df, fits, forward_fn)
    print("\nRMSE by maturity x moneyness bucket:")
    print(rmse_table(residual_df).round(4).to_string())

    overall_rmse = float(np.sqrt(np.mean(residual_df["residual"] ** 2)))
    print(f"\nOverall RMSE (fitted vs. market IV): {overall_rmse:.4f}")

    plot_svi_fits(df, fits, forward_fn, out_path="svi_calibration_live.png")


if __name__ == "__main__":
    main()
