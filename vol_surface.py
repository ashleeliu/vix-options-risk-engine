import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime
from scipy.stats import norm

# pull options data
ticker= yf.Ticker("^VIX")
spot = ticker.history(period="1d")["Close"].iloc[-1]
print(f"VIX spot price: {spot:.2f}")

today = datetime.now()

expiries = ['2026-09-16', '2026-10-21', '2026-11-18', '2026-12-16', '2027-02-17']
all_data = []

for exp in expiries:
    chain = ticker.option_chain(exp)
    calls = chain.calls.copy()
    calls['expiry'] = exp
    days = (parse(exp) - today).days
    calls['T'] = days / 365
    calls['mid'] = (calls['bid'] + calls['ask']) / 2
    calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
    calls = calls[(calls['strike'] > spot * 0.85) & (calls['strike'] < spot * 1.15)]
    all_data.append(calls)
    print(f"{exp}: {len(calls)} contracts loaded")

# black scholes IV extraction
r = 0.045  # risk-free rate (approx 3-month T-bill)
q = 0.013  # SPY dividend yield

def bs_call_price(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def get_iv(market_price, S, K, T, r, q):
    try:
        iv = brentq(lambda sigma: bs_call_price(S, K, T, r, q, sigma) - market_price, 1e-6, 10)
        return iv
    except:
        return np.nan

# apply IV extraction to all contracts
import pandas as pd
df = pd.concat(all_data, ignore_index=True)

print("Extracting implied volatilities...")
df['iv'] = df.apply(lambda row: get_iv(row['mid'], spot, row['strike'], row['T'], r, q), axis=1)
df = df.dropna(subset=['iv'])
df = df[(df['iv'] > 0.01) & (df['iv'] < 2.0)]  # filter out bad IVs

print(f"Successfully extracted IV for {len(df)} contracts")
print(df[['strike', 'expiry', 'T', 'mid', 'iv']].head(10))

from scipy.interpolate import griddata
# plot volatility surface
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

strikes = df['strike'].values
maturities = df['T'].values
ivs = df['iv'].values

# create grid for smooth surface
strike_grid = np.linspace(strikes.min(), strikes.max(), 50)
T_grid = np.linspace(maturities.min(), maturities.max(), 50)
K_mesh, T_mesh = np.meshgrid(strike_grid, T_grid)

# interpolate IV onto grid
iv_mesh = griddata((strikes, maturities), ivs, (K_mesh, T_mesh), method='cubic')

# plot
surf = ax.plot_surface(K_mesh, T_mesh, iv_mesh, cmap='plasma', alpha=0.85)
fig.colorbar(surf, ax=ax, shrink=0.5, label='Implied Volatility')

ax.set_xlabel('Strike')
ax.set_ylabel('Time to Expiry (Years)')
ax.set_zlabel('Implied Volatility')
ax.set_title('SPY Implied Volatility Surface')

plt.tight_layout()
plt.savefig('vol_surface.png', dpi=150)
plt.close()
print("Surface saved as vol_surface.png")

# arbitrage detection

violations = []

# calendar arbitrage
for strike in df['strike'].unique():
    subset = df[df['strike'] == strike].sort_values('T')
    if len(subset) < 2:
        continue
    total_var = (subset['iv'].values ** 2) * subset['T'].values
    for i in range(1, len(total_var)):
        if total_var[i] <= total_var[i-1]:
            violations.append({
                'type': 'Calendar Arbitrage',
                'strike': strike,
                'expiry_1': subset['expiry'].iloc[i-1],
                'expiry_2': subset['expiry'].iloc[i],
                'detail': f"Total var decreased: {total_var[i-1]:.4f} → {total_var[i]:.4f}"
            })

# butterfly arbitrage
for exp in df['expiry'].unique():
    subset = df[df['expiry'] == exp].sort_values('strike')
    if len(subset) < 3:
        continue
    prices = subset['mid'].values
    strikes = subset['strike'].values
    for i in range(1, len(prices) - 1):
        dK1 = strikes[i] - strikes[i-1]
        dK2 = strikes[i+1] - strikes[i]
        butterfly = (prices[i-1]/dK1 - prices[i]*(1/dK1 + 1/dK2) + prices[i+1]/dK2)
        if butterfly < 0:
            violations.append({
                'type': 'Butterfly Arbitrage',
                'strike': strikes[i],
                'expiry_1': exp,
                'expiry_2': exp,
                'detail': f"Negative butterfly spread: {butterfly:.4f}"
            })

# print results
vdf = pd.DataFrame(violations)
total = len(df)
n_cal = len(vdf[vdf['type'] == 'Calendar Arbitrage']) if len(vdf) > 0 else 0
n_but = len(vdf[vdf['type'] == 'Butterfly Arbitrage']) if len(vdf) > 0 else 0

print(f"\n{'='*50}")
print(f"ARBITRAGE DETECTION RESULTS")
print(f"{'='*50}")
print(f"Total contracts analyzed: {total}")
print(f"Calendar arbitrage violations: {n_cal}")
print(f"Butterfly arbitrage violations: {n_but}")
print(f"Total violations: {n_cal + n_but}")
print(f"Arbitrage-free rate: {(1 - (n_cal + n_but)/total)*100:.1f}%")
print(f"\nSample violations:")
if len(vdf) > 0:
    print(vdf.head(10).to_string())