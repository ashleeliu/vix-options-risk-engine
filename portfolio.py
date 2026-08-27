"""
Multi-option portfolio: aggregate pricing and Greeks across positions.

A Position is one option leg (long or short, via signed quantity) on a
single underlying. Portfolio just sums position-level price/Greeks scaled
by quantity - trivial to reason about, but it's the building block every
later module (scenarios, VaR, backtesting) reuses for full revaluation.
"""

from dataclasses import dataclass

from greeks import Greeks, analytical_greeks, bs_price


@dataclass
class Position:
    option_type: str   # "call" or "put"
    K: float            # strike
    T: float            # time to expiry, years
    quantity: float      # signed: positive = long, negative = short
    label: str = ""      # optional, for readable reports

    def price(self, S: float, r: float, q: float, sigma: float) -> float:
        return bs_price(S, self.K, self.T, r, q, sigma, self.option_type)

    def greeks(self, S: float, r: float, q: float, sigma: float) -> Greeks:
        return analytical_greeks(S, self.K, self.T, r, q, sigma, self.option_type)


class Portfolio:
    def __init__(self, positions: list[Position] = None):
        self.positions: list[Position] = positions or []

    def add(self, position: Position):
        self.positions.append(position)

    def value(self, S: float, r: float, q: float, sigma) -> float:
        """Total portfolio value. `sigma` can be a single float (flat vol
        for every position) or a dict {label: sigma} for per-position vol
        (useful once each leg has its own smile-implied IV)."""
        total = 0.0
        for pos in self.positions:
            iv = sigma[pos.label] if isinstance(sigma, dict) else sigma
            total += pos.quantity * pos.price(S, r, q, iv)
        return total

    def aggregate_greeks(self, S: float, r: float, q: float, sigma) -> Greeks:
        """Portfolio Greeks = quantity-weighted sum of position Greeks.
        Valid because Black-Scholes Greeks are linear in position size for
        a fixed (S, T, sigma) - no cross terms between positions."""
        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        for pos in self.positions:
            iv = sigma[pos.label] if isinstance(sigma, dict) else sigma
            g = pos.greeks(S, r, q, iv)
            for name in totals:
                totals[name] += pos.quantity * getattr(g, name)
        return Greeks(**totals)

    def summary(self, S: float, r: float, q: float, sigma) -> dict:
        return {
            "value": self.value(S, r, q, sigma),
            "greeks": self.aggregate_greeks(S, r, q, sigma),
            "n_positions": len(self.positions),
        }


def example_spy_portfolio(spot: float = 580.0) -> Portfolio:
    """A representative multi-leg SPY options portfolio: a long call
    spread, a short put (income), and a long far-dated call (a small
    convexity/tail hedge) - varied enough that delta, gamma, and vega all
    matter, which is the point of testing risk on a real book rather than
    a single option."""
    p = Portfolio()
    p.add(Position("call", K=spot * 1.02, T=45 / 365, quantity=100, label="call_45d_102pct"))
    p.add(Position("call", K=spot * 1.08, T=45 / 365, quantity=-100, label="call_45d_108pct"))
    p.add(Position("put", K=spot * 0.95, T=30 / 365, quantity=-50, label="put_30d_95pct"))
    p.add(Position("call", K=spot * 1.10, T=180 / 365, quantity=20, label="call_180d_110pct"))
    return p


if __name__ == "__main__":
    spot, r, q, sigma = 580.0, 0.045, 0.013, 0.18
    port = example_spy_portfolio(spot)
    summary = port.summary(spot, r, q, sigma)
    print(f"Portfolio value: ${summary['value']:,.2f}")
    print(f"Aggregate Greeks: {summary['greeks']}")
