"""Capital-constrained portfolio replay accountant (M8 §3 / §13).

The M7 economic metrics scored a *series of independent per-trade net returns* and built the
"equity curve" by compounding them sequentially (``cumprod(1+r)``). With ~3000 strided entries at
~−0.0012 each that compounds to ~e^−3.6 ≈ 2.7% of capital — a ~97% "drawdown". That number is a
**single-position, all-in, sequential, fully-reinvested** curve. It is NOT executable portfolio
equity, because the entries **overlap in time** and in reality would be **concurrent positions
sharing one capital-constrained account** (with position sizing, a cash balance, exposure caps and
unused capital earning nothing).

This module models the executable account honestly:

* entries across all symbols are merged and processed in **chronological** order;
* each admitted entry is sized by a **fixed fraction of current equity** (fractional shares),
  subject to a **max concurrent positions** cap and a **max gross exposure** cap;
* **cash is conserved** — capital is reserved on entry and returned (± realized P&L − costs) on
  exit; capital that cannot be deployed simply stays in cash;
* the portfolio is **marked-to-market on the 1-minute time grid**, giving a genuine time-indexed
  equity curve from which drawdown, CAGR and (properly annualized) Sharpe are computed;
* the position exit uses the **identical conservative fill model** as the M7 simulator
  (stop-before-target within a bar, gap-through at the open, −0.5% protective stop, −3% emergency,
  time-stop at ``max_hold``), so only the *accounting*, not the fill assumptions, changes.

This is historical-replay on genuine bars — simulated, not live, never a profitability claim.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

REPLAY_VERSION = "portfolio-replay-1.0.0"
_NS_PER_YEAR = 365.25 * 24 * 3600 * 1_000_000_000
_TRADING_MINUTES_PER_YEAR = 252 * 390  # for per-minute Sharpe annualization


@dataclass
class PortfolioReplayConfig:
    initial_capital: float = 100_000.0
    capital_fraction_per_position: float = 0.10     # 10% of equity per position
    max_concurrent_positions: int = 10
    max_gross_exposure: float = 1.0                 # <=100% of equity deployed
    cost_frac: float = 0.0004                       # per side
    protective_stop: float = 0.005                  # mandatory −0.5%
    emergency_stop: float = 0.03                    # inactive candidate reference (−3%)
    max_hold_bars: int = 120
    allow_fractional_shares: bool = True

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class ReplayResult:
    config: dict
    n_entry_signals: int = 0
    n_trades_taken: int = 0
    n_skipped_no_capital: int = 0
    n_skipped_concurrency: int = 0
    final_equity: float = 0.0
    min_cash_observed: float = 0.0
    cash_never_negative: bool = True
    total_return: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0            # time-indexed portfolio-equity drawdown (fraction)
    sharpe_annualized: float = 0.0
    sortino_annualized: float = 0.0
    volatility_annualized: float = 0.0
    win_rate: float = 0.0
    net_expectancy_per_trade: float = 0.0
    profit_factor: float = 0.0
    avg_concurrent_positions: float = 0.0
    peak_concurrent_positions: int = 0
    avg_capital_utilization: float = 0.0
    turnover: float = 0.0
    span_days: float = 0.0
    note: str = ""
    per_symbol_trades: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class _Position:
    symbol: str
    entry_idx: int
    exit_idx: int
    entry_ns: int
    exit_ns: int
    entry_price: float
    exit_price: float
    net_return: float          # fractional, net of both-side costs
    shares: float
    capital: float             # capital reserved at entry


def resolve_exit(bars: dict, entry_i: int, *, cost_frac: float, protective_stop: float,
                 emergency_stop: float, max_hold: int) -> tuple | None:
    """Resolve one long position's exit with the M7 conservative fill model.

    Returns (exit_idx, entry_price, exit_price, net_return) or None if not resolvable.
    Entry at next-bar open + cost; stop fills at the stop level or the bar open if it gapped
    through; a stop is assumed to trigger before any favorable target within a bar; −3% emergency
    is independent; time-stop at close. Net return subtracts the exit-side cost (entry-side cost is
    already in the entry price)."""
    o = bars["o"]
    low = bars["l"]
    c = bars["c"]
    n = len(c)
    if entry_i + 1 >= n:
        return None
    entry = float(o[entry_i + 1]) * (1.0 + cost_frac)
    if entry <= 0:
        return None
    hi = min(n, entry_i + 1 + max_hold)
    win_low = low[entry_i + 1:hi].astype(np.float64)
    win_open = o[entry_i + 1:hi].astype(np.float64)
    stop_price = entry * (1.0 - protective_stop)
    emerg_price = entry * (1.0 - emergency_stop)
    breach = np.nonzero(win_low <= stop_price)[0]
    if len(breach):
        k = int(breach[0])
        level = emerg_price if win_low[k] <= emerg_price else stop_price
        exit_price = float(min(win_open[k], level))
        exit_idx = entry_i + 1 + k
    else:
        exit_price = float(c[hi - 1])
        exit_idx = hi - 1
    net = exit_price / entry - 1.0 - cost_frac
    return exit_idx, entry, exit_price, net


def replay_portfolio(symbol_bars: dict, symbol_entries: dict, *,
                     config: PortfolioReplayConfig | None = None) -> ReplayResult:
    """Capital-constrained multi-symbol portfolio replay.

    ``symbol_bars``: {symbol: bars_dict(ts,o,h,l,c,...)}; ``symbol_entries``: {symbol: [entry_idx]}.
    Entries are merged chronologically by their execution timestamp (bar entry_idx+1 open).
    """
    cfg = config or PortfolioReplayConfig()
    # Build a chronological event list of (exec_ns, symbol, entry_idx).
    events: list = []
    n_signals = 0
    for sym, entries in symbol_entries.items():
        bars = symbol_bars.get(sym)
        if bars is None:
            continue
        ts = bars["ts"]
        n = len(ts)
        for e in entries:
            if e + 1 < n:
                events.append((int(ts[e + 1]), sym, int(e)))
                n_signals += 1
    events.sort(key=lambda x: x[0])
    if not events:
        return ReplayResult(config=cfg.to_dict(), note="no entry signals")

    cash = cfg.initial_capital
    equity = cfg.initial_capital
    open_positions: list = []          # heap of (exit_ns, seq, _Position)
    closed: list = []
    seq = 0
    taken = 0
    skip_cap = 0
    skip_conc = 0
    min_cash = cfg.initial_capital
    # equity samples at each event time (plus position exits) for a time-indexed curve
    equity_samples: list = []          # (ns, equity)
    reserved = 0.0

    def _close_due(now_ns: int):
        nonlocal cash, reserved, min_cash
        while open_positions and open_positions[0][0] <= now_ns:
            _, _, pos = heapq.heappop(open_positions)
            proceeds = pos.capital * (1.0 + pos.net_return)
            cash += proceeds
            reserved -= pos.capital
            min_cash = min(min_cash, cash)
            closed.append(pos)

    for exec_ns, sym, entry_i in events:
        _close_due(exec_ns)
        equity = cash + reserved        # reserved held at cost basis (MTM handled at sampling)
        # admission checks
        if len(open_positions) >= cfg.max_concurrent_positions:
            skip_conc += 1
            equity_samples.append((exec_ns, equity))
            continue
        alloc = min(cfg.capital_fraction_per_position * equity,
                    cfg.max_gross_exposure * equity - reserved)
        if alloc <= 1.0 or alloc > cash:
            skip_cap += 1
            equity_samples.append((exec_ns, equity))
            continue
        res = resolve_exit(symbol_bars[sym], entry_i, cost_frac=cfg.cost_frac,
                           protective_stop=cfg.protective_stop,
                           emergency_stop=cfg.emergency_stop, max_hold=cfg.max_hold_bars)
        if res is None:
            equity_samples.append((exec_ns, equity))
            continue
        exit_idx, entry_price, exit_price, net = res
        exit_ns = int(symbol_bars[sym]["ts"][exit_idx])
        shares = (alloc / entry_price if cfg.allow_fractional_shares
                  else float(int(alloc / entry_price)))
        if shares <= 0:
            skip_cap += 1
            continue
        used = shares * entry_price
        cash -= used
        reserved += used
        min_cash = min(min_cash, cash)
        seq += 1
        pos = _Position(symbol=sym, entry_idx=entry_i, exit_idx=exit_idx, entry_ns=exec_ns,
                        exit_ns=exit_ns, entry_price=entry_price, exit_price=exit_price,
                        net_return=net, shares=shares, capital=used)
        heapq.heappush(open_positions, (exit_ns, seq, pos))
        taken += 1
        equity_samples.append((exec_ns, cash + reserved))

    # close any still-open positions
    last_ns = events[-1][0]
    _close_due(float("inf"))
    final_equity = cash
    equity_samples.append((last_ns, final_equity))

    return _summarize(cfg, closed, equity_samples, n_signals, taken, skip_cap, skip_conc,
                      final_equity, min_cash)


def _summarize(cfg, closed, equity_samples, n_signals, taken, skip_cap, skip_conc,
               final_equity, min_cash) -> ReplayResult:
    res = ReplayResult(config=cfg.to_dict(), n_entry_signals=n_signals, n_trades_taken=taken,
                       n_skipped_no_capital=skip_cap, n_skipped_concurrency=skip_conc,
                       final_equity=round(final_equity, 2),
                       min_cash_observed=round(min_cash, 2),
                       cash_never_negative=bool(min_cash >= -1e-6),
                       total_return=float(final_equity / cfg.initial_capital - 1.0))
    if not closed:
        res.note = "no trades taken"
        return res
    rets = np.array([p.net_return for p in closed], dtype=np.float64)
    res.net_expectancy_per_trade = float(rets.mean())
    res.win_rate = float((rets > 0).mean())
    wins = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    res.profit_factor = float(wins / losses) if losses > 0 else float("inf")
    res.total_return = float(final_equity / cfg.initial_capital - 1.0)

    # time-indexed equity curve (sorted, forward-filled)
    eq = sorted(equity_samples, key=lambda x: x[0])
    ns = np.array([e[0] for e in eq], dtype=np.float64)
    vals = np.array([e[1] for e in eq], dtype=np.float64)
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    res.max_drawdown = float(-dd.min()) if len(dd) else 0.0
    span_ns = ns[-1] - ns[0] if len(ns) > 1 else 0.0
    res.span_days = round(span_ns / (24 * 3600 * 1e9), 1)
    years = span_ns / _NS_PER_YEAR if span_ns > 0 else 0.0
    if years > 0 and final_equity > 0:
        res.cagr = float((final_equity / cfg.initial_capital) ** (1.0 / years) - 1.0)

    # periodic (per-event) equity returns -> annualized Sharpe/Sortino via sqrt(periods/year)
    if len(vals) > 2:
        eqr = np.diff(vals) / vals[:-1]
        eqr = eqr[np.isfinite(eqr)]
        if len(eqr) > 2 and eqr.std() > 0:
            # annualize by the measured event frequency
            periods = len(eqr)
            per_year = periods / years if years > 0 else 0.0
            ann = np.sqrt(per_year) if per_year > 0 else 0.0
            res.volatility_annualized = float(eqr.std() * ann)
            res.sharpe_annualized = float(eqr.mean() / eqr.std() * ann)
            downside = eqr[eqr < 0]
            if len(downside) > 1 and downside.std() > 0:
                res.sortino_annualized = float(eqr.mean() / downside.std() * ann)

    # concurrency + utilization (approx via trade spans)
    res.peak_concurrent_positions = _peak_concurrency(closed)
    res.avg_concurrent_positions = _avg_concurrency(closed)
    total_deployed = sum(p.capital for p in closed)
    res.turnover = round(total_deployed / cfg.initial_capital, 2)
    res.avg_capital_utilization = round(
        min(1.0, res.avg_concurrent_positions * cfg.capital_fraction_per_position), 4)
    by_sym: dict = {}
    for p in closed:
        by_sym[p.symbol] = by_sym.get(p.symbol, 0) + 1
    res.per_symbol_trades = by_sym
    res.note = ("capital-constrained portfolio replay on genuine bars; executable-equity "
                "drawdown (time-indexed), not compounded per-trade; simulated, not live")
    return res


def _peak_concurrency(closed: list) -> int:
    events = []
    for p in closed:
        events.append((p.entry_ns, 1))
        events.append((p.exit_ns, -1))
    events.sort()
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


def _avg_concurrency(closed: list) -> float:
    if not closed:
        return 0.0
    events = []
    for p in closed:
        events.append((p.entry_ns, 1))
        events.append((p.exit_ns, -1))
    events.sort()
    cur = 0
    last_ns = events[0][0]
    area = 0.0
    total = events[-1][0] - events[0][0]
    for ns, d in events:
        area += cur * (ns - last_ns)
        cur += d
        last_ns = ns
    return round(area / total, 3) if total > 0 else 0.0
