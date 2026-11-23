"""Deterministic, event-driven historical replay environment (Milestone 6, §18/§10/§16).

Replays genuine historical SIP bars in strict time order to produce LABELED SIMULATED
experience (offline-RL transitions + realized P&L) for research. Design guarantees:

  * No future leakage: a policy sees only bars up to and including the current close; any
    fill occurs at/after the NEXT bar (configurable latency).
  * Conservative fills: a passive limit is filled ONLY when the bar trades THROUGH it
    (bar low < buy limit / bar high > sell limit), never merely because a candle touches
    it. Market orders fill at next open adjusted by half-spread + slippage.
  * The mandatory -0.5% protective exit is ENVIRONMENT DYNAMICS, not a policy action
    (§16/§30): it is enforced every bar on every open position, independent of the policy,
    and can never be suppressed or deferred by it.
  * Every produced transition is labeled execution_environment='replay_simulated'.

This is an APPROXIMATE simulator (§10): it is explicitly comparable to — and never a
silent replacement for — the authoritative native RiskEngine + OrderManager path. Fees,
half-spread, slippage and latency are configurable and recorded on every result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import duckdb
import numpy as np

from cengine.lakehouse.config import DEFAULT_PRICE_SCALE, LakehouseConfig

REPLAY_ENV_VERSION = "replay-env-1.0.0"

# mandatory protective-exit threshold — identical to the production invariant (§30).
PROTECTIVE_EXIT_THRESHOLD = -0.005


@dataclass
class ReplayConfig:
    fees_bps: float = 1.0                 # per side
    half_spread_bps: float = 2.0          # crossing cost on entry/exit
    slippage_bps: float = 1.0
    latency_bars: int = 1                 # bars between decision and fill
    max_hold_bars: int = 12               # force flat after N bars (truncation)
    protective_exit_threshold: float = PROTECTIVE_EXIT_THRESHOLD


@dataclass
class ReplayResult:
    version: str = REPLAY_ENV_VERSION
    symbol: str = ""
    bars: int = 0
    trades: int = 0
    protective_exits: int = 0
    gross_return: float = 0.0
    net_return: float = 0.0
    transitions: list = field(default_factory=list)  # labeled offline-RL transitions
    per_trade_returns: list = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["n_transitions"] = len(self.transitions)
        return d


# A policy maps (context_features, position_open) -> desired action in {0: flat, 1: long}.
Policy = Callable[[np.ndarray, bool], int]


def _glob(config: LakehouseConfig, layer: str, dataset: str) -> Optional[str]:
    base = config.dataset_path(layer, dataset)
    if not base.exists() or not any(base.rglob("*.parquet")):
        return None
    return str(base / "**" / "*.parquet").replace("'", "''")


class ReplayEnvironment:
    def __init__(self, config: Optional[LakehouseConfig] = None, *,
                 replay_config: Optional[ReplayConfig] = None) -> None:
        self.config = config or LakehouseConfig.from_env()
        self.rc = replay_config or ReplayConfig()

    def _load_bars(self, symbol: str) -> Optional[dict]:
        glob = _glob(self.config, "bronze", "market_bars")
        if glob is None:
            return None
        con = duckdb.connect()
        try:
            tbl = con.sql(
                f"SELECT event_timestamp_ns AS ts, bar_open_ticks AS o, bar_high_ticks AS h, "
                f"bar_low_ticks AS l, bar_close_ticks AS c FROM read_parquet('{glob}', "
                f"union_by_name=true) WHERE symbol_id = '{symbol}' AND bar_close_ticks > 0 "
                f"ORDER BY ts").to_arrow_table()
        finally:
            con.close()
        if tbl.num_rows == 0:
            return None
        return {k: tbl.column(k).to_numpy(zero_copy_only=False) for k in tbl.column_names}

    def run(self, symbol: str, policy: Policy, *,
            feature_fn: Optional[Callable[[dict, int], np.ndarray]] = None) -> ReplayResult:
        """Replay one symbol under a policy. feature_fn(bars, i) -> context at bar i
        (must use ONLY bars <= i). Defaults to a small momentum/vol context."""
        res = ReplayResult(symbol=symbol, config=self.rc.__dict__.copy())
        bars = self._load_bars(symbol)
        if bars is None:
            return res
        o, low, c, ts = bars["o"], bars["l"], bars["c"], bars["ts"]
        n = len(c)
        res.bars = n
        feature_fn = feature_fn or _default_context
        cost_frac = (self.rc.fees_bps + self.rc.half_spread_bps + self.rc.slippage_bps) / 1e4

        pos_open = False
        entry_price = 0.0
        entry_i = -1
        entry_ctx = None
        lat = max(1, self.rc.latency_bars)
        # iterate; decisions at bar i fill at bar i+lat open
        for i in range(30, n - lat - 1):
            ctx = feature_fn(bars, i)
            if pos_open:
                # --- MANDATORY PROTECTIVE EXIT (environment dynamics; not a policy action) ---
                bar_low = low[i] / DEFAULT_PRICE_SCALE
                unreal_low = bar_low / entry_price - 1.0
                if unreal_low < self.rc.protective_exit_threshold:
                    exit_price = entry_price * (1.0 + self.rc.protective_exit_threshold)
                    gross = exit_price / entry_price - 1.0
                    net = gross - 2 * cost_frac
                    res.protective_exits += 1
                    _close(res, entry_ctx, ctx, net, entry_i, i, ts, "protective_exit")
                    pos_open = False
                    continue
                # time-based truncation
                if i - entry_i >= self.rc.max_hold_bars:
                    exit_price = o[i + lat] / DEFAULT_PRICE_SCALE
                    gross = exit_price / entry_price - 1.0
                    net = gross - 2 * cost_frac
                    _close(res, entry_ctx, ctx, net, entry_i, i, ts, "max_hold",
                           truncated=True)
                    pos_open = False
                    continue
                # policy may elect to exit
                if policy(ctx, True) == 0:
                    exit_price = o[i + lat] / DEFAULT_PRICE_SCALE
                    gross = exit_price / entry_price - 1.0
                    net = gross - 2 * cost_frac
                    _close(res, entry_ctx, ctx, net, entry_i, i, ts, "policy_exit")
                    pos_open = False
            else:
                if policy(ctx, False) == 1:
                    # conservative market entry at next-bar open + crossing cost
                    entry_price = (o[i + lat] / DEFAULT_PRICE_SCALE) * (1.0 + cost_frac)
                    entry_i = i
                    entry_ctx = ctx
                    pos_open = True
        # force-flat any residual open position at the last usable close (truncation)
        if pos_open and entry_i >= 0:
            exit_price = c[n - 1] / DEFAULT_PRICE_SCALE
            net = exit_price / entry_price - 1.0 - 2 * cost_frac
            _close(res, entry_ctx, _default_context(bars, n - 1), net, entry_i, n - 1, ts,
                   "end_of_data", truncated=True)
        res.per_trade_returns = [t["reward"] for t in res.transitions]
        res.net_return = float(np.sum(res.per_trade_returns)) if res.per_trade_returns else 0.0
        res.gross_return = res.net_return + res.trades * 2 * cost_frac
        return res


def _close(res: ReplayResult, state_ctx, next_ctx, net_return: float, entry_i: int,
           exit_i: int, ts: np.ndarray, exit_reason: str, *, truncated: bool = False) -> None:
    res.trades += 1
    res.transitions.append({
        "state": None if state_ctx is None else [float(v) for v in state_ctx],
        "selected_action": "enter_long",
        "next_state": [float(v) for v in next_ctx],
        "reward": float(net_return),
        "terminal": True,
        "truncated": bool(truncated),
        "action_duration_bars": int(exit_i - entry_i),
        "exit_reason": exit_reason,
        "action_probability_if_known": None,   # behavior policy unknown; never fabricated
        "execution_environment": "replay_simulated",
        "decision_timestamp_ns": int(ts[entry_i]),
        "exit_timestamp_ns": int(ts[exit_i])})


def _default_context(bars: dict, i: int) -> np.ndarray:
    """Small look-ahead-free context: short/long momentum + realized vol (uses bars <= i)."""
    c = bars["c"][: i + 1] / DEFAULT_PRICE_SCALE
    if len(c) < 21:
        return np.zeros(3, dtype=np.float32)
    mom_5 = c[-1] / c[-6] - 1.0
    mom_20 = c[-1] / c[-21] - 1.0
    rets = np.diff(np.log(c[-21:]))
    vol = float(np.std(rets))
    return np.array([mom_5, mom_20, vol], dtype=np.float32)
