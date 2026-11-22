"""Heartbeat performance baseline + no-regression test (#1/#2).

Establishes a measurable baseline for heartbeat duration, scheduling jitter, and
(simulated) broker API latency, then repeats the SAME measurement with a
representative scanner workload running concurrently in a daemon thread. Asserts
the scanner adds no material heartbeat-latency regression (the key risk being a
scanner that blocks the execution thread via the GIL or a shared lock).

No live broker/market: broker latency is simulated deterministically.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import universe
from market_scan.config import MarketScanConfig
from market_scan.ranking import CandidateFeatures, MarketContext, rank
from order_flow_engine import OrderFlowEngine

BROKER_LATENCY_S = 0.002  # simulated broker API latency per heartbeat read


class _LatBroker:
    def get_account_snapshot(self):
        time.sleep(BROKER_LATENCY_S)
        return SimpleNamespace(equity=10000.0, cash=5000.0, buying_power=5000.0)


def _measure_heartbeat(broker, iterations=200) -> dict:
    durations: list[float] = []
    intervals: list[float] = []
    last = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        acct = broker.get_account_snapshot()          # the heartbeat's broker read
        _ = float(acct.equity) * 0.0                   # trivial heartbeat work
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000.0)
        if last is not None:
            intervals.append((t0 - last) * 1000.0)
        last = t0
    ds = sorted(durations)
    return {
        "mean_ms": statistics.mean(durations),
        "median_ms": statistics.median(durations),
        "p95_ms": ds[max(0, int(0.95 * len(ds)) - 1)],
        "jitter_ms": statistics.pstdev(intervals) if intervals else 0.0,
    }


def _scanner_workload(stop: dict, tmp: Path) -> None:
    """Representative steady-state scanner load: ingest + rank + atomic publish."""
    ofe = OrderFlowEngine()
    syms = [f"S{i}" for i in range(50)]
    cfg = MarketScanConfig()
    mkt = MarketContext(regime_label="BULL", regime_score=40.0, spy_score=10.0, qqq_score=10.0)
    up = tmp / "perf_universe.txt"
    i = 0
    while not stop["v"]:
        now = 1_000_000.0 + i
        for s in syms:
            ofe.ingest_trade(s, now, 50.0, 100)
            ofe.ingest_quote(s, now, 49.99, 50.01, 100, 100)
        feats = [CandidateFeatures(
            symbol=s, price=50.0, avg_daily_volume=5_000_000,
            avg_daily_dollar_volume=250_000_000, rel_volume=1.2, spread_pct=0.0004,
            quote_age_s=1.0,
            directional_row={"pdi": 30, "mdi": 10, "adx": 25, "vwap_dist": 0.01,
                             "short_mom": 0.02, "bb_dist": 0.01},
            snapshot=ofe.snapshot(s, now), vwap_dist=0.01, bb_mid_dist=0.01) for s in syms]
        ranked = rank(feats, config=cfg, market=mkt)
        try:
            universe.write_canonical_universe([r.symbol for r in ranked[:30]], up)
        except Exception:  # noqa: BLE001
            pass
        i += 1
        time.sleep(0.003)


def test_heartbeat_not_materially_regressed_by_scanner(tmp_path, capsys):
    broker = _LatBroker()
    base = _measure_heartbeat(broker)

    stop = {"v": False}
    th = threading.Thread(target=_scanner_workload, args=(stop, tmp_path), daemon=True)
    th.start()
    time.sleep(0.05)  # warm up the concurrent workload
    with_scan = _measure_heartbeat(broker)
    stop["v"] = True
    th.join(timeout=2.0)

    with capsys.disabled():
        print(f"\n[perf] simulated broker API latency = {BROKER_LATENCY_S * 1000:.2f} ms")
        print(f"[perf] baseline    median={base['median_ms']:.3f}ms "
              f"p95={base['p95_ms']:.3f}ms jitter={base['jitter_ms']:.3f}ms")
        print(f"[perf] w/ scanner  median={with_scan['median_ms']:.3f}ms "
              f"p95={with_scan['p95_ms']:.3f}ms jitter={with_scan['jitter_ms']:.3f}ms")

    # No material regression: the scanner must not block the heartbeat. Generous
    # tolerance (median within 2x + 5ms) to avoid CI flakiness while still
    # catching a scanner that holds the GIL / a shared lock across the heartbeat.
    assert with_scan["median_ms"] <= base["median_ms"] * 2.0 + 5.0


def test_broker_latency_is_the_heartbeat_floor():
    # Sanity: the heartbeat cost is dominated by the (simulated) broker latency,
    # confirming the harness measures what it claims.
    m = _measure_heartbeat(_LatBroker(), iterations=50)
    assert m["median_ms"] >= BROKER_LATENCY_S * 1000.0 * 0.5
