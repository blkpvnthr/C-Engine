"""L1 Microstructure Intelligence Engine tests (Milestone 4 addendum).

Verifies feature derivation from genuine SIP L1 quote/trade events, the
observed-vs-inferred separation, quality flagging (crossed / stale), and that
unknowns stay None. No L2 depth is fabricated (there is no depth field).
"""

from __future__ import annotations

from cengine.microstructure import (
    MicrostructureEngine,
    MicrostructureFeatures,
    SymbolMicrostructure,
)
from market_data.alpaca_stream import QuoteEvent, TradeEvent

BASE = 1_700_000_000_000_000_000  # ns
STEP = 100_000_000               # 100 ms


def _q(sym, ns, bid_ticks, ask_ticks, bid_size=500, ask_size=300):
    return QuoteEvent(sym, ns, bid_ticks, bid_size, "V", ask_ticks, ask_size, "V",
                      (), "C", 1, ns)


def _t(sym, ns, price_ticks, size):
    return TradeEvent(sym, ns, 1, "V", price_ticks, size, (), "C", 1, ns)


def _rising_buy_tape(eng, sym="AAA", n=60):
    px = 1_000_000  # $100 at scale 10_000
    sizes = [100, 300, 900, 200, 500, 700]
    for i in range(n):
        ns = BASE + i * STEP
        eng.on_quote_event(_q(sym, ns, px - 100, px + 100, bid_size=500, ask_size=300))
        eng.on_trade_event(_t(sym, ns + 1, px + 100, sizes[i % len(sizes)]))  # buy at ask
        px += 40
    return BASE + n * STEP


def test_rising_buy_tape_features():
    eng = MicrostructureEngine()
    now = _rising_buy_tape(eng)
    f = eng.features("AAA", now_ns=now)
    assert f is not None
    assert f.quality == "OK" and not f.is_stale and not f.crossed_or_locked
    # inferred flow signals
    assert f.order_flow_imbalance is not None and f.order_flow_imbalance > 0
    assert f.cumulative_volume_delta is not None and f.cumulative_volume_delta > 0
    assert f.trade_arrival_intensity is not None and f.trade_arrival_intensity > 0
    # microprice displacement: bid_size(500) > ask_size(300) -> microprice above mid
    assert f.microprice_displacement_bps is not None and f.microprice_displacement_bps > 0
    # price-impact metrics computed once enough variable-volume trades seen
    assert f.kyle_lambda is not None
    assert f.amihud_illiquidity is not None
    # adverse selection computed (buy aggressors in a rising tape)
    assert f.adverse_selection_bps is not None
    assert f.n_trades_window >= 50 and f.confidence > 0


def test_observed_vs_inferred_and_unknowns_none_before_data():
    book = SymbolMicrostructure("ZZZ")
    f = book.features(BASE / 1e9)
    # No data -> observed prices + inferred metrics all unknown (None), never fabricated
    assert f.bid is None and f.ask is None and f.mid is None
    assert f.kyle_lambda is None and f.amihud_illiquidity is None
    assert f.microprice is None and f.adverse_selection_bps is None
    assert f.is_stale is True
    # observed fields exist on the contract separate from inferred ones
    assert hasattr(f, "spread_bps") and hasattr(f, "order_flow_imbalance")
    # no L2 depth is modeled
    assert not hasattr(f, "depth") and not hasattr(f, "level2")


def test_crossed_quote_flagged():
    eng = MicrostructureEngine()
    ns = BASE
    # bid >= ask (crossed/locked); both positive so it is ingested
    eng.on_quote_event(_q("CRS", ns, 1_000_500, 1_000_000))
    f = eng.features("CRS", now_ns=ns + STEP)
    assert f.crossed_or_locked is True
    assert f.quality == "CROSSED"
    assert f.confidence == 0.0


def test_stale_when_queried_far_in_future():
    eng = MicrostructureEngine()
    now = _rising_buy_tape(eng, sym="STL", n=20)
    # query 60 seconds after the last quote -> stale
    f = eng.features("STL", now_ns=now + 60 * 1_000_000_000)
    assert f.is_stale is True
    assert f.quality == "STALE"


def test_fill_probability_feature_contract():
    eng = MicrostructureEngine()
    now = _rising_buy_tape(eng, sym="FPM")
    f = eng.features("FPM", now_ns=now)
    fp = f.fill_probability_features()
    for key in ("feature_version", "spread_bps", "quote_imbalance",
                "order_flow_imbalance", "microprice_displacement_bps",
                "trade_arrival_intensity", "aggressive_imbalance", "kyle_lambda",
                "amihud_illiquidity", "liquidity_resilience", "adverse_selection_bps",
                "is_stale", "confidence"):
        assert key in fp


def test_unknown_symbol_returns_none():
    eng = MicrostructureEngine()
    assert eng.features("NOPE", now_ns=BASE) is None


def test_features_is_immutable_snapshot():
    f = MicrostructureFeatures(symbol="X")
    try:
        f.bid = 1.0  # frozen dataclass
        raised = False
    except Exception:
        raised = True
    assert raised
