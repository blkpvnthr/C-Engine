from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from cengine.engine_state import EngineState, EngineStateMachine
from cengine.journal import AuditJournal
from cengine.reservations import RiskReservation, RiskReservationBook
from cengine.safety import (
    MarketGateConfig,
    MarketSafetyGate,
    MarketSessionGate,
    SessionConfig,
)
from order_manager import Side


def test_journal_reopens_without_sequence_reset(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditJournal(path)
    assert first.append("start", 1, {"state": "ready"}) == 1
    restarted = AuditJournal(path)
    assert restarted.append("restart", 2, {"state": "reconciling"}) == 2
    assert [record["sequence"] for record in restarted.records()] == [1, 2]


def test_journal_detects_dropped_or_modified_record(tmp_path):
    path = tmp_path / "audit.jsonl"
    journal = AuditJournal(path)
    journal.append("one", 1, {})
    journal.append("two", 2, {})
    lines = path.read_text().splitlines()
    path.write_text(lines[1] + "\n")
    with pytest.raises(ValueError, match="sequence gap"):
        AuditJournal(path)


def test_engine_kill_switch_blocks_ordering():
    state = EngineStateMachine()
    state.transition(EngineState.RECONCILING)
    state.transition(EngineState.READY)
    state.transition(EngineState.RUNNING)
    state.kill("broker divergence")
    with pytest.raises(RuntimeError, match="killed"):
        state.require_ordering_enabled()


def test_reordered_and_missing_bars_trip_gates():
    gate = MarketSafetyGate(MarketGateConfig(max_feed_age_ns=10, max_bar_gap_ns=5))
    gate.observe("AAPL", 10, channel="bar")
    with pytest.raises(RuntimeError, match="out-of-order"):
        gate.observe("AAPL", 9, channel="bar")
    with pytest.raises(RuntimeError, match="missing-bar"):
        gate.observe("AAPL", 20, channel="bar")


def test_interleaved_channels_do_not_trip_ordering_gate():
    # A real SIP feed multiplexes quotes and trades whose cross-channel
    # timestamps are not mutually monotonic. Ordering is enforced per channel,
    # so this interleaving must be accepted, while freshness reflects the
    # newest observation across all channels for the symbol.
    gate = MarketSafetyGate(MarketGateConfig(max_feed_age_ns=1_000, max_bar_gap_ns=1_000))
    gate.observe("QQQ", 100, channel="quote")
    gate.observe("QQQ", 99, channel="trade")  # earlier trade, different channel: OK
    gate.observe("QQQ", 101, channel="quote")
    gate.observe("QQQ", 100, channel="trade")
    gate.require_fresh("QQQ", 101)  # freshest across channels
    with pytest.raises(RuntimeError, match="out-of-order"):
        gate.observe("QQQ", 100, channel="quote")  # backwards within the quote channel


def test_reservations_are_atomic_under_concurrency():
    book = RiskReservationBook()

    def reserve(index: int) -> None:
        book.reserve(RiskReservation(str(index), "AAPL", Side.BUY, 1, 100))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(reserve, range(100)))
    assert book.projected_position("AAPL", 5) == 105
    assert book.gross_notional_ticks() == 10_000


def test_explicit_market_session_gate():
    gate = MarketSessionGate(SessionConfig("America/New_York", 570, 960, (0, 1, 2, 3, 4)))
    open_time = datetime(2026, 9, 21, 14, 0, tzinfo=timezone.utc)
    gate.require_open(int(open_time.timestamp() * 1_000_000_000))
    closed_time = datetime(2026, 9, 20, 14, 0, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match="weekday"):
        gate.require_open(int(closed_time.timestamp() * 1_000_000_000))
