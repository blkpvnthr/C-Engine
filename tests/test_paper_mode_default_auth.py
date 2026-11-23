"""P0.3 (follow-up): paper mode auto-enables PAPER_RESEARCH out-of-the-box.

`python3 trade.py` (zero-arg -> paper mode) should submit paper orders without
extra flags, while an explicit CENGINE_STRATEGY_EXECUTION_AUTH is always honored
(including setting it to a disabling value), and non-paper modes never auto-enable.
"""

from __future__ import annotations

from trade import _default_strategy_auth_for_mode


def test_paper_mode_defaults_to_paper_research_when_unset():
    assert _default_strategy_auth_for_mode("paper", None) == "paper_research"


def test_explicit_value_is_always_honored():
    # Any explicitly-set value (incl. an explicit empty string) suppresses the
    # default so the operator's choice -- including disabling -- always wins.
    assert _default_strategy_auth_for_mode("paper", "none") is None
    assert _default_strategy_auth_for_mode("paper", "mock") is None
    assert _default_strategy_auth_for_mode("paper", "") is None


def test_non_paper_modes_never_auto_enable():
    assert _default_strategy_auth_for_mode("live", None) is None
    assert _default_strategy_auth_for_mode("shadow", None) is None
