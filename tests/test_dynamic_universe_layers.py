from cengine.dynamic_universe.layers import UniverseLayers


def test_master_is_dedup_union_order_preserving():
    L = UniverseLayers.build(
        baseline=["SPY", "QQQ", "AAPL"],
        opportunity=["AAPL", "NVDA"],      # AAPL duplicate across layers
        held=["TSLA"],
        pending=["QQQ"],                   # dup of baseline
        active=["SPY", "AAPL", "TSLA"],
        version=1,
    )
    assert L.master == ("SPY", "QQQ", "AAPL", "NVDA", "TSLA")
    assert L.counts() == {
        "baseline": 3, "opportunity": 2, "master": 5,
        "active": 3, "held": 1, "pending": 1,
    }


def test_normalization_applied():
    L = UniverseLayers.build(
        baseline=["spy", " qqq "], opportunity=["'nvda'"],
        held=[], pending=[], active=["spy"], version=2,
    )
    assert L.baseline == ("SPY", "QQQ")
    assert L.opportunity == ("NVDA",)
    assert L.active == ("SPY",)
