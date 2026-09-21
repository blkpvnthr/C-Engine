import _cengine_native as native


def test_native_risk_kill_switch_is_authoritative():
    engine = native.RiskEngine()
    engine.activate_kill_switch()
    request = native.RiskRequest()
    request.order_id = 1
    request.symbol = "AAPL"
    request.side = native.Side.BUY
    request.quantity = 1
    request.price_ticks = 100
    request.now_ns = 10
    assert not engine.evaluate(request).accepted


def test_native_limit_order_book_matches_crossing_orders():
    book = native.LimitOrderBook("AAPL")
    assert book.add_limit(1, "AAPL", native.Side.SELL, 2, 101) == []
    executions = book.add_limit(2, "AAPL", native.Side.BUY, 2, 101)
    assert len(executions) == 1
    assert executions[0].quantity == 2
