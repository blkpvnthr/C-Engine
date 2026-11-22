# Execution Strategy Inventory (§16)

Factual audit of the ACTUAL strategy code paths — implementation, activation, and
test status. Activation is determined by what `python3 trade.py` (no env overrides)
really registers, not by the existence of a class.

Evidence: `reports/execution_strategy_audit.json`.

## Live signal path (the only one that produces orders)

`MarketEngine._handle_bar` (trade.py:1094) → `StrategyCoordinator.on_bar`
(strategies.py:661, runs every registered strategy per bar, concatenates
candidates) → `CandidateRouter.route` (trade.py:410) → `OrderManager.submit` →
native `RiskEngine`. `build_strategy_coordinator` (strategies.py:702) always builds
Turtle + DualMA + APO; DIBB only when `config.di_bb` is set.

The CandidateRouter now also applies the **exposure-authorization gate** (commit
4101622): geared/inverse entries are blocked unless authorized; exits never gated.

## Strategy table

| Strategy | file:line | Implemented | Default-on under `python3 trade.py` | Gate / params | Tests | Direction | Target family |
|---|---|---|---|---|---|---|---|
| TurtleStrategy | strategies.py:141 | Yes (Donchian breakout/exit, look-ahead safe) | **Yes** (always built) | `CENGINE_TURTLE_ENTRY_LOOKBACK/_EXIT_LOOKBACK`; no on/off | none dedicated | LONG+SHORT+exits | VOLATILITY_BREAKOUT |
| DualMAStrategy | strategies.py:264 | Yes (SMA crossover) | **Yes** (always built) | `CENGINE_DUAL_MA_FAST/_SLOW_PERIOD`; no on/off | none | LONG+SHORT+exits | MOMENTUM_CONTINUATION |
| APOMeanReversionStrategy | strategies.py:365 | Yes (EMA APO contrarian + warm-up guard) | **Yes** (always built) | `CENGINE_APO_*`; no on/off | none | LONG+SHORT+exits | MEAN_REVERSION |
| DIBBStrategy | strategies.py:523 | Yes (Wilder DMI + BB-mid + VWAP + momentum) | **No** (off) | `CENGINE_DIBB_ENABLED` (default 0) | tests/test_dibb_strategy.py | LONG-only (+EXIT_LONG; inverse-exit rule) | TREND_PULLBACK |

## Implemented but NOT on the order path

- **amps_v4_regime.py** — regime scoring + bull/bear/leveraged/inverse ETF registry; used only by the scanner/discovery/ranking layer to select subscriptions, NOT to emit orders. Encodes the long-inverse-ETF concept. (Tested indirectly via test_gpu_discovery.py.)
- **order_flow_engine.py** (`OrderFlowEngine`, exhaustion features) — instantiated only inside the scanner; produces ranking features, not TradeCandidates.
- **ProtectiveExitCoordinator** (protective_exit.py) — mandatory −0.5% stop backstop, not a signal strategy (tests/test_protective_exit.py).

## Not implemented (no code)

Standalone momentum-breakout, opening-range, pullback, DailyGreen, VWAP-continuation,
and multi-day-swing **strategies**. VWAP/momentum exist only as indicators.

## Target-family mapping + gaps

| Family | Existing fit | Gap |
|---|---|---|
| MOMENTUM_CONTINUATION | DualMAStrategy (crossover) | no dedicated momentum/order-flow entry |
| VOLATILITY_BREAKOUT | TurtleStrategy (Donchian) | opening-range variant missing |
| TREND_PULLBACK | DIBBStrategy (off by default) | enable + validate |
| MEAN_REVERSION | APOMeanReversionStrategy | recovery-confirmed variant to validate (§21) |
| MULTI_DAY_SWING | none | not implemented |

## Risks flagged

- Turtle / DualMA / APO are **always on with no individual disable** and have **no
  dedicated unit tests** — a consolidation to the five families should add per-family
  enable flags and tests before changing default behavior.
- DIBB (the closest TREND_PULLBACK fit) is off by default.
- Inverse/leveraged participation today is only at the scanner eligibility layer
  (AMPS), not in the order-emitting strategies.

Consolidation into the five families must preserve existing validated behavior until
replacements are independently evaluated (directive §72).
