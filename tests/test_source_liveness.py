from pathlib import Path

import liquidity
import native_adapters
import order_manager
import run
import strategies


def test_top_level_modules_load_from_repository() -> None:
    repo = Path(__file__).resolve().parents[1]

    modules = (
        run,
        order_manager,
        strategies,
        liquidity,
        native_adapters,
    )

    for module in modules:
        path = Path(module.__file__).resolve()

        assert path.is_relative_to(repo), (
            f"{module.__name__} loaded from shadow location {path}; "
            f"expected source beneath {repo}"
        )
