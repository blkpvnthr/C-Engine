"""High-level Metal indicator backend."""

from __future__ import annotations

from cengine.metal_buffers import MetalBuffers
from cengine.metal_runtime import MetalRuntime


class MetalIndicators:
    def __init__(self, runtime: MetalRuntime) -> None:
        self.runtime = runtime
        self.buffers = MetalBuffers(runtime.device)

    def sma(self, prices: list[int], window: int) -> list[int]:
        if not prices:
            raise ValueError("prices cannot be empty")

        if window <= 0:
            raise ValueError("window must be positive")

        count = len(prices)

        prices_buf = self.buffers.int64(prices)
        output_buf = self.buffers.zero_int64(count)
        count_buf = self.buffers.uint32(count)
        window_buf = self.buffers.uint32(window)

        self.runtime.dispatch(
            "rolling_sma_ticks",
            [
                prices_buf,
                output_buf,
                count_buf,
                window_buf,
            ],
            element_count=count,
        )

        return self.buffers.read_int64(output_buf, count)

    def ema(
        self,
        prices: list[int],
        alpha_numerator: int,
        alpha_denominator: int,
    ) -> list[int]:

        if not prices:
            raise ValueError("prices cannot be empty")

        if alpha_denominator <= 0:
            raise ValueError("alpha_denominator must be positive")

        if not 0 <= alpha_numerator <= alpha_denominator:
            raise ValueError(
                "alpha_numerator must be between 0 and alpha_denominator"
            )

        count = len(prices)

        prices_buf = self.buffers.int64(prices)
        output_buf = self.buffers.zero_int64(count)
        count_buf = self.buffers.uint32(count)
        alpha_num_buf = self.buffers.uint32(alpha_numerator)
        alpha_den_buf = self.buffers.uint32(alpha_denominator)

        # EMA is recursive. The Metal kernel itself walks the complete
        # series sequentially, so exactly one GPU thread is dispatched.
        self.runtime.dispatch(
            "exponential_moving_average_ticks",
            [
                prices_buf,
                output_buf,
                count_buf,
                alpha_num_buf,
                alpha_den_buf,
            ],
            element_count=1,
        )

        return self.buffers.read_int64(output_buf, count)

    def apo(
        self,
        fast_ema: list[int],
        slow_ema: list[int],
    ) -> list[int]:

        if not fast_ema or len(fast_ema) != len(slow_ema):
            raise ValueError(
                "fast_ema and slow_ema must have equal non-zero lengths"
            )

        count = len(fast_ema)

        fast_buf = self.buffers.int64(fast_ema)
        slow_buf = self.buffers.int64(slow_ema)
        output_buf = self.buffers.zero_int64(count)
        count_buf = self.buffers.uint32(count)

        self.runtime.dispatch(
            "average_price_oscillator_ticks",
            [
                fast_buf,
                slow_buf,
                output_buf,
                count_buf,
            ],
            element_count=count,
        )

        return self.buffers.read_int64(output_buf, count)

    def true_range(
        self,
        highs: list[int],
        lows: list[int],
        previous_closes: list[int],
    ) -> list[int]:

        count = len(highs)

        if (
            count == 0
            or len(lows) != count
            or len(previous_closes) != count
        ):
            raise ValueError(
                "highs, lows, and previous_closes must have equal non-zero lengths"
            )

        high_buf = self.buffers.int64(highs)
        low_buf = self.buffers.int64(lows)
        close_buf = self.buffers.int64(previous_closes)
        output_buf = self.buffers.zero_int64(count)
        count_buf = self.buffers.uint32(count)

        self.runtime.dispatch(
            "true_range_ticks",
            [
                high_buf,
                low_buf,
                close_buf,
                output_buf,
                count_buf,
            ],
            element_count=count,
        )

        return self.buffers.read_int64(output_buf, count)
