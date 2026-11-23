"""GPU/CPU benchmark certification for the learning layer (Milestone 6 addendum §19)."""

from cengine.learning.benchmarks.gpu_benchmarks import (
    BENCHMARK_VERSION,
    benchmark_inference_latency,
    benchmark_memory,
    benchmark_training_throughput,
    run_all_benchmarks,
)

__all__ = [
    "BENCHMARK_VERSION",
    "benchmark_inference_latency",
    "benchmark_memory",
    "benchmark_training_throughput",
    "run_all_benchmarks",
]
