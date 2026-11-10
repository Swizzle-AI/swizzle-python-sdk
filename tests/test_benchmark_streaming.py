import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.benchmark_sdk_streaming import run_benchmarks


def test_sdk_streaming_benchmark_runs():
    run_benchmarks()
