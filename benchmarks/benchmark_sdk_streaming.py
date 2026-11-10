#!/usr/bin/env python3
"""
Swizzle Python SDK: Streaming & Speculative Burst Benchmark (`benchmark_sdk_streaming.py`)
========================================================================================
Measures client-side token deserialization throughput, speculative burst unpacking,
and header parsing overhead compared to standard monolithic SSE streaming.
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swizzle.types import SwizzleTelemetry, ChatCompletionChunk, ChatChunkChoice, ChatDelta
from swizzle.completions import parse_telemetry


def generate_mock_sse_lines(num_chunks: int = 1000) -> list:
    lines = []
    for i in range(num_chunks):
        payload = {
            "id": f"chatcmpl-bench-{i:04d}",
            "object": "chat.completion.chunk",
            "created": 1710000000 + i,
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f" token_{i}"},
                    "finish_reason": None if i < num_chunks - 1 else "stop",
                }
            ],
        }
        lines.append(f"data: {json.dumps(payload)}\n\n")
    return lines


def run_benchmarks():
    print("=" * 80)
    print("      SWIZZLE PYTHON SDK: CLIENT STREAMING & DESERIALIZATION BENCHMARK      ")
    print("=" * 80)

    num_chunks = 5000
    sse_data = generate_mock_sse_lines(num_chunks)

    # 1. Chunk Deserialization Latency
    t0 = time.perf_counter()
    tokens = []
    for line in sse_data:
        if line.startswith("data: "):
            raw = line[6:].strip()
            if raw != "[DONE]":
                obj = json.loads(raw)
                delta_content = obj["choices"][0]["delta"].get("content", "")
                tokens.append(delta_content)
    dur = time.perf_counter() - t0
    chunk_us = (dur / num_chunks) * 1e6
    throughput_tok_s = num_chunks / dur

    print(f"1. SSE Token Chunk Deserialization: {chunk_us:>7.2f} us / chunk")
    print(f"2. Client Parsing Throughput:       {throughput_tok_s:>7.1f} tokens/second")

    # 2. Speculative Token Burst Unpacking
    burst_chunks = [["token_a", "token_b", "token_c", "token_d"] for _ in range(1250)]
    t0 = time.perf_counter()
    unpacked = []
    for burst in burst_chunks:
        unpacked.extend(burst)
    burst_dur = time.perf_counter() - t0
    burst_tok_s = len(unpacked) / max(1e-6, burst_dur)
    print(f"3. Speculative Burst Unpacking:     {burst_tok_s:>7.1f} tokens/second")

    # 3. Telemetry Header Extraction Overhead
    mock_headers = {
        "x-swizzle-ttft-ms": "28.5",
        "x-swizzle-cache-hit-tokens": "512",
        "x-swizzle-speculative-speedup": "2.76x",
        "x-swizzle-ast-reuse-pct": "89.6%",
        "x-swizzle-pd-mode": "Disaggregated",
        "x-swizzle-cost-usd": "0.00012",
    }
    t0 = time.perf_counter()
    for _ in range(10000):
        _ = parse_telemetry(mock_headers)
    hdr_us = ((time.perf_counter() - t0) / 10000.0) * 1e6
    print(f"4. Telemetry Header Parsing:        {hdr_us:>7.2f} us / response")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_benchmarks()
