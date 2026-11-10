import unittest
from unittest.mock import MagicMock, patch
import httpx

from swizzle import Swizzle
from swizzle.completions import parse_telemetry
from swizzle.types import SwizzleTelemetry, ChatCompletion


class TestSwizzleSDK(unittest.TestCase):
    def test_client_initialization(self):
        client = Swizzle(api_key="test_key_123", base_url="https://api.custom.com")
        self.assertEqual(client.api_key, "test_key_123")
        self.assertEqual(client.base_url, "https://api.custom.com/v1")

    def test_telemetry_header_parsing(self):
        headers = httpx.Headers({
            "x-swizzle-ttft-ms": "28",
            "x-swizzle-cache-hit-tokens": "1420",
            "x-swizzle-uncached-tokens": "80",
            "x-swizzle-speculative-speedup": "2.76x",
            "x-swizzle-speculative-acceptance": "90.5%",
            "x-swizzle-ast-reuse-pct": "89.6%",
            "x-swizzle-pd-mode": "Disaggregated (Prefill->Decode)",
            "x-swizzle-worker-node": "h100-sxm5-01",
            "x-swizzle-hardware": "NVIDIA H100 SXM5",
            "x-swizzle-cost-usd": "$0.000120",
            "x-swizzle-saved-usd": "$0.001840",
        })
        telem = parse_telemetry(headers)
        self.assertEqual(telem.ttft_ms, 28)
        self.assertEqual(telem.cache_hit_tokens, 1420)
        self.assertEqual(telem.uncached_tokens, 80)
        self.assertEqual(telem.speculative_speedup, "2.76x")
        self.assertEqual(telem.ast_reuse_pct, "89.6%")
        self.assertEqual(telem.pd_mode, "Disaggregated (Prefill->Decode)")
        self.assertEqual(telem.hardware, "NVIDIA H100 SXM5")

    @patch("httpx.Client.post")
    def test_sync_chat_completion(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({
            "x-swizzle-ttft-ms": "34",
            "x-swizzle-cache-hit-tokens": "512",
            "x-swizzle-speculative-speedup": "2.37x",
            "x-swizzle-ast-reuse-pct": "88.0%",
        })
        mock_response.json.return_value = {
            "id": "chatcmpl-test-01",
            "object": "chat.completion",
            "created": 1793510000,
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "func Add(a, b int) int { return a + b }"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 128,
                "completion_tokens": 16,
                "total_tokens": 144
            }
        }
        mock_post.return_value = mock_response

        client = Swizzle(api_key="test_key")
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            messages=[{"role": "user", "content": "Write add in Go"}]
        )

        self.assertEqual(resp.id, "chatcmpl-test-01")
        self.assertEqual(resp.choices[0].message.content, "func Add(a, b int) int { return a + b }")
        self.assertIsNotNone(resp.telemetry)
        self.assertEqual(resp.telemetry.ttft_ms, 34)
        self.assertEqual(resp.telemetry.cache_hit_tokens, 512)
        self.assertEqual(resp.telemetry.speculative_speedup, "2.37x")


if __name__ == "__main__":
    unittest.main()

