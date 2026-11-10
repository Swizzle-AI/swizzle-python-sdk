"""
Chat completions interface implementation for Swizzle.
"""

import json
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator, Union
import httpx

from .exceptions import SwizzleError, AuthenticationError, RateLimitError
from .types import ChatCompletion, ChatCompletionChunk, SwizzleTelemetry


def parse_telemetry(headers: httpx.Headers) -> SwizzleTelemetry:
    """Extracts Swizzle hardware, cache, speculative, and AST telemetry headers."""
    ttft_raw = headers.get("x-swizzle-ttft-ms")
    ttft_val = int(ttft_raw) if ttft_raw and ttft_raw.isdigit() else None

    hit_raw = headers.get("x-swizzle-cache-hit-tokens")
    hit_val = int(hit_raw) if hit_raw and hit_raw.isdigit() else None

    uncached_raw = headers.get("x-swizzle-uncached-tokens")
    uncached_val = int(uncached_raw) if uncached_raw and uncached_raw.isdigit() else None

    saved_raw = headers.get("x-swizzle-tokens-saved")
    saved_val = int(saved_raw) if saved_raw and saved_raw.isdigit() else None

    return SwizzleTelemetry(
        ttft_ms=ttft_val,
        cache_hit_tokens=hit_val,
        uncached_tokens=uncached_val,
        tokens_saved=saved_val,
        compression_ratio=headers.get("x-swizzle-compression-ratio"),
        speculative_engine=headers.get("x-swizzle-speculative-engine"),
        speculative_speedup=headers.get("x-swizzle-speculative-speedup"),
        speculative_acceptance=headers.get("x-swizzle-speculative-acceptance"),
        effective_tps=headers.get("x-swizzle-effective-tps"),
        ast_reuse_pct=headers.get("x-swizzle-ast-reuse-pct"),
        ast_diff_blocks=headers.get("x-swizzle-ast-diff-blocks"),
        pd_mode=headers.get("x-swizzle-pd-mode"),
        worker_node=headers.get("x-swizzle-worker-node"),
        hardware=headers.get("x-swizzle-hardware"),
        cost_usd=headers.get("x-swizzle-cost-usd"),
        saved_usd=headers.get("x-swizzle-saved-usd"),
    )


class Completions:
    """Synchronous chat completions resource."""
    def __init__(self, client):
        self._client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools

        url = f"{self._client.base_url.rstrip('/')}/chat/completions"

        if not stream:
            resp = self._client._http_client.post(url, json=payload, headers=self._client._auth_headers())
            self._handle_error(resp)
            data = resp.json()
            completion = ChatCompletion.model_validate(data)
            completion.telemetry = parse_telemetry(resp.headers)
            return completion

        return self._stream_generator(url, payload)

    def _stream_generator(self, url: str, payload: Dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        with self._client._http_client.stream("POST", url, json=payload, headers=self._client._auth_headers()) as resp:
            self._handle_error(resp)
            telemetry = parse_telemetry(resp.headers)

            for line in resp.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    content = line[6:].strip()
                    if content == "[DONE]":
                        break
                    try:
                        data = json.loads(content)
                        chunk = ChatCompletionChunk.model_validate(data)
                        chunk.telemetry = telemetry
                        yield chunk
                    except Exception:
                        continue

    def _handle_error(self, resp: httpx.Response):
        if resp.status_code == 401:
            raise AuthenticationError("Invalid or revoked Swizzle API key", status_code=401, response_body=resp.text)
        elif resp.status_code == 429:
            raise RateLimitError("Swizzle rate limit or token bucket exhausted", status_code=429, response_body=resp.text)
        elif resp.status_code >= 400:
            raise SwizzleError(f"HTTP {resp.status_code}: {resp.text}", status_code=resp.status_code, response_body=resp.text)


class AsyncCompletions:
    """Asynchronous chat completions resource."""
    def __init__(self, client):
        self._client = client

    async def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools

        url = f"{self._client.base_url.rstrip('/')}/chat/completions"

        if not stream:
            resp = await self._client._http_client.post(url, json=payload, headers=self._client._auth_headers())
            self._handle_error(resp)
            data = resp.json()
            completion = ChatCompletion.model_validate(data)
            completion.telemetry = parse_telemetry(resp.headers)
            return completion

        return self._async_stream_generator(url, payload)

    async def _async_stream_generator(self, url: str, payload: Dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        async with self._client._http_client.stream("POST", url, json=payload, headers=self._client._auth_headers()) as resp:
            self._handle_error(resp)
            telemetry = parse_telemetry(resp.headers)

            async for line in resp.aiter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    content = line[6:].strip()
                    if content == "[DONE]":
                        break
                    try:
                        data = json.loads(content)
                        chunk = ChatCompletionChunk.model_validate(data)
                        chunk.telemetry = telemetry
                        yield chunk
                    except Exception:
                        continue

    def _handle_error(self, resp: httpx.Response):
        if resp.status_code == 401:
            raise AuthenticationError("Invalid or revoked Swizzle API key", status_code=401, response_body=resp.text)
        elif resp.status_code == 429:
            raise RateLimitError("Swizzle rate limit or token bucket exhausted", status_code=429, response_body=resp.text)
        elif resp.status_code >= 400:
            raise SwizzleError(f"HTTP {resp.status_code}: {resp.text}", status_code=resp.status_code, response_body=resp.text)

