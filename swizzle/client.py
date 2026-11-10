"""
Core client classes for Swizzle.
"""

import os
from typing import Optional, Dict, Any, List
import httpx

from .completions import Completions, AsyncCompletions


DEFAULT_BASE_URL = "https://swizzle-ten.vercel.app/v1"


class ChatNamespace:
    def __init__(self, client):
        self.completions = Completions(client)


class AsyncChatNamespace:
    def __init__(self, client):
        self.completions = AsyncCompletions(client)


class ModelsNamespace:
    def __init__(self, client):
        self._client = client

    def list(self) -> List[Dict[str, Any]]:
        url = f"{self._client.base_url.rstrip('/')}/models"
        resp = self._client._http_client.get(url, headers=self._client._auth_headers())
        if resp.status_code >= 400:
            resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])


class AsyncModelsNamespace:
    def __init__(self, client):
        self._client = client

    async def list(self) -> List[Dict[str, Any]]:
        url = f"{self._client.base_url.rstrip('/')}/models"
        resp = await self._client._http_client.get(url, headers=self._client._auth_headers())
        if resp.status_code >= 400:
            resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])


class Swizzle:
    """
    Synchronous Swizzle Client.
    
    Usage:
        client = Swizzle(api_key="sk-swz-live-...")
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            messages=[{"role": "user", "content": "Write quicksort in Go"}]
        )
        print("TTFT:", completion.telemetry.ttft_ms)
        print(completion.choices[0].message.content)
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("SWIZZLE_API_KEY", "swz_live_free_tier")
        self.base_url = (base_url or os.environ.get("SWIZZLE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        if not self.base_url.endswith("/v1") and not "/v1" in self.base_url:
            self.base_url += "/v1"
        self._http_client = httpx.Client(timeout=timeout)
        self.chat = ChatNamespace(self)
        self.models = ModelsNamespace(self)

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-API-Key": self.api_key,
            "User-Agent": "swizzle-py/0.1.0",
        }

    def close(self):
        self._http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncSwizzle:
    """
    Asynchronous Swizzle Client.
    
    Usage:
        client = AsyncSwizzle(api_key="sk-swz-live-...")
        completion = await client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            messages=[{"role": "user", "content": "Write quicksort in Go"}]
        )
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("SWIZZLE_API_KEY", "swz_live_free_tier")
        self.base_url = (base_url or os.environ.get("SWIZZLE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        if not self.base_url.endswith("/v1") and not "/v1" in self.base_url:
            self.base_url += "/v1"
        self._http_client = httpx.AsyncClient(timeout=timeout)
        self.chat = AsyncChatNamespace(self)
        self.models = AsyncModelsNamespace(self)

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-API-Key": self.api_key,
            "User-Agent": "swizzle-py/0.1.0",
        }

    async def close(self):
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

