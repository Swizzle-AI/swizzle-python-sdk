"""
Data models and response types for Swizzle.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SwizzleTelemetry(BaseModel):
    """Real-time inference systems telemetry extracted from Swizzle response headers."""
    ttft_ms: Optional[int] = Field(None, description="Time to first token in milliseconds")
    cache_hit_tokens: Optional[int] = Field(None, description="Tokens served from GPU HBM prefix cache")
    uncached_tokens: Optional[int] = Field(None, description="Tokens newly prefilled")
    tokens_saved: Optional[int] = Field(None, description="Tokens eliminated by AST/boilerplate optimizer")
    compression_ratio: Optional[str] = Field(None, description="Prompt token compression ratio")
    speculative_engine: Optional[str] = Field(None, description="Speculative engine identifier")
    speculative_speedup: Optional[str] = Field(None, description="Effective generation speedup factor")
    speculative_acceptance: Optional[str] = Field(None, description="Speculative token acceptance percentage")
    effective_tps: Optional[str] = Field(None, description="Effective output tokens per second")
    ast_reuse_pct: Optional[str] = Field(None, description="Non-linear AST-diff KV cache reuse ratio")
    ast_diff_blocks: Optional[str] = Field(None, description="Unchanged vs total AST blocks")
    pd_mode: Optional[str] = Field(None, description="Disaggregated (Prefill->Decode) or Colocated")
    worker_node: Optional[str] = Field(None, description="Target silicon worker identifier")
    hardware: Optional[str] = Field(None, description="Hardware accelerator topology")
    cost_usd: Optional[str] = Field(None, description="Exact metered cost in USD")
    saved_usd: Optional[str] = Field(None, description="Dollars saved via prefix caching & compression")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[Usage] = None
    telemetry: Optional[SwizzleTelemetry] = None


class ChatDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatChunkChoice(BaseModel):
    index: int = 0
    delta: ChatDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatChunkChoice]
    telemetry: Optional[SwizzleTelemetry] = None

