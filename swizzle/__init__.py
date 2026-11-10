"""
Swizzle Python SDK
~~~~~~~~~~~~~~~~~~
Ultra-low latency, prefix-cached, speculative LLM, Robotics VLA, and World-Model inference client.
"""

from .client import Swizzle, AsyncSwizzle
from .exceptions import SwizzleError, AuthenticationError, RateLimitError, WorkerUnavailableError
from .types import SwizzleTelemetry, ChatCompletion, ChatCompletionChunk
from .robotics import (
    HardwareDetector,
    StudentHardwareProfile,
    PatchCacheFilter,
    KinematicSpecActPredictor,
    TemporalActionEnsembler,
    VLAPolicy,
    VLAActionOutput,
    extract_patch_intensities_from_frame,
    quantize_action_to_token,
    dequantize_token_to_action,
)
from .world import WorldPlanner, WorldPlanResult

__version__ = "0.2.0"
__all__ = [
    "Swizzle",
    "AsyncSwizzle",
    "SwizzleError",
    "AuthenticationError",
    "RateLimitError",
    "WorkerUnavailableError",
    "SwizzleTelemetry",
    "ChatCompletion",
    "ChatCompletionChunk",
    "HardwareDetector",
    "StudentHardwareProfile",
    "PatchCacheFilter",
    "KinematicSpecActPredictor",
    "TemporalActionEnsembler",
    "VLAPolicy",
    "VLAActionOutput",
    "extract_patch_intensities_from_frame",
    "WorldPlanner",
    "WorldPlanResult",
    "quantize_action_to_token",
    "dequantize_token_to_action",
]
