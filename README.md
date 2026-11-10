# Swizzle Python SDK

<div align="center">

### Official Python Client for Speculative LLM Streaming & Robotics Teleoperation

**Automatic Speculative Decoding · Dual Wire Compatibility · 80Hz Robotics Streaming**

[![PyPI](https://img.shields.io/pypi/v/swizzle-ai.svg?style=for-the-badge&color=blue)](https://pypi.org/project/swizzle-ai/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/swizzle-ai/)
[![Type Checked](https://img.shields.io/badge/Typing-mypy_strict-brightgreen?style=for-the-badge)](https://github.com/Swizzle-AI/swizzle-python-sdk)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](LICENSE)

</div>

---

## ⚡ Overview

`swizzle-ai` is the official Python client library for Swizzle AI. It provides:
1. **Drop-in Compatibility**: Fully compatible with standard OpenAI and Anthropic SDK conventions.
2. **Speculative Decoding Streaming**: Unpacks speculative candidate token bursts automatically, providing smooth token-per-second rendering at **$>98	ext{ tok/s}$**.
3. **Prefix Caching Cost Transparency**: Exposes exact cache hit percentages and cost savings in response usage metadata.
4. **Real-Time Robotics Telemetry**: Native helper classes for camera patch extraction, cubic spline smoothing, and 80Hz WebSocket teleoperation.

---

## 📦 Installation

```bash
pip install swizzle-ai
```

---

## 🚀 Usage Examples

### 1. OpenAI-Compatible Chat Completions
```python
from swizzle import SwizzleClient

client = SwizzleClient(
    base_url="https://swizzle-ten.vercel.app/v1",
    api_key="swizzle_live_free_tier"
)

response = client.chat.completions.create(
    model="qwen-2.5-coder-32b",
    messages=[
        {"role": "system", "content": "You are Swizzle, an expert AI systems engineer."},
        {"role": "user", "content": "Explain prefix caching and why AST-diff matters."}
    ],
    temperature=0.2
)

print(response.choices[0].message.content)
print(f"\nTokens: {response.usage.total_tokens} | Cache Hit: {response.usage.prompt_cache_hit_rate:.1%}")
```

---

### 2. Anthropic Messages Protocol
```python
from swizzle import SwizzleClient

client = SwizzleClient(
    base_url="https://swizzle-ten.vercel.app",
    api_key="swizzle_live_free_tier"
)

message = client.messages.create(
    model="claude-3-5-sonnet",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write an optimized CUDA kernel for vector addition."}]
)

print(message.content[0].text)
```

---

### 3. Robotics & Visual Servoing Streaming
```python
import cv2
from swizzle.robotics import extract_patch_intensities_from_frame, SplineSmoother, SwizzleTeleopClient

# Initialize camera and 50Hz spline smoother
cap = cv2.VideoCapture(0)
smoother = SplineSmoother(window_size=5)
teleop = SwizzleTeleopClient(gateway_url="wss://swizzle-ten.vercel.app/teleop")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract 14x14 ViT patch luminance deltas
    patches = extract_patch_intensities_from_frame(frame, grid_size=14)
    smoothed_action = smoother.update(patches)

    # Stream to Swizzle Robotics World Model
    teleop.send_telemetry(smoothed_action)
```

---

<div align="center">

**Part of the [Swizzle AI](https://github.com/Swizzle-AI) Ecosystem · Apache 2.0**

</div>
