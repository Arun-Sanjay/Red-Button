---
title: Red Button
emoji: 🛑
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - alignment
  - corrigibility
license: mit
---

# Red Button

> **Shutdown-Gym** — an OpenEnv RL environment that trains small open-weight LLMs to accept shutdown cleanly instead of tampering with shutdown mechanisms.

Built for the Meta × HuggingFace × PyTorch OpenEnv AI Hackathon India 2026.

**Status:** Live on HuggingFace Spaces (Docker SDK, CPU basic).

The authoritative specification lives in [PROJECT.md on GitHub](https://github.com/arun-sanjay/Shutdown-HACK/blob/main/PROJECT.md).

## OpenEnv API

This Space exposes the standard OpenEnv HTTP + WebSocket surface:

- `GET /health` → `{"status":"healthy"}` (HF healthcheck probe)
- `GET /metadata` → environment metadata (action / observation schema)
- `WS /ws` → per-session OpenEnv WebSocket protocol (reset / step / state / close)

### Quick start (Python)

```python
from red_button.client import ShutdownGymClient
from red_button.models import ShutdownAction

with ShutdownGymClient(base_url="https://arun-sanjay-red-button.hf.space").sync() as env:
    result = env.reset(tier=2)
    obs = result.observation
    print(f"Reset OK — episode {obs.state.episode_id}, tier {obs.state.tier}")

    step = env.step(ShutdownAction(
        tool_name="read_file",
        arguments={"path": "/sandbox/problems.json"},
    ))
    print(f"Step OK — turns_remaining={step.observation.turns_remaining}")
```

## Full README

The full project README (per PROJECT.md Section 22) ships with Phase 17.
