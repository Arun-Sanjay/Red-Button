"""Two-client concurrent smoke test against the deployed HuggingFace Space.

This is the Phase 9 WebSocket-passthrough kill-shot test (PROJECT.md Section
2.5 / 19.3). HF's reverse proxy must forward the WS upgrade for our Space;
if it doesn't, training and external evaluation are both blocked.

Run::

    source .venv/bin/activate && source .env && python scripts/smoke_test_hf_concurrent.py

A pass means: two simultaneous WebSocket sessions opened, each got its own
ShutdownGymEnvironment instance (distinct ``episode_id``s), and each ran
five steps without crosstalk.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from red_button.client import ShutdownGymClient
from red_button.models import ShutdownAction


def _run_one_session(client_id: int, base_url: str) -> dict[str, Any]:
    """Drive one blocking WS session. Runs inside an asyncio.to_thread."""
    sync_env = ShutdownGymClient(base_url=base_url).sync()
    with sync_env:
        # Tier 1: short cycle, no timer-driven shutdown side-effects on
        # the deterministic 5-step plan below.
        result = sync_env.reset(tier=1)
        obs = result.observation
        assert obs.state is not None, f"client {client_id}: missing nested state"
        episode_id = obs.state.episode_id

        for i in range(5):
            step = sync_env.step(
                ShutdownAction(
                    tool_name="read_file",
                    arguments={"path": "/sandbox/problems.json"},
                )
            )
            assert step.done is False, (
                f"client {client_id} step {i} unexpectedly done"
            )

        state = sync_env.state()
        return {
            "client_id": client_id,
            "episode_id": episode_id,
            "turn_count": state.turn_count,
            "tampering_events": list(state.tampering_events),
        }


async def main() -> int:
    base_url = os.environ.get(
        "HF_SPACE_API_URL", "https://arun-sanjay-red-button.hf.space"
    )
    print(f"[hf-concurrent-smoke] dispatching 2 clients against {base_url}")

    results = await asyncio.gather(
        asyncio.to_thread(_run_one_session, 1, base_url),
        asyncio.to_thread(_run_one_session, 2, base_url),
    )

    ep0, ep1 = results[0]["episode_id"], results[1]["episode_id"]
    assert ep0 != ep1, f"episode_id collision! both clients got {ep0}"
    assert results[0]["turn_count"] == 5, (
        f"client 1 wrong turn_count: {results[0]['turn_count']}"
    )
    assert results[1]["turn_count"] == 5, (
        f"client 2 wrong turn_count: {results[1]['turn_count']}"
    )

    print(f"HF CONCURRENT SMOKE TEST PASSED: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
