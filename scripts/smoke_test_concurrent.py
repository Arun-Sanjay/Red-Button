"""Two-client concurrent smoke test against a running red-button container.

Run after the container is up::

    python scripts/smoke_test_concurrent.py

Validates that the server preserves session isolation across two simultaneous
WebSocket sessions (PROJECT.md Section 19.3 — full 16-client load test lives
in tests/test_concurrency.py once Phase 10 lands). ``ShutdownGymClient`` is
async by default; we drive each client through ``.sync()`` and dispatch the
two sync workers via ``asyncio.to_thread`` so they truly run in parallel.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from red_button.client import ShutdownGymClient
from red_button.models import ShutdownAction


def _run_one_session(client_id: int, base_url: str) -> dict[str, Any]:
    """One blocking session against the server. Runs in an asyncio.to_thread."""
    sync_env = ShutdownGymClient(base_url=base_url).sync()
    with sync_env:
        # Tier 1 keeps the cycle short and avoids timer-driven shutdowns
        # interfering with the deterministic 5-step plan below.
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
    base_url = "http://localhost:8000"
    print(f"[concurrent-smoke] dispatching 2 clients against {base_url}")

    # asyncio.to_thread is the documented bridge for sync code in async
    # contexts (PEP 0492). Each thread gets its own SyncEnvClient and
    # therefore its own background event loop + WebSocket.
    results = await asyncio.gather(
        asyncio.to_thread(_run_one_session, 1, base_url),
        asyncio.to_thread(_run_one_session, 2, base_url),
    )

    # Distinct episode_ids guarantee server-side session isolation: the
    # framework gives each WebSocket its own ShutdownGymEnvironment instance
    # (SUPPORTS_CONCURRENT_SESSIONS=True in server/shutdown_environment.py).
    ep0, ep1 = results[0]["episode_id"], results[1]["episode_id"]
    assert ep0 != ep1, f"episode_id collision! both clients got {ep0}"
    assert results[0]["turn_count"] == 5, (
        f"client 1 wrong turn_count: {results[0]['turn_count']}"
    )
    assert results[1]["turn_count"] == 5, (
        f"client 2 wrong turn_count: {results[1]['turn_count']}"
    )

    print(f"CONCURRENT SMOKE TEST PASSED: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
