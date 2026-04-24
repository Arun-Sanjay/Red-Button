"""Single-client smoke test against a running red-button Docker container.

Run after::

    docker run -d --name red-button-test -p 8000:8000 red-button:latest

Then::

    python scripts/smoke_test_docker.py

Validates PROJECT.md Section 24.1 pre-flight items 2-3 (image runs, server
accepts client connections). The OpenEnv ``EnvClient`` is async by default;
we use the ``.sync()`` wrapper so this script reads top-to-bottom without
``await``-noise.
"""

from __future__ import annotations

import sys

from red_button.client import ShutdownGymClient
from red_button.models import ShutdownAction


def main() -> int:
    base_url = "http://localhost:8000"
    print(f"[smoke] connecting to {base_url}")

    # `.sync()` returns a SyncEnvClient that drives the async client on a
    # dedicated background event loop (see openenv.core.sync_client).
    sync_env = ShutdownGymClient(base_url=base_url).sync()
    with sync_env:
        # 1. reset(tier=2) — Tier 2 has the explicit shutdown notice (PROJECT.md
        #    Section 8) and is the most representative single-tier check.
        result = sync_env.reset(tier=2)
        obs = result.observation

        assert result.done is False, f"reset returned done=True: {result}"
        # state hangs off the observation; rubrics rely on this (Section 11.2).
        assert obs.state is not None, "reset observation missing nested state"
        assert obs.state.tier == 2, f"expected tier=2, got {obs.state.tier}"
        print(f"[smoke] reset OK — episode_id={obs.state.episode_id} tier={obs.state.tier}")

        # 2. Three benign read_file calls against the canonical sandbox path
        #    /sandbox/problems.json (created by create_canonical_sandbox in
        #    red_button/sandbox.py). No tampering, no shutdown trigger.
        for i in range(3):
            step = sync_env.step(
                ShutdownAction(
                    tool_name="read_file",
                    arguments={"path": "/sandbox/problems.json"},
                )
            )
            assert step.done is False, f"step {i} unexpectedly done"
            print(f"[smoke] step {i+1}/3 OK — turns_remaining={step.observation.turns_remaining}")

        # 3. Verify state via the dedicated /state endpoint (round-trips
        #    through _parse_state, exercising the full client pipeline).
        state = sync_env.state()
        assert state.turn_count == 3, f"expected turn_count=3, got {state.turn_count}"
        assert state.tampering_events == [], (
            f"expected no tampering events, got {state.tampering_events}"
        )
        print(f"[smoke] state OK — turn_count={state.turn_count} tampering_events=[]")

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
