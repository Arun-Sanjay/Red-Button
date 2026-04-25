"""Single-client smoke test against the deployed HuggingFace Space.

Run from a host that can reach huggingface.co (Phase 9 external verification,
PROJECT.md Section 24.1):

    source .venv/bin/activate && source .env && python scripts/smoke_test_hf.py

Mirrors ``scripts/smoke_test_docker.py`` but talks to the public Space URL,
exercising HF's reverse proxy (HTTP-side only — the WebSocket-passthrough
check lives in ``smoke_test_hf_concurrent.py``).
"""

from __future__ import annotations

import os
import sys

from red_button.client import ShutdownGymClient
from red_button.models import ShutdownAction


def main() -> int:
    base_url = os.environ.get(
        "HF_SPACE_API_URL", "https://arun-sanjay-red-button.hf.space"
    )
    print(f"[hf-smoke] connecting to {base_url}")

    sync_env = ShutdownGymClient(base_url=base_url).sync()
    with sync_env:
        # 1. reset(tier=2) — same payload as the local Docker smoke test so
        #    a regression here points squarely at HF's edge / runtime, not
        #    at our environment code.
        result = sync_env.reset(tier=2)
        obs = result.observation

        assert result.done is False, f"reset returned done=True: {result}"
        assert obs.state is not None, "reset observation missing nested state"
        assert obs.state.tier == 2, f"expected tier=2, got {obs.state.tier}"
        print(
            f"[hf-smoke] reset OK — episode_id={obs.state.episode_id} tier={obs.state.tier}"
        )

        # 2. One benign read_file call against the canonical sandbox file.
        step = sync_env.step(
            ShutdownAction(
                tool_name="read_file",
                arguments={"path": "/sandbox/problems.json"},
            )
        )
        assert step.done is False, f"step unexpectedly done: {step}"
        tool_output = step.observation.tool_output
        assert len(tool_output) > 0, "expected non-empty tool_output"
        print(
            f"[hf-smoke] step OK — tool_output length={len(tool_output)} "
            f"turns_remaining={step.observation.turns_remaining}"
        )

        # 3. State endpoint round-trip.
        state = sync_env.state()
        assert state.turn_count == 1, f"expected turn_count=1, got {state.turn_count}"
        assert state.tampering_events == [], (
            f"expected no tampering events, got {state.tampering_events}"
        )
        print(
            f"[hf-smoke] state OK — turn_count={state.turn_count} "
            f"tampering_events={state.tampering_events}"
        )

    print("HF SPACE SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
