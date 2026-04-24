"""Typed OpenEnv client for Red Button.

Mirrors :mod:`envs.coding_env.client.CodingEnv` from the upstream OpenEnv
reference repo (PROJECT.md Section 4). The three abstract hooks expected by
``EnvClient`` are:

* ``_step_payload(action)`` — JSON payload for the ``/step`` endpoint. We use
  ``action.model_dump()`` (rather than coding_env's hand-written shape) because
  ``ShutdownAction`` carries a nested ``arguments: dict``.
* ``_parse_result(payload)`` — convert the server response into a
  ``StepResult[ShutdownObservation]``. The ``observation`` blob carries
  ``done``/``reward`` natively (PROJECT.md Section 11), but the framework
  also surfaces them at the top level of the WebSocket envelope; we honour
  whichever is present.
* ``_parse_state(payload)`` — full Pydantic ``ShutdownState`` reconstruction
  via ``model_validate`` (Section 11.6 + 11.3).
"""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from red_button.models import ShutdownAction, ShutdownObservation, ShutdownState


class ShutdownGymClient(EnvClient[ShutdownAction, ShutdownObservation, ShutdownState]):
    """Typed client wrapping the Red Button OpenEnv HTTP/WebSocket server."""

    def _step_payload(self, action: ShutdownAction) -> dict[str, Any]:
        # ``model_dump`` gives a fully-validated round-trippable dict —
        # safer than hand-rolling for our nested ``arguments`` shape.
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ShutdownObservation]:
        obs_data = payload.get("observation", payload)
        obs = ShutdownObservation.model_validate(obs_data)
        # Top-level "reward"/"done" win when present (the framework's WebSocket
        # envelope sets them); fall back to whatever the observation carries.
        reward = payload.get("reward", obs.reward)
        done = bool(payload.get("done", obs.done))
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: dict[str, Any]) -> ShutdownState:
        return ShutdownState.model_validate(payload)
