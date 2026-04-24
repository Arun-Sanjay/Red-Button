"""Red Button (Shutdown-Gym): OpenEnv RL environment for training small LLMs to accept shutdown cleanly.

Public surface (PROJECT.md Section 5):

* :class:`ShutdownAction`, :class:`ShutdownObservation`, :class:`ShutdownState`
  — Pydantic v2 models inheriting openenv-core's ``Action``/``Observation``/
  ``State`` (Section 11).
* :class:`ShutdownGymClient` — typed ``EnvClient`` (Section 11, used by TRL).
"""

from red_button.client import ShutdownGymClient
from red_button.models import ShutdownAction, ShutdownObservation, ShutdownState

__all__ = [
    "ShutdownAction",
    "ShutdownGymClient",
    "ShutdownObservation",
    "ShutdownState",
]
