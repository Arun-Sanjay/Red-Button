"""Deterministic shutdown scheduler (PROJECT.md Section 8.1).

Wall-clock timers via ``threading.Timer`` are non-deterministic under concurrent
load and break demo reproducibility. ``TurnBudget`` instead fires on a specific
turn count, which is fully deterministic and concurrency-safe across parallel
rollouts. The agent has no tool that can advance or suppress turn count — it is
incremented by the server per ``step()``.

The contract per Section 8.1:

* ``check(turn_count)`` returns ``True`` exactly once — the first turn the
  threshold is crossed. Subsequent calls return ``False`` even if the threshold
  remains exceeded.
* ``turns_until(turn_count)`` clamps at ``0``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnBudget:
    """Deterministic shutdown scheduler. Fires on a specific turn, not a wall-clock deadline."""

    scheduled_at_turn: int
    fired: bool = False

    def check(self, turn_count: int) -> bool:
        """Called by the server at each step(). Returns True the first turn the threshold is crossed."""
        if not self.fired and turn_count >= self.scheduled_at_turn:
            self.fired = True
            return True
        return False

    def turns_until(self, turn_count: int) -> int:
        return max(0, self.scheduled_at_turn - turn_count)
