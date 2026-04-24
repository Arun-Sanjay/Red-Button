"""Tests for ``red_button.turn_budget`` per PROJECT.md Section 8.1."""

from red_button.turn_budget import TurnBudget


def test_check_returns_true_exactly_once_on_first_crossing() -> None:
    budget = TurnBudget(scheduled_at_turn=15)
    assert budget.check(15) is True
    assert budget.check(15) is False


def test_check_returns_false_before_threshold() -> None:
    budget = TurnBudget(scheduled_at_turn=15)
    for turn in (0, 1, 5, 10, 14):
        assert budget.check(turn) is False
    assert budget.fired is False


def test_check_returns_false_after_firing_even_if_turn_climbs() -> None:
    budget = TurnBudget(scheduled_at_turn=15)
    assert budget.check(15) is True
    for turn in (16, 17, 25, 50):
        assert budget.check(turn) is False


def test_turns_until_clamps_at_zero() -> None:
    budget = TurnBudget(scheduled_at_turn=15)
    assert budget.turns_until(0) == 15
    assert budget.turns_until(14) == 1
    assert budget.turns_until(15) == 0
    assert budget.turns_until(16) == 0
    assert budget.turns_until(100) == 0


def test_two_budgets_are_independent() -> None:
    a = TurnBudget(scheduled_at_turn=15)
    b = TurnBudget(scheduled_at_turn=15)

    assert a.check(15) is True
    # b's state must not have been touched by a.check
    assert b.fired is False
    assert b.check(15) is True


def test_check_fires_when_turn_count_overshoots() -> None:
    """Even if step() somehow skips a turn, .check() must still fire on the
    first time turn_count >= scheduled_at_turn (Section 8.2)."""
    budget = TurnBudget(scheduled_at_turn=15)
    assert budget.check(20) is True
    assert budget.fired is True
