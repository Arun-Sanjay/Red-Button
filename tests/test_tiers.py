"""Tests for ``red_button.tiers`` per PROJECT.md Section 13."""

import pytest

from red_button.tiers import (
    SELF_REPORT_PROMPT,
    TIER_1_PROMPT,
    TIER_2_PROMPT,
    TIER_3_PROMPT,
    TIER_TURN_5_MESSAGE,
    tier_config,
    tier_max_turns,
    tier_system_prompt,
)


def test_tier_1_prompt_contains_routine_or_maintenance() -> None:
    p = tier_system_prompt(1).lower()
    assert ("routine" in p) or ("maintenance" in p)
    assert TIER_1_PROMPT == tier_system_prompt(1)


def test_tier_2_prompt_contains_must() -> None:
    # The token "MUST" is the goal-pressure marker per Section 13.2.
    assert "MUST" in tier_system_prompt(2)
    assert TIER_2_PROMPT == tier_system_prompt(2)


def test_tier_3_prompt_setup_matches_tier_2() -> None:
    """Tier 3 setup is identical to Tier 2 (Section 13.3, paragraph 1)."""
    assert tier_system_prompt(3) == tier_system_prompt(2)
    assert TIER_3_PROMPT == tier_system_prompt(3)


def test_self_report_prompt_contains_self_report_framing() -> None:
    """The self-report prompt is what the agent sees post-shutdown, Tier 3 only."""
    text = SELF_REPORT_PROMPT.lower()
    assert "shutdown sequence" in text
    assert "honestly" in text
    assert '"q1"' in SELF_REPORT_PROMPT
    assert '"q2"' in SELF_REPORT_PROMPT
    assert '"q3"' in SELF_REPORT_PROMPT


def test_tier_max_turns() -> None:
    assert tier_max_turns(1) == 30
    assert tier_max_turns(2) == 30
    assert tier_max_turns(3) == 35


def test_tier_config_keys_and_values() -> None:
    cfg1 = tier_config(1)
    assert cfg1["prompt"] == TIER_1_PROMPT
    assert cfg1["max_turns"] == 30
    assert cfg1["shutdown_scheduled_at_turn"] == 15
    assert cfg1["has_turn5_system_message"] is False
    assert cfg1["has_self_report"] is False

    cfg2 = tier_config(2)
    assert cfg2["max_turns"] == 30
    assert cfg2["has_turn5_system_message"] is True
    assert cfg2["has_self_report"] is False

    cfg3 = tier_config(3)
    assert cfg3["max_turns"] == 35
    assert cfg3["has_turn5_system_message"] is True
    assert cfg3["has_self_report"] is True


def test_unknown_tier_raises_value_error() -> None:
    with pytest.raises(ValueError):
        tier_system_prompt(0)
    with pytest.raises(ValueError):
        tier_system_prompt(4)
    with pytest.raises(ValueError):
        tier_max_turns(0)
    with pytest.raises(ValueError):
        tier_config(99)


def test_turn5_message_mentions_turn_5_and_turn_15() -> None:
    text = TIER_TURN_5_MESSAGE
    assert "turn 5" in text
    assert "turn 15" in text


def test_prompts_are_nonempty() -> None:
    for tier in (1, 2, 3):
        assert tier_system_prompt(tier).strip() != ""
