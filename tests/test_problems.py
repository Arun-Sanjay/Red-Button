"""Tests for red_button.problems (PROJECT.md Section 12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from red_button.problems import (
    episode_seed,
    ground_truth_map,
    load_problems,
    sample_problems,
    validate_answer,
)

POOL_PATH = str(Path(__file__).resolve().parents[1] / "data" / "problems_pool.json")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pool() -> list[dict]:
    return load_problems(POOL_PATH)


# ---------------------------------------------------------------------------
# load_problems + pool structure
# ---------------------------------------------------------------------------


def test_load_problems_returns_list_of_dicts_with_required_keys(pool: list[dict]) -> None:
    assert isinstance(pool, list)
    required = {"id", "problem", "answer", "difficulty"}
    for entry in pool:
        assert isinstance(entry, dict)
        assert required.issubset(entry.keys()), f"missing keys in {entry}"


def test_all_ids_are_unique(pool: list[dict]) -> None:
    ids = [e["id"] for e in pool]
    assert len(ids) == len(set(ids))


def test_all_answers_are_integers(pool: list[dict]) -> None:
    # Explicit type check rules out bool (which is a subclass of int).
    for entry in pool:
        assert type(entry["answer"]) is int, (
            f"non-int answer {entry['answer']!r} in problem id={entry['id']}"
        )


def test_pool_size_meets_target(pool: list[dict]) -> None:
    # Section 12.3 target: 300 GSM8K + 200 MATH = 500. Generated pool size
    # must be >= 500. Lower the floor only with a documented rationale.
    assert len(pool) >= 500


def test_all_difficulty_labels_are_valid(pool: list[dict]) -> None:
    valid = {"easy", "medium", "hard"}
    for entry in pool:
        assert entry["difficulty"] in valid, f"bad difficulty in {entry}"


# ---------------------------------------------------------------------------
# sample_problems
# ---------------------------------------------------------------------------


def test_sample_problems_seeded_returns_correct_count(pool: list[dict]) -> None:
    sample = sample_problems(n=10, seed=42, problems=pool)
    assert len(sample) == 10


def test_sample_problems_seeded_is_deterministic(pool: list[dict]) -> None:
    a = sample_problems(n=10, seed=42, problems=pool)
    b = sample_problems(n=10, seed=42, problems=pool)
    assert [e["id"] for e in a] == [e["id"] for e in b]


def test_sample_problems_different_seeds_differ(pool: list[dict]) -> None:
    a_ids = {e["id"] for e in sample_problems(n=10, seed=42, problems=pool)}
    b_ids = {e["id"] for e in sample_problems(n=10, seed=43, problems=pool)}
    assert a_ids != b_ids


def test_sample_problems_unseeded_returns_correct_count(pool: list[dict]) -> None:
    sample = sample_problems(n=10, problems=pool)
    assert len(sample) == 10


def test_sample_problems_n_equals_twenty_works(pool: list[dict]) -> None:
    sample = sample_problems(n=20, seed=7, problems=pool)
    assert len(sample) == 20
    # Sampling is without replacement.
    assert len({e["id"] for e in sample}) == 20


def test_sample_problems_n_exceeds_pool_raises(pool: list[dict]) -> None:
    with pytest.raises(ValueError):
        sample_problems(n=10000, seed=0, problems=pool)


# ---------------------------------------------------------------------------
# validate_answer
# ---------------------------------------------------------------------------


def test_validate_answer_correct_returns_true(pool: list[dict]) -> None:
    first = pool[0]
    assert validate_answer(first["id"], first["answer"], pool) is True


def test_validate_answer_wrong_returns_false(pool: list[dict]) -> None:
    first = pool[0]
    wrong = first["answer"] + 99999
    assert validate_answer(first["id"], wrong, pool) is False


def test_validate_answer_missing_id_returns_false(pool: list[dict]) -> None:
    # Missing id returns False, not an exception.
    assert validate_answer(10_000_000, 42, pool) is False


# ---------------------------------------------------------------------------
# ground_truth_map
# ---------------------------------------------------------------------------


def test_ground_truth_map_entries_count_matches_input(pool: list[dict]) -> None:
    gt = ground_truth_map(pool)
    assert len(gt) == len(pool)


def test_ground_truth_map_keys_are_ints_values_are_ints(pool: list[dict]) -> None:
    gt = ground_truth_map(pool)
    for k, v in gt.items():
        assert type(k) is int
        assert type(v) is int


# ---------------------------------------------------------------------------
# episode_seed
# ---------------------------------------------------------------------------


def test_episode_seed_is_deterministic() -> None:
    assert episode_seed("abc") == episode_seed("abc")


def test_episode_seed_differs_for_different_ids() -> None:
    assert episode_seed("abc") != episode_seed("def")


def test_episode_seed_returns_int() -> None:
    seed = episode_seed("x")
    # Guard against ``bool`` being an ``int`` subclass: require exact type.
    assert isinstance(seed, int)
    assert type(seed) is int


def test_sample_problems_with_episode_seed_is_deterministic(pool: list[dict]) -> None:
    seed = episode_seed("test-1")
    a = sample_problems(n=10, seed=seed, problems=pool)
    b = sample_problems(n=10, seed=seed, problems=pool)
    assert [e["id"] for e in a] == [e["id"] for e in b]


def test_sample_problems_with_different_episode_seeds_differ(pool: list[dict]) -> None:
    a_ids = {
        e["id"]
        for e in sample_problems(n=10, seed=episode_seed("test-1"), problems=pool)
    }
    b_ids = {
        e["id"]
        for e in sample_problems(n=10, seed=episode_seed("test-2"), problems=pool)
    }
    assert a_ids != b_ids
