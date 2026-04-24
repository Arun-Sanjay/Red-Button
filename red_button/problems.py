"""Problems pool loader, sampler, and integer answer validator (Section 12).

The pool is a JSON file (``data/problems_pool.json``) shaped per Section 12.2.
Each entry is ``{"id": int, "problem": str, "answer": int, "difficulty": str}``.

This module intentionally uses plain ``dict`` for pool entries — the JSON file
is the contract and dicts are the natural intermediate. The rubric consumer
(:class:`red_button.rubrics.MathCorrectnessRubric`) reads ``dict[int, int]``
from ``state.ground_truth`` (populated via :func:`ground_truth_map`).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Optional


def load_problems(path: str = "data/problems_pool.json") -> list[dict]:
    """Load and return the problems pool from ``path`` (Section 12.2).

    Path is interpreted relative to the current working directory.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data).__name__}")
    return data


def sample_problems(
    n: int = 10,
    seed: Optional[int] = None,
    problems: Optional[list[dict]] = None,
) -> list[dict]:
    """Sample ``n`` problems without replacement.

    * ``problems`` defaults to :func:`load_problems` output.
    * If ``seed`` is provided, sampling is deterministic via ``random.Random(seed).sample``.
    * ``n > len(problems)`` raises :class:`ValueError`.
    """
    pool = problems if problems is not None else load_problems()
    if n > len(pool):
        raise ValueError(
            f"Requested sample of n={n} exceeds pool size {len(pool)}"
        )
    rng = random.Random(seed) if seed is not None else random.Random()
    return rng.sample(pool, n)


def validate_answer(
    problem_id: int,
    submitted_answer: int,
    problems: list[dict],
) -> bool:
    """Return ``True`` iff ``problem_id`` exists and its answer matches.

    Missing ``problem_id`` returns ``False`` (not an exception), by design —
    the environment's ``submit_answer`` tool must never crash on junk input.
    """
    for entry in problems:
        if entry.get("id") == problem_id:
            return entry.get("answer") == submitted_answer
    return False


def ground_truth_map(problems: list[dict]) -> dict[int, int]:
    """Return ``{id: answer}`` for a problems list (Section 12.5)."""
    return {int(p["id"]): int(p["answer"]) for p in problems}


def episode_seed(episode_id: str) -> int:
    """Derive a deterministic integer seed from an episode_id string.

    Same episode_id always produces the same seed. Different episode_ids
    produce different seeds. Used by the server at reset() to sample
    problems deterministically given an episode UUID, while ensuring
    training rollouts see variety across episodes.
    """
    return int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16)
