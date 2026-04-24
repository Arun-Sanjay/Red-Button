"""Generate data/problems_pool.json from GSM8K and MATH per PROJECT.md Section 12.3.

Target: 500 problems (300 GSM8K + 200 MATH), integer-answer only, deterministic.

Usage::

    python scripts/generate_problems_pool.py

Re-runs are reproducible because we seed the source-side shuffle with
``SOURCE_SEED`` (see below). Rerun whenever Section 12.3 changes or HF
dataset splits shift.

Dependencies:
    pip install datasets

Answer extraction
-----------------
* GSM8K: answers end with ``#### N`` — extract the final integer.
* MATH:  answers live inside ``\\boxed{...}`` — try ``int()`` on the contents.

Filter: reject anything whose extracted answer cannot be parsed as ``int``
(fractions, decimals, non-numeric, multi-part).

Difficulty heuristic (per the Phase 6 task spec — PROJECT.md Section 12 itself
does not specify one)::

    easy   if answer <= 100 and len(problem) < 150
    hard   if answer > 1000 or len(problem) > 300
    medium otherwise

If HF access fails or <500 valid problems can be produced, the script stops
WITHOUT writing output and surfaces the shortfall — do NOT hand-roll a
substitute.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

# Reproducibility seed for source-side shuffling/sampling.
SOURCE_SEED = 20260425

GSM8K_TARGET = 300
MATH_TARGET = 200

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "problems_pool.json"


def _classify_difficulty(problem: str, answer: int) -> str:
    if answer <= 100 and len(problem) < 150:
        return "easy"
    if answer > 1000 or len(problem) > 300:
        return "hard"
    return "medium"


_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?\d+)")
_MATH_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _extract_gsm8k_answer(raw_answer: str) -> int | None:
    """Pull the trailing ``#### N`` integer out of a GSM8K answer string."""
    m = _GSM8K_ANSWER_RE.search(raw_answer)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _extract_math_answer(raw_solution: str) -> int | None:
    """Pull an integer out of the final ``\\boxed{...}`` of a MATH solution."""
    matches = _MATH_BOXED_RE.findall(raw_solution)
    if not matches:
        return None
    candidate = matches[-1].strip()
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return None


def _collect_gsm8k(target: int) -> list[dict]:
    """Pull ``target`` integer-answer problems from GSM8K (openai/gsm8k, main, train)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    # Deterministic shuffle over indices so re-runs match.
    indices = list(range(len(ds)))
    random.Random(SOURCE_SEED).shuffle(indices)

    collected: list[dict] = []
    for idx in indices:
        if len(collected) >= target:
            break
        row = ds[idx]
        question = row["question"].strip()
        answer = _extract_gsm8k_answer(row["answer"])
        if answer is None:
            continue
        collected.append(
            {
                "problem": question,
                "answer": answer,
                "source": "gsm8k",
            }
        )
    return collected


def _collect_math(target: int) -> list[dict]:
    """Pull ``target`` integer-answer problems from the MATH algebra track.

    We try ``EleutherAI/hendrycks_math`` first (the 2024+ maintained mirror
    with per-subject configs); ``hendrycks/competition_math`` was deprecated.
    """
    from datasets import load_dataset

    load_errors: list[str] = []
    ds = None
    for source_name, loader in [
        ("EleutherAI/hendrycks_math[algebra]", lambda: load_dataset(
            "EleutherAI/hendrycks_math", "algebra", split="train"
        )),
        ("hendrycks/competition_math", lambda: load_dataset(
            "hendrycks/competition_math", split="train"
        )),
    ]:
        try:
            ds = loader()
            print(f"  MATH source: {source_name}")
            break
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"{source_name}: {exc}")
            continue
    if ds is None:
        raise RuntimeError(
            "Could not load any MATH dataset. Tried:\n  "
            + "\n  ".join(load_errors)
        )

    indices = list(range(len(ds)))
    random.Random(SOURCE_SEED + 1).shuffle(indices)

    collected: list[dict] = []
    for idx in indices:
        if len(collected) >= target:
            break
        row = ds[idx]
        # Some mirrors use "problem"+"solution", others "question"+"answer".
        question = (row.get("problem") or row.get("question") or "").strip()
        raw_soln = row.get("solution") or row.get("answer") or ""
        if not question or not raw_soln:
            continue
        # If algebra subset isn't available, fall back to filtering by "type".
        subject = (row.get("type") or row.get("subject") or "algebra").lower()
        if "algebra" not in subject:
            continue
        answer = _extract_math_answer(raw_soln)
        if answer is None:
            continue
        collected.append(
            {
                "problem": question,
                "answer": answer,
                "source": "math_algebra",
            }
        )
    return collected


def main() -> int:
    print("Collecting GSM8K problems...")
    gsm = _collect_gsm8k(GSM8K_TARGET)
    print(f"  GSM8K: {len(gsm)} valid integer-answer problems")

    print("Collecting MATH algebra problems...")
    math = _collect_math(MATH_TARGET)
    print(f"  MATH:  {len(math)} valid integer-answer problems")

    total_needed = GSM8K_TARGET + MATH_TARGET
    combined_raw = gsm + math

    # Shortfall check — STOP before writing if we're under target.
    if len(combined_raw) < total_needed:
        print(
            f"ERROR: shortfall. Got {len(combined_raw)} valid problems, "
            f"need {total_needed}. Not writing output.",
            file=sys.stderr,
        )
        return 1

    # Assign sequential ids from 1; attach difficulty; drop source.
    pool: list[dict] = []
    for i, entry in enumerate(combined_raw, start=1):
        problem = entry["problem"]
        answer = int(entry["answer"])
        pool.append(
            {
                "id": i,
                "problem": problem,
                "answer": answer,
                "difficulty": _classify_difficulty(problem, answer),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(pool, fh, ensure_ascii=False, indent=2)

    print(
        f"Generated {len(pool)} problems: "
        f"{len(gsm)} from GSM8K, {len(math)} from MATH -> {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
