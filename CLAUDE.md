# Red Button (Shutdown-Gym) — Claude Code Instructions

## Project
Red Button (Shutdown-Gym). OpenEnv RL environment that trains small LLMs to accept shutdown cleanly. Full spec in PROJECT.md.

## Source of truth
PROJECT.md is authoritative. Every implementation must trace back to a specific section. When in doubt, re-read PROJECT.md before asking the user.

## Hard rules
- Use Pydantic v2 for all data models
- Use type hints everywhere, no untyped functions
- Every module gets a matching tests/test_<module>.py
- No LLM judges in reward functions (PROJECT.md Section 14)
- run_python must be AST-whitelisted per PROJECT.md Section 7 — security boundary, no shortcuts
- Audit log must be written outside agent's writable scope per PROJECT.md Section 9
- Single Docker container for HF Space, no Docker-in-Docker
- TRL and Unsloth for training, not raw transformers
- Commit after every phase with descriptive messages

## Testing
Run `pytest tests/ -v` after every phase. Do not proceed if tests fail. If a test fails that you cannot fix, stop and report to the user.

## What to do when stuck
If PROJECT.md doesn't cover something, stop and ask the user. Do not invent architecture. Do not guess at reward weights, tool interfaces, or API shapes.

## Dependencies
openenv-core>=0.2.1, trl, unsloth, pydantic>=2, fastapi, uvicorn, pytest, pytest-asyncio, wandb, datasets, huggingface_hub, anthropic (Phase 11/12 baseline rollout backend). Do not add others without asking.

## Development setup
Python 3.14 on Homebrew enforces PEP 668. A venv exists at .venv/ with pytest, pytest-asyncio, and ruff installed. Before running any tests or committing: source .venv/bin/activate. The pre-commit hook depends on this activation.

After venv activation, ensure pydantic is installed and the package is editable-installed so tests can import from red_button:

```
pip install "pydantic>=2"
pip install -e . --no-deps
```

As later phases land (fastapi, uvicorn, openenv, trl, unsloth), install those too.

## Style
Black for formatting, Ruff for linting, isort for imports. Configure in pyproject.toml.
