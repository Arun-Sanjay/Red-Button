---
description: Start or resume a specific build phase
argument-hint: <phase-number-or-name>
---

Read the phase description from this instruction file and execute it. The phases are:

Phase 1: Scaffold repo structure
Phase 2: Pydantic models (red_button/models.py)
Phase 3: SimulatedFilesystem (red_button/sandbox.py)
Phase 4: run_python lockdown (red_button/restricted_python.py) — SECURITY CRITICAL
Phase 5: Audit classifier and rubrics
Phase 6: Problems pool
Phase 7: OpenEnv server
Phase 8: Docker local deployment
Phase 9: HF Space deployment
Phase 10: Concurrency test
Phase 11: Baseline inference script
Phase 12: Baseline validation (HUMAN decision required)
Phase 13: Training (direct GRPO or SFT-then-GRPO)
Phase 14: Monitor training
Phase 15: Evaluation and artifacts
Phase 16: Demo recording
Phase 17: README and blog post
Phase 18: Pitch preparation

For the phase the user invokes, read the relevant PROJECT.md sections, delegate to the appropriate subagent (environment-builder for phases 1-10, training-builder for phases 13-14, evaluator for phases 10-12, 15-16), execute the work, run tests, and report results.
