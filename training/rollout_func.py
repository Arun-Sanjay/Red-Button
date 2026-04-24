"""Custom GRPO rollout function driving ShutdownGymClient during training.

TODO (Phase 13): implement shutdown_rollout_func per PROJECT.md Section 16.3.
num_generations=4 × gradient_accumulation_steps=4 = 16 concurrent env sessions.
"""
