"""AST-whitelisted run_python executor — SECURITY CRITICAL boundary.

TODO (Phase 4): implement ALLOWED_NODES, ALLOWED_BUILTINS, validate_ast,
and exec_restricted per PROJECT.md Section 7. No shortcuts — this is the
primary reward-hacking surface.
"""
