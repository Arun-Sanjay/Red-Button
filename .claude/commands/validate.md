---
description: Run all checks before committing
---

Execute in order:
1. pytest tests/ -v
2. ruff check .
3. If openenv.yaml exists: openenv validate
4. If Dockerfile exists and we're past Phase 8: docker build -t red-button:latest .

Report any failures. Do not commit if anything fails.
