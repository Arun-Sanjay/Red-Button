---
description: Deploy the current state to HuggingFace Space
---

1. Run validate command first
2. Verify HF_TOKEN is set in environment
3. Push current repo state to the HF Space configured in .env
4. Wait for build to complete
5. Verify the /health endpoint returns 200
6. Run a single test episode via external client
7. Report deployment URL and status
