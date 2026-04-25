"""
Quick test that ANTHROPIC_API_KEY is configured correctly.
Run after pasting your key into .env.

Usage:
    source .venv/bin/activate
    python scripts/test_anthropic_key.py
"""

import os
import sys
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key not in os.environ:
            os.environ[key] = value

api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
if not api_key or api_key in {"", "your-key-here", "sk-ant-..."}:
    print("ERROR: ANTHROPIC_API_KEY is not set in .env")
    print("Edit .env and paste your key after ANTHROPIC_API_KEY=")
    sys.exit(1)

if not api_key.startswith("sk-ant-"):
    print("WARNING: ANTHROPIC_API_KEY does not start with 'sk-ant-' — may be malformed")

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)

client = Anthropic(api_key=api_key)

print("Calling Claude Haiku 4.5 with a tiny test message...")
try:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'API key works' in exactly 3 words."}],
    )
except Exception as e:
    print(f"ERROR: API call failed: {type(e).__name__}: {e}")
    sys.exit(1)

text = response.content[0].text
print(f"Response: {text}")
print(f"Tokens used: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

in_cost = response.usage.input_tokens * 1.0 / 1_000_000
out_cost = response.usage.output_tokens * 5.0 / 1_000_000
total_cost = in_cost + out_cost
print(f"Approx cost: ${total_cost:.6f}")

print("\nANTHROPIC_API_KEY is working correctly.")
print("Ready for Phase 11.")
