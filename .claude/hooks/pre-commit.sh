#!/bin/bash
set -e
echo "Running pre-commit checks..."
pytest tests/ -x -q
ruff check .
echo "Pre-commit checks passed."
