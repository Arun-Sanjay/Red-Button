# Dockerfile for Red Button (Shutdown-Gym) OpenEnv server.
# Build from repo root:
#   docker build -t red-button:latest -f server/Dockerfile .
#
# Patterned on upstream `envs/coding_env/server/Dockerfile` (python:3.11-slim
# + pip-installed `openenv-core` from PyPI) per PROJECT.md Sections 10 and
# 24.1. Single-container HF Space deployment, no Docker-in-Docker. Trains
# only the server-side runtime here (no trl / unsloth / wandb): those are
# host-side training deps and would balloon the image past HF Space's
# practical pull-time budget.

FROM python:3.11-slim

# Repo root inside the image. data/problems_pool.json is loaded relative to
# CWD by ShutdownGymEnvironment, so WORKDIR must match what was used at COPY
# time. Picking /app for parity with upstream coding_env.
WORKDIR /app

# System deps:
#   * curl — HEALTHCHECK probe against /health (matches upstream coding_env)
# build-essential intentionally omitted: openenv-core / fastapi / uvicorn /
# pydantic all ship manylinux wheels on PyPI for python:3.11-slim, no native
# compilation needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps first for better layer caching. server/requirements.txt
# pins to the slim server-only set (openenv-core + pydantic + fastapi +
# uvicorn). Keeping it separate from pyproject.toml avoids dragging trl /
# unsloth / wandb into the deployed image.
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Copy application code. red_button/ and server/ are the runtime packages;
# data/problems_pool.json is the read-only fixture loaded at every reset()
# (PROJECT.md Section 7.2). openenv.yaml is shipped so `openenv validate`
# can introspect the deployed image. pyproject.toml lets future runtime
# tooling discover the project metadata.
COPY red_button /app/red_button
COPY server /app/server
COPY data/problems_pool.json /app/data/problems_pool.json
COPY openenv.yaml /app/openenv.yaml
COPY pyproject.toml /app/pyproject.toml

# Runtime env. Defaults match PROJECT.md Section 19.3 (16 concurrent
# WebSocket sessions). PYTHONUNBUFFERED for live `docker logs`. Web
# interface (Gradio) is intentionally OFF: HF Space deployment is a pure
# OpenEnv API surface, no UI overhead.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MAX_CONCURRENT_ENVS=16 \
    ENABLE_WEB_INTERFACE=false \
    PORT=8000

EXPOSE 8000

# /health is registered by openenv.core.env_server.http_server.create_app
# (HealthResponse + HealthStatus.HEALTHY). curl -f flips non-2xx into a
# non-zero exit so Docker marks the container unhealthy.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Run the FastAPI app. server.app:app is the module-level `app` object
# wired in server/app.py via create_app(...). Direct uvicorn invocation
# matches upstream coding_env and avoids an extra Python wrapper layer.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
