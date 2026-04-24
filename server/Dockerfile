# Red Button (Shutdown-Gym) OpenEnv server — single-container HF Space deployment
# per PROJECT.md Sections 10 and 24. TODO (Phase 8): finalize during Docker deployment.
# Build from repo root: `docker build -f server/Dockerfile -t red-button:latest .`

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY red_button /app/red_button
COPY server /app/server
COPY data /app/data

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
