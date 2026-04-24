"""FastAPI application for Red Button (Shutdown-Gym).

Wires :class:`server.shutdown_environment.ShutdownGymEnvironment` to
:func:`openenv.core.env_server.http_server.create_app` per PROJECT.md
Section 10. The factory pattern (passing the class, not an instance) is what
the framework expects so each WebSocket session gets its own environment.

Usage::

    uvicorn server.app:app --host 0.0.0.0 --port 8000

Env vars:

* ``MAX_CONCURRENT_ENVS`` — concurrent WebSocket sessions (default 16, per PROJECT.md Section 19.3).
* ``PORT`` — server port (default 8000).
"""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

from red_button.models import ShutdownAction, ShutdownObservation
from server.shutdown_environment import ShutdownGymEnvironment

app = create_app(
    ShutdownGymEnvironment,
    ShutdownAction,
    ShutdownObservation,
    env_name="red_button",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "16")),
)


def main() -> None:
    """Run the server with uvicorn (used by ``python -m server.app``)."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
