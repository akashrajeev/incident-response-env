# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Incident Response Env Environment.

This module creates an HTTP server that exposes the IncidentResponseEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from __future__ import annotations

import inspect

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    # Docker/validator mode: run as `uvicorn server.app:app` from repo root.
    from fastapi import FastAPI
    from fastapi.routing import APIRoute
    from models import (
        IncidentResponseAction,
        IncidentResponseObservation,
        IncidentState,
    )
    from server.environment import IncidentResponseEnvironment
except Exception:  # pragma: no cover
    # Package mode: `uvicorn incident_response_env.server.app:app`
    from fastapi import FastAPI
    from fastapi.routing import APIRoute

    from models import (
        IncidentResponseAction,
        IncidentResponseObservation,
        IncidentState,
    )
    from .environment import IncidentResponseEnvironment

# OpenEnv's HTTP /reset and /step handlers invoke the factory for every request.
# A fresh Environment per request breaks episode state (each /step would hit
# scenario=None and fall back to task_easy). Use a single shared instance so
# stateless HTTP clients behave like one continuous episode.
_shared_incident_env = IncidentResponseEnvironment()


def incident_env_factory() -> IncidentResponseEnvironment:
    return _shared_incident_env


# Create the app. Prefer the hackathon-style signature: create_app(factory)
sig = None
try:  # pragma: no cover
    sig = inspect.signature(create_app)
except Exception:  # pragma: no cover
    sig = None

if sig is not None and len(sig.parameters) == 1:
    app = create_app(incident_env_factory)  # type: ignore[misc]
else:
    # Older signature used by the OpenEnv template.
    app = create_app(  # type: ignore[misc]
        incident_env_factory,
        IncidentResponseAction,
        IncidentResponseObservation,
        env_name="incident_response_env",
        max_concurrent_envs=1,
    )


def _reregister_state_route(application: FastAPI) -> None:
    """OpenEnv registers GET /state with base State, which omits IncidentState fields."""
    application.router.routes = [
        route
        for route in application.router.routes
        if not (
            isinstance(route, APIRoute)
            and route.path == "/state"
            and "GET" in route.methods
        )
    ]

    @application.get(
        "/state",
        response_model=IncidentState,
        tags=["State Management"],
        summary="Get current environment state",
    )
    async def incident_state() -> IncidentState:
        return _shared_incident_env.state


_reregister_state_route(app)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m incident_response_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn incident_response_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # openenv validate expects a literal `main()` substring in this file
    if args.port == 8000:
        main()
    else:
        main(port=args.port)
