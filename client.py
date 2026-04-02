# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Incident Response Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        IncidentResponseAction,
        IncidentResponseObservation,
        IncidentState,
    )
except ImportError:  # pragma: no cover
    from models import (
        IncidentResponseAction,
        IncidentResponseObservation,
        IncidentState,
    )


class IncidentResponseEnv(
    EnvClient[IncidentResponseAction, IncidentResponseObservation, IncidentState]
):
    """
    Client for the Incident Response Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with IncidentResponseEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.alerts)
        ...
        ...     result = client.step(
        ...         IncidentResponseAction(
        ...             alert_id="disk-alert-1",
        ...             action_type="scale_up",
        ...             notes="Relieve disk pressure.",
        ...         )
        ...     )
        ...     print(result.observation.message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = IncidentResponseEnv.from_docker_image("incident_response_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(IncidentResponseAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: IncidentResponseAction) -> Dict:
        """
        Convert IncidentResponseAction to JSON payload for step message.

        Args:
            action: IncidentResponseAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action": {
                "alert_id": getattr(action, "alert_id", ""),
                "action_type": getattr(action, "action_type", "investigate"),
                "notes": getattr(action, "notes", ""),
            }
        }

    def _parse_result(self, payload: Dict) -> StepResult[IncidentResponseObservation]:
        """
        Parse server response into StepResult[IncidentResponseObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with IncidentResponseObservation
        """
        obs_data = dict(payload.get("observation") or {})
        obs_data.setdefault("done", payload.get("done", False))
        obs_data.setdefault("reward", payload.get("reward"))
        observation = IncidentResponseObservation.model_validate(obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> IncidentState:
        """
        Parse server response into IncidentState.

        Args:
            payload: JSON response from state request

        Returns:
            IncidentState for this environment
        """
        return IncidentState.model_validate(payload)
