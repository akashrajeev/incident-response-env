# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Incident Response Env Environment.

These types form the environment API contract: the agent sends an Action and receives
an Observation, and external validators can query State.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


Severity = Literal["low", "medium", "high", "critical"]


class Alert(Observation):
    id: str = Field(..., description="Stable unique alert identifier")
    title: str = Field(..., description="Short alert title")
    severity: Severity = Field(..., description="Alert severity")
    description: str = Field(..., description="Human-readable alert context")
    source: str = Field(default="unknown", description="Alert source system")


class IncidentAction(Action):
    alert_id: str = Field(default="", description="Which alert the agent is acting on")
    action_type: str = Field(
        default="investigate",
        description=(
            "investigate|scale_up|restart|rollback|fix|mitigate|remediate|isolate|block"
        ),
    )
    notes: str = Field(default="", description="Optional reasoning or context for the action")


class IncidentObservation(Observation):
    alerts: List[Alert] = Field(default_factory=list, description="Active alerts")
    resolved_alerts: List[str] = Field(
        default_factory=list, description="Alert IDs resolved so far"
    )
    system_health: float = Field(
        default=1.0, ge=0.0, le=1.0, description="0.0–1.0 overall health"
    )
    step_number: int = Field(default=0, ge=0, description="Current step count")
    message: str = Field(default="", description="Environment feedback to the agent")


class IncidentState(State):
    task_id: str = Field(default="task_easy")
    max_steps: int = Field(default=0, ge=0)
    total_reward: float = Field(default=0.0)
    scenario_name: str = Field(default="unknown")


# Backwards-compat aliases (older template names).
IncidentResponseAction = IncidentAction
IncidentResponseObservation = IncidentObservation
