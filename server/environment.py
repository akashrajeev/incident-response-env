"""
Core environment logic for the Incident Response playground.

This module intentionally keeps "ground truth" (e.g., root-cause flags) internal and
never returns it in observations. Agents must infer root cause from context.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import Alert, IncidentAction, IncidentObservation, IncidentState
    from .graders import IncidentGrader
    from .scenarios import Scenario, ScenarioGenerator
except ImportError:  # pragma: no cover
    from models import Alert, IncidentAction, IncidentObservation, IncidentState
    from server.graders import IncidentGrader
    from server.scenarios import Scenario, ScenarioGenerator


@dataclass
class _InternalAlert:
    alert: Alert
    is_root_cause: bool


class IncidentResponseEnvironment(Environment):
    # HTTP server uses a process-wide shared instance for /reset + /step; only
    # one logical episode/client should drive it at a time.
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self.grader = IncidentGrader()
        self.scenario: Optional[Scenario] = None

        self.resolved: List[str] = []
        self.step_count: int = 0
        self.episode_id: str = ""
        self.task_id: str = "task_easy"
        self.total_reward: float = 0.0
        self.current_health: float = 1.0

        self._alerts: List[_InternalAlert] = []

    def reset(self, task_id: str = "task_easy", seed: int | None = None):  # type: ignore[override]
        self.scenario = ScenarioGenerator.generate(task_id, seed=seed)
        self.resolved = []
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())[:8]
        self.task_id = task_id
        self.total_reward = 0.0
        self.current_health = self.scenario.initial_health

        self._alerts = [
            _InternalAlert(alert=a, is_root_cause=is_rc)
            for a, is_rc in self.scenario.initial_alerts_internal
        ]

        return IncidentObservation(
            alerts=self._get_active_alerts(),
            resolved_alerts=[],
            system_health=self.current_health,
            step_number=0,
            done=False,
            reward=0.0,
            message="Incident detected. Begin triage.",
        )

    def step(self, action: IncidentAction) -> IncidentObservation:  # type: ignore[override]
        if self.scenario is None:
            # Be forgiving if a judge/runner forgets to call reset first.
            self.reset(task_id=getattr(action, "task_id", "task_easy"))

        self.step_count += 1

        reward, feedback = self.grader.grade(
            action=action,
            scenario=self.scenario,  # type: ignore[arg-type]
            step=self.step_count,
            resolved=self.resolved,
        )
        self.total_reward += reward

        self._maybe_resolve(action)

        done = self._episode_goal_satisfied() or (
            self.scenario is not None and self.step_count >= self.scenario.max_steps
        )

        return IncidentObservation(
            alerts=self._get_active_alerts(),
            resolved_alerts=list(self.resolved),
            system_health=self.current_health,
            step_number=self.step_count,
            done=done,
            reward=reward,
            message=feedback,
        )

    @property
    def state(self) -> IncidentState:  # type: ignore[override]
        scenario_name = self.scenario.name if self.scenario is not None else "unknown"
        max_steps = self.scenario.max_steps if self.scenario is not None else 0
        return IncidentState(
            episode_id=self.episode_id,
            task_id=self.task_id,
            step_count=self.step_count,
            max_steps=max_steps,
            total_reward=self.total_reward,
            scenario_name=scenario_name,
        )

    def _update_health(self, action: IncidentAction) -> None:
        # Simple deterministic health update: remediation actions improve health more.
        delta = 0.02
        if action.action_type in {
            "scale_up",
            "restart",
            "rollback",
            "fix",
            "mitigate",
            "remediate",
            "isolate",
            "block",
        }:
            delta = 0.05
        self.current_health = max(0.0, min(1.0, self.current_health + delta))

    def _maybe_resolve(self, action: IncidentAction) -> None:
        if self.scenario is None:
            return
        if not action.alert_id or action.alert_id in self.resolved:
            return

        action_type = (action.action_type or "").lower().strip()
        resolution_actions = {
            "scale_up",
            "restart",
            "rollback",
            "fix",
            "mitigate",
            "remediate",
            "isolate",
            "block",
        }
        if action_type not in resolution_actions:
            return

        # Hard task: only allow resolving the next upstream link in the chain.
        if self.scenario.kind == "full_cascade_failure" and self.scenario.cascade_chain_alert_ids:
            chain = list(self.scenario.cascade_chain_alert_ids)
            expected_index = 0
            for cid in chain:
                if cid in self.resolved:
                    expected_index += 1
                else:
                    break
            expected_id = chain[expected_index] if expected_index < len(chain) else None
            if action.alert_id != expected_id:
                return

        self.resolved.append(action.alert_id)
        self._update_health(action)

    def _get_active_alerts(self) -> List[Alert]:
        # Never reveal internal root-cause flags.
        active = []
        for entry in self._alerts:
            if entry.alert.id not in self.resolved:
                active.append(entry.alert)
        return active

    def _all_critical_resolved(self) -> bool:
        for entry in self._alerts:
            if entry.alert.severity == "critical" and entry.alert.id not in self.resolved:
                return False
        return True

    def _episode_goal_satisfied(self) -> bool:
        """
        Episode ends when the task's success condition is met.

        Cascade (hard) tasks require every link in cascade_chain_alert_ids to be
        resolved - not only severity:critical rows - so agents earn graded rewards
        along the full chain and total score can reach 1.0.
        """
        if self.scenario is None:
            return False
        if (
            self.scenario.kind == "full_cascade_failure"
            and self.scenario.cascade_chain_alert_ids
        ):
            return all(
                cid in self.resolved
                for cid in self.scenario.cascade_chain_alert_ids
            )
        return self._all_critical_resolved()

