"""
Core environment logic for the Incident Response playground.

Ground truth (root-cause flags, chain order) is intentionally kept internal
and never returned in observations. Agents must infer from context.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import Alert, IncidentAction, IncidentObservation, IncidentState
    from .graders import IncidentGrader
    from .scenarios import Scenario, ScenarioGenerator
except ImportError:
    from models import Alert, IncidentAction, IncidentObservation, IncidentState
    from server.graders import IncidentGrader
    from server.scenarios import Scenario, ScenarioGenerator


@dataclass
class _InternalAlert:
    alert: Alert
    is_root_cause: bool


class IncidentResponseEnvironment(Environment):
    # HTTP server uses a single shared process-wide episode.
    # Callers must POST /reset before each new task.
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self.grader = IncidentGrader()
        self.scenario: Optional[Scenario] = None
        self.resolved: List[str] = []
        self._investigated: List[str] = []
        self.step_count: int = 0
        self.episode_id: str = ""
        self.task_id: str = "task_easy"
        self.total_reward: float = 0.0
        self.current_health: float = 1.0
        self._alerts: List[_InternalAlert] = []

    def reset(self, task_id: str = "task_easy", seed: int | None = None):
        self.scenario = ScenarioGenerator.generate(task_id, seed=seed)
        self.resolved = []
        self._investigated = []
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

    def step(self, action: IncidentAction) -> IncidentObservation:
        if self.scenario is None:
            self.reset(task_id="task_easy")

        self.step_count += 1

        reward, feedback = self.grader.grade(
            action=action,
            scenario=self.scenario,
            step=self.step_count,
            resolved=self.resolved,
            investigated=self._investigated,
        )

        # Track investigated alerts (only when grader gave positive reward for investigate)
        if (
            (action.action_type or "").lower().strip() == "investigate"
            and reward > 0
            and action.alert_id
            and action.alert_id not in self._investigated
        ):
            sk = self.scenario.kind
            if sk in ("full_cascade_failure", "alert_storm"):
                nxt = self._next_unresolved_chain_id()
                if nxt is not None and action.alert_id == nxt:
                    self._investigated.append(action.alert_id)
            else:
                self._investigated.append(action.alert_id)

        self.total_reward += reward
        self._maybe_resolve(action)

        # Update system health on resolution actions
        if (action.action_type or "").lower().strip() in {
            "scale_up", "restart", "rollback", "fix",
            "mitigate", "remediate", "isolate", "block",
        }:
            self._update_health()

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

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name=self.__class__.__name__,
            description=(
                "Incident triage environment with 4 tasks (easy/medium/hard/expert). "
                "HTTP mode: single shared episode per process. "
                "Always POST /reset before starting a new task. "
                "SUPPORTS_CONCURRENT_SESSIONS=False."
            ),
            version="1.0.0",
        )

    @property
    def state(self) -> IncidentState:
        return IncidentState(
            episode_id=self.episode_id,
            task_id=self.task_id,
            step_count=self.step_count,
            max_steps=self.scenario.max_steps if self.scenario else 0,
            total_reward=self.total_reward,
            scenario_name=self.scenario.name if self.scenario else "unknown",
        )

    def _update_health(self) -> None:
        """Improve system health slightly on each resolution action."""
        total = len(self._alerts)
        remaining = total - len(self.resolved)
        improvement = 0.05 * (remaining / max(total, 1))
        self.current_health = min(1.0, self.current_health + improvement)

    def _next_unresolved_chain_id(self) -> Optional[str]:
        if self.scenario is None or not self.scenario.cascade_chain_alert_ids:
            return None
        chain = list(self.scenario.cascade_chain_alert_ids)
        for cid in chain:
            if cid not in self.resolved:
                return cid
        return None

    def _maybe_resolve(self, action: IncidentAction) -> None:
        if self.scenario is None:
            return
        if not action.alert_id or action.alert_id in self.resolved:
            return

        action_type = (action.action_type or "").lower().strip()
        if action_type not in {
            "scale_up", "restart", "rollback", "fix",
            "mitigate", "remediate", "isolate", "block",
        }:
            return

        if self.scenario.kind == "cascading_db_failure":
            # Medium task: allow resolving root without requiring investigate.
            # (Grader can still award higher total when investigate is done first.)
            pass

        # Cascade tasks: correct next link only, and only after investigate on that link
        if (
            self.scenario.kind in ("full_cascade_failure", "alert_storm")
            and self.scenario.cascade_chain_alert_ids
        ):
            chain = list(self.scenario.cascade_chain_alert_ids)
            expected_index = sum(1 for cid in chain if cid in self.resolved)
            expected_id = chain[expected_index] if expected_index < len(chain) else None
            if action.alert_id != expected_id:
                return
            # Allow resolution without prior investigation (bonus handled in grader).

        self.resolved.append(action.alert_id)

    def _get_active_alerts(self) -> List[Alert]:
        return [
            entry.alert
            for entry in self._alerts
            if entry.alert.id not in self.resolved
        ]

    def _all_critical_resolved(self) -> bool:
        return all(
            entry.alert.id in self.resolved
            for entry in self._alerts
            if entry.alert.severity == "critical"
        )

    def _episode_goal_satisfied(self) -> bool:
        if self.scenario is None:
            return False
        if (
            self.scenario.kind in ("full_cascade_failure", "alert_storm")
            and self.scenario.cascade_chain_alert_ids
        ):
            return all(
                cid in self.resolved
                for cid in self.scenario.cascade_chain_alert_ids
            )
        return self._all_critical_resolved()
