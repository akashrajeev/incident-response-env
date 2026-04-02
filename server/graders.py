"""
Deterministic scoring logic for the incident response tasks.

Implements the 3 required tasks for judging:
- Task 1 (easy): single obvious alert, single correct action.
- Task 2 (medium): identify root cause among symptoms, penalize wasted steps.
- Task 3 (hard): resolve a cascade chain in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:
    from ..models import IncidentAction
    from .scenarios import Scenario
except ImportError:  # pragma: no cover
    from models import IncidentAction
    from server.scenarios import Scenario


@dataclass(frozen=True)
class IncidentGrader:
    _RESOLUTION_ACTIONS = {
        "scale_up",
        "restart",
        "rollback",
        "fix",
        "mitigate",
        "remediate",
        "isolate",
        "block",
    }

    def grade(
        self,
        *,
        action: IncidentAction,
        scenario: Scenario,
        step: int,
        resolved: List[str],
    ) -> Tuple[float, str]:
        if not action.alert_id:
            return 0.0, "No alert selected. Choose an alert_id to investigate or remediate."

        if action.alert_id in resolved:
            return 0.0, "That alert was already resolved. Pick an unresolved alert."

        alert_by_id = {a.id: a for a, _ in scenario.initial_alerts_internal}
        if action.alert_id not in alert_by_id:
            return 0.0, "Unknown alert_id. Pick one of the active alerts."

        action_type = (action.action_type or "").lower().strip()
        is_resolution = action_type in self._RESOLUTION_ACTIONS

        if scenario.kind == "disk_full":
            # Required Task 1 grading.
            if action.alert_id != "disk-alert-1":
                return 0.0, "Wrong alert. Triage the disk alert."
            if action_type == "scale_up":
                return 1.0, "Correct: scaled storage to relieve disk pressure."
            return 0.4, "Correct alert, but wrong action_type. Use scale_up."

        if scenario.kind == "cascading_db_failure":
            # Required Task 2 grading (meaningful reward across steps).
            root_id = scenario.root_cause_alert_id or "db-001"
            if action.alert_id == root_id:
                reward = 1.0 if step == 1 else 0.5
                feedback = "Addressed root cause." + (" Great first move." if step == 1 else "")
            else:
                reward = 0.1
                feedback = "You addressed a symptom; root cause remains unresolved."

            # End bonus: if this action (once resolved by environment) would complete all critical.
            # We approximate deterministically: if action targets the root cause with a resolution action.
            if is_resolution and action.alert_id == root_id:
                # Count remaining critical alerts besides this one.
                remaining_critical = [
                    a.id
                    for a, _ in scenario.initial_alerts_internal
                    if a.severity == "critical" and a.id not in resolved and a.id != action.alert_id
                ]
                if not remaining_critical:
                    reward += 0.3
                    feedback += " All critical alerts resolved. Bonus awarded."

            return min(1.0, reward), feedback

        # scenario.kind == "full_cascade_failure"
        chain = list(scenario.cascade_chain_alert_ids)
        if not chain:
            return 0.0, "Scenario misconfigured: missing cascade chain."

        # Determine expected next link in chain based on what's already resolved.
        expected_index = 0
        for cid in chain:
            if cid in resolved:
                expected_index += 1
            else:
                break

        expected_id = chain[expected_index] if expected_index < len(chain) else None
        if expected_id is None:
            return 0.0, "Cascade already resolved."

        if action.alert_id == expected_id:
            reward = 0.25
            feedback = "Correct next step in the cascade chain."
            # Bonus if notes mention the correct service/source.
            svc = alert_by_id[expected_id].source
            if svc and svc.lower() in (action.notes or "").lower():
                reward = min(1.0, reward + 0.1)
                feedback += " Reasoning mentions the correct service."
            # If this is the final link, cap to 1.0 total (environment accumulates).
            if expected_index == len(chain) - 1:
                feedback += " Chain complete."
            return reward, feedback

        return 0.05, "Out of order. Trace dependencies and resolve the next upstream failure first."

