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

    def _notes_bonus(self, action: IncidentAction, alert_obj) -> float:
        notes = (action.notes or "").lower()
        source = (alert_obj.source or "").lower()
        if source and source in notes:
            return 0.1
        return 0.0

    def _time_decay(self, reward: float, step: int) -> float:
        if step <= 2:
            return reward
        decay = max(0.6, 1.0 - 0.04 * (step - 2))
        return round(reward * decay, 4)

    def grade(
        self,
        *,
        action: IncidentAction,
        scenario: Scenario,
        step: int,
        resolved: List[str],
        investigated: List[str],
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
            root_id = scenario.root_cause_alert_id
            if action.alert_id != root_id:
                return 0.0, "Wrong alert. Triage the disk alert."
            correct_action = getattr(scenario, "correct_action", "scale_up") or "scale_up"
            correct_action = correct_action.lower()

            if action_type == correct_action:
                reward = 1.0
                if correct_action == "scale_up":
                    feedback = "Correct: scaled storage to relieve disk pressure."
                elif correct_action == "restart":
                    feedback = "Correct: restarted the overloaded service to stabilize."
                elif correct_action == "mitigate":
                    feedback = "Correct: applied mitigation to restore stability."
                else:
                    feedback = f"Correct: applied the recommended action ({correct_action})."
            else:
                if is_resolution:
                    reward = 0.4
                    feedback = f"Correct alert, but wrong action_type. Use {correct_action}."
                else:
                    reward = 0.0
                    feedback = f"Correct alert, but choose a resolution action ({correct_action}), not {action_type or 'investigate'}."

            reward = min(1.0, reward + self._notes_bonus(action, alert_by_id[action.alert_id]))
            reward = self._time_decay(reward, step)
            return reward, feedback

        if scenario.kind == "cascading_db_failure":
            root_id = scenario.root_cause_alert_id or "db-001"

            # Step 1 must be investigate on root cause.
            # Step 2+ can be resolution.
            if action.alert_id == root_id:
                if action_type == "investigate":
                    if root_id not in resolved:
                        return 0.6, "Root cause identified. Now apply fix or remediate to resolve it."
                    return 0.0, "Already investigated. Apply a resolution action."
                elif is_resolution:
                    # Check if they investigated first.
                    # If step == 1, they skipped investigation — partial reward only.
                    if step == 1:
                        return 0.4, "Resolved root cause but skipped investigation. Investigate first for full score."
                    return 0.9, "Correct: investigated then resolved root cause. Well done."
                return 0.1, "Use investigate first, then a resolution action on the root cause."
            else:
                return 0.0, "That is a symptom not the root cause. Find and address the critical severity alert first."

        if scenario.kind == "alert_storm":
            # Expert grading: reward only for ordered progress.
            chain = list(scenario.cascade_chain_alert_ids)
            expected_index = sum(1 for cid in chain if cid in resolved)
            expected_id = chain[expected_index] if expected_index < len(chain) else None
            if expected_id is None:
                return 0.0, "All real alerts resolved."

            if action_type == "investigate":
                if action.alert_id != expected_id:
                    if step == 1:
                        return 0.05, (
                            "Investigating - this may not be the root cause. "
                            "Try a different alert. Look for critical severity alerts "
                            "that other services depend on."
                        )
                    elif step == 2:
                        return 0.02, (
                            "Still not the right alert. Switch to a different one - "
                            "check which service has the most downstream dependencies."
                        )
                    else:
                        return 0.0, (
                            "Wrong alert. Stop investigating this service. "
                            "Find the critical upstream failure and remediate it."
                        )
                if action.alert_id in investigated:
                    return 0.0, "Already investigated. Apply a resolution action."
                feedback = (
                    f"Root cause confirmed at {expected_id}. Now apply fix, restart, or remediate to {expected_id} - do not move to another alert yet."
                )
                return 0.10, feedback

            if not is_resolution:
                return 0.0, "Use investigate first, then apply a resolution action."

            if action.alert_id != expected_id:
                return 0.0, (
                    "Incorrect alert. In the expert scenario, start from the alert with "
                    "the highest severity and trace its upstream dependencies. "
                    "Look for the service that other failing services depend on."
                )

            reward = round(0.6 / len(chain), 3)
            next_index = expected_index + 1
            if next_index < len(chain):
                feedback = (
                    f"Correct. {expected_id} ({alert_by_id[expected_id].source}) resolved. "
                    "System health improving. Identify and resolve the next upstream failure in the chain."
                )
            else:
                feedback = f"Correct. {expected_id} resolved. Cascade complete - all services restored."
            return reward, feedback

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

        if action_type == "investigate":
            if action.alert_id != expected_id:
                return 0.0, (
                    "Wrong alert. That service is not the current upstream blocker. "
                    "Re-read the active alerts and trace dependencies - resolve the earliest upstream failure first."
                )
            if action.alert_id in investigated:
                return 0.0, f"Already investigated {action.alert_id}. Stop investigating and apply a resolution action: fix, restart, or remediate."
            feedback = (
                f"Correct alert identified. Now remediate {expected_id} to continue the chain."
            )
            return round(0.15 / len(chain), 3), feedback

        if not is_resolution:
            return 0.0, "Use investigate first, then apply a resolution action."

        if action.alert_id != expected_id:
            return 0.0, (
                "Wrong alert. That service is not the current upstream blocker. "
                "Re-read the active alerts and trace dependencies - resolve the earliest upstream failure first."
            )

        n = len(chain)
        if expected_id not in investigated:
            reward = round(0.5 / n, 3)  # half credit for skipping investigation
            return reward, (
                f"Partially correct - resolved {expected_id} but skipped investigation. "
                f"Investigate before remediating for full score."
            )
        reward = round(1.0 / n, 3)
        next_index = expected_index + 1
        if next_index < len(chain):
            feedback = "Correct step: investigated then resolved. Move to next alert."
        else:
            feedback = f"Correct. {expected_id} resolved. Cascade complete - all services restored."
        return reward, feedback

