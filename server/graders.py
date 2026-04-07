"""
Deterministic scoring logic for the incident response tasks.

Scoring philosophy (progress-based):
- task_easy:   perfect action ~= 0.95; wrong resolution type on correct alert ~= 0.15
- task_medium: investigate root ~= 0.35, resolve root after investigate ~= 0.55 (perfect ~= 0.90)
               resolving root without investigate ~= 0.25
- task_hard:   ordered chain. Score ~ proportional to fraction of links cleared:
               investigate expected link ~= 0.05/n, resolve expected link ~= 0.90/n (perfect ~= 0.95)
- task_expert: same scoring as hard; difficulty comes from harder scenarios (more distractors, tighter budgets).

Key invariants:
- Perfect runs for hard and expert sum to exactly 1.0 over the full chain
- Noise alerts always score 0.0
- Wrong chain / wrong alert scores 0.0 (no exploration bonuses)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:
    from ..models import IncidentAction
    from .scenarios import Scenario
except ImportError:
    from models import IncidentAction
    from server.scenarios import Scenario


@dataclass(frozen=True)
class IncidentGrader:

    _RESOLUTION_ACTIONS = {
        "scale_up", "restart", "rollback", "fix",
        "mitigate", "remediate", "isolate", "block",
    }

    def grade(
        self,
        *,
        action: IncidentAction,
        scenario: Scenario,
        step: int,
        resolved: List[str],
        investigated: List[str],
    ) -> Tuple[float, str]:

        # --- Universal guard checks ---
        if not action.alert_id:
            return 0.0, "No alert selected. Choose an alert_id from the active alerts list."

        if action.alert_id in resolved:
            return 0.0, "That alert is already resolved. Pick an unresolved alert."

        alert_by_id = {a.id: a for a, _ in scenario.initial_alerts_internal}
        if action.alert_id not in alert_by_id:
            return 0.0, "Unknown alert_id. Pick one of the active alerts."

        action_type = (action.action_type or "").lower().strip()
        is_resolution = action_type in self._RESOLUTION_ACTIONS

        # ======================================================
        # TASK EASY — single alert, single correct action
        # ======================================================
        if scenario.kind == "disk_full":
            root_id = scenario.root_cause_alert_id
            correct_action = (getattr(scenario, "correct_action", None) or "scale_up").lower()

            if action.alert_id != root_id:
                return 0.0, "Wrong alert. Triage the single critical alert shown."

            if action_type == correct_action:
                return 0.95, f"Correct: {correct_action} resolved the {alert_by_id[action.alert_id].source} issue."

            if is_resolution:
                return 0.15, f"Correct alert, but wrong action. The description says to use {correct_action}."

            return 0.0, f"Use a resolution action. The description tells you exactly what to do: {correct_action}."

        # ======================================================
        # TASK MEDIUM — investigate root then remediate
        # ======================================================
        if scenario.kind == "cascading_db_failure":
            root_id = scenario.root_cause_alert_id or ""

            if action.alert_id != root_id:
                return 0.0, (
                    "That is a downstream symptom, not the root cause. "
                    "Find and address the critical severity alert first."
                )

            if action_type == "investigate":
                if root_id in investigated:
                    return 0.0, "Already investigated the root cause. Now apply remediate or fix to resolve it."
                return 0.35, "Root cause identified. Now apply remediate or fix to resolve it."

            if is_resolution:
                if root_id not in investigated:
                    return 0.40, (
                        "Resolved root cause without investigation. "
                        "Investigate first next time for a higher score."
                    )
                return 0.55, "Correct: investigated then resolved the root cause. All clear."

            return 0.1, "Use investigate first, then a resolution action on the root cause."

        # ======================================================
        # TASK HARD — ordered cascade + noise (see scenarios for n)
        # ======================================================
        if scenario.kind == "full_cascade_failure":
            chain = list(scenario.cascade_chain_alert_ids)
            if not chain:
                return 0.0, "Scenario misconfigured: missing cascade chain."

            n = len(chain)
            expected_index = sum(1 for cid in chain if cid in resolved)
            if expected_index >= n:
                return 0.0, "Cascade already fully resolved."
            expected_id = chain[expected_index]

            # Noise alert — always 0
            if action.alert_id not in chain:
                return 0.0, (
                    "That alert is not part of the incident chain. "
                    "Ignore low-severity monitoring alerts and focus on the high/critical service failures."
                )

            # Wrong chain alert
            if action.alert_id != expected_id:
                if action_type == "investigate" and step == 1:
                    return 0.0, (
                        "Not the current upstream blocker. Re-check dependencies and focus on the earliest "
                        "unresolved service in the chain."
                    )
                return 0.0, (
                    "Wrong service. That is not the current upstream blocker. "
                    "Trace dependencies from the earliest unresolved failure in the chain."
                )

            # Correct alert — investigate
            if action_type == "investigate":
                if action.alert_id in investigated:
                    return 0.0, (
                        f"Already investigated {action.alert_id}. "
                        "Now apply a resolution action: fix, restart, or remediate."
                    )
                return round(0.05 / n, 3), (
                    f"Correct alert identified. Now apply fix, restart, or remediate to {action.alert_id}."
                )

            # Correct alert — non-resolution non-investigate
            if not is_resolution:
                return 0.0, "Use a resolution action: fix, restart, remediate, scale_up, etc."

            # Correct alert — resolution
            # Keep totals below 1.0 even with small exploration rewards.
            reward = round(0.85 / n, 3)
            if expected_index + 1 < n:
                next_id = chain[expected_index + 1]
                next_alert = alert_by_id.get(next_id)
                next_source = next_alert.source if next_alert else "unknown"
                return reward, (
                    f"Correct. {action.alert_id} resolved. "
                    f"Next upstream failure to address involves {next_source}. "
                    "Trace its dependencies in the active alerts list."
                )
            completion_bonus = 0.04
            return round(min(0.95, reward + completion_bonus), 3), (
                f"Correct. {action.alert_id} resolved. Cascade complete — all services restored."
            )

        # ======================================================
        # TASK EXPERT — ordered cascade + noise (same per-link totals as hard)
        # ======================================================
        if scenario.kind == "alert_storm":
            chain = list(scenario.cascade_chain_alert_ids)
            n = len(chain)
            expected_index = sum(1 for cid in chain if cid in resolved)
            if expected_index >= n:
                return 0.0, "All real alerts resolved."
            expected_id = chain[expected_index]

            # Noise alert — always 0
            if action.alert_id not in chain:
                return 0.0, (
                    "Noise alert — ignore low-severity monitoring alerts. "
                    "Find the critical upstream service failure."
                )

            # Wrong chain alert
            if action.alert_id != expected_id:
                if action_type == "investigate" and step == 1:
                    return 0.0, (
                        "Not the current upstream blocker. Re-check dependencies and focus on the earliest "
                        "unresolved service in the chain."
                    )
                return 0.0, (
                    "Wrong alert. Stop and reconsider. "
                    "Find the upstream failure that is blocking the rest of the chain."
                )

            # Correct alert — investigate
            if action_type == "investigate":
                if action.alert_id in investigated:
                    return 0.0, (
                        f"Already investigated {action.alert_id}. "
                        "Now apply a resolution action: fix, restart, or remediate."
                    )
                return round(0.05 / n, 3), (
                    f"Correct alert identified at {action.alert_id}. "
                    "Now apply fix, restart, or remediate to resolve it."
                )

            # Correct alert — non-resolution
            if not is_resolution:
                return 0.0, "Use a resolution action: fix, restart, remediate, scale_up, etc."

            # Correct alert — resolution
            reward = round(0.80 / n, 3)
            if expected_index + 1 < n:
                next_id = chain[expected_index + 1]
                next_alert = alert_by_id.get(next_id)
                next_source = next_alert.source if next_alert else "unknown"
                return reward, (
                    f"Correct. {action.alert_id} resolved. "
                    f"Next upstream failure to address involves {next_source}. "
                    "Trace its dependencies in the active alerts list."
                )
            completion_bonus = 0.04
            return round(min(0.95, reward + completion_bonus), 3), (
                f"Correct. {action.alert_id} resolved. Expert cascade complete."
            )

        return 0.0, "Unknown scenario kind."
