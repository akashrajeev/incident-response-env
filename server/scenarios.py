"""
Synthetic incident scenario generation (Tasks 1–3).

Scenarios contain internal ground truth (e.g., root-cause IDs / chain order) that
must never be returned to agents directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

try:
    from ..models import Alert
except ImportError:  # pragma: no cover
    from models import Alert


ScenarioKind = Literal["disk_full", "cascading_db_failure", "full_cascade_failure"]


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: ScenarioKind
    max_steps: int
    initial_health: float
    initial_alerts_internal: Sequence[Tuple[Alert, bool]]

    # Internal ground truth (never shown to agent)
    root_cause_alert_id: str | None = None
    cascade_chain_alert_ids: Sequence[str] = ()


class ScenarioGenerator:
    SERVICE_NAMES = [
        "auth-service",
        "payment-service",
        "user-db",
        "order-service",
        "cache-layer",
        "api-gateway",
        "storage-service",
        "database",
        "user-service",
    ]

    @staticmethod
    def generate(
        task_id: str, seed: int | None = None, *, n_services: int | None = None, chain_length: int | None = None
    ) -> Scenario:
        """
        Produce unlimited variations via randomness.

        Note: env.reset() may also seed randomness; passing seed here makes generation
        self-contained for judge harnesses that call ScenarioGenerator directly.
        """

        if seed is not None:
            random.seed(seed)

        if task_id == "task_easy":
            return ScenarioGenerator._single_alert()

        if task_id == "task_medium":
            return ScenarioGenerator._root_cause(n_services=n_services or random.randint(3, 5))

        # task_hard (or anything else) maps to cascade chain
        return ScenarioGenerator._cascade_chain(chain_length=chain_length or random.randint(3, 5))

    @staticmethod
    def _pick_services(k: int) -> list[str]:
        names = list(ScenarioGenerator.SERVICE_NAMES)
        random.shuffle(names)
        return names[:k]

    @staticmethod
    def _single_alert() -> Scenario:
        # Required Task 1 scenario: "disk_full"
        return Scenario(
            name="disk_full",
            kind="disk_full",
            max_steps=3,
            initial_health=0.55,
            initial_alerts_internal=[
                (
                    Alert(
                        id="disk-alert-1",
                        title="Disk at 99%",
                        severity="critical",
                        description="Storage node nearly out of space. Writes failing intermittently.",
                        source="storage-service",
                    ),
                    True,
                )
            ],
            root_cause_alert_id="disk-alert-1",
        )

    @staticmethod
    def _root_cause(*, n_services: int) -> Scenario:
        # Required Task 2 scenario: "cascading_db_failure"
        services = ScenarioGenerator._pick_services(max(3, n_services))
        db_service = "database"
        if db_service not in services:
            services[0] = db_service

        api_service = "api-gateway" if "api-gateway" in services else services[1]
        pay_service = "payment-service" if "payment-service" in services else services[2]

        alerts: list[Tuple[Alert, bool]] = [
            (
                Alert(
                    id="db-001",
                    title="DB connection timeout",
                    severity="critical",
                    description="Database pool exhausted; connections timing out. Downstream services likely impacted.",
                    source=db_service,
                ),
                True,
            ),
            (
                Alert(
                    id="api-002",
                    title="High error rate",
                    severity="medium",
                    description="5xx rate elevated. Errors correlate with DB timeout spikes.",
                    source=api_service,
                ),
                False,
            ),
            (
                Alert(
                    id="pay-003",
                    title="Requests failing",
                    severity="medium",
                    description="Payment calls failing with dependency errors (DB).",
                    source=pay_service,
                ),
                False,
            ),
        ]

        # Optionally add one extra noisy alert for variety.
        if n_services >= 4:
            noise_src = services[3]
            alerts.append(
                (
                    Alert(
                        id="aux-004",
                        title="Cache miss rate increased",
                        severity="low",
                        description="Cache miss rate above baseline; could be secondary effect.",
                        source=noise_src,
                    ),
                    False,
                )
            )

        return Scenario(
            name="cascading_db_failure",
            kind="cascading_db_failure",
            max_steps=8,
            initial_health=0.6,
            initial_alerts_internal=alerts,
            root_cause_alert_id="db-001",
        )

    @staticmethod
    def _cascade_chain(*, chain_length: int) -> Scenario:
        # Required Task 3 scenario: "full_cascade_failure"
        chain_services = ["auth-service", "user-service", "order-service", "payment-service"]
        if chain_length != 4:
            # Allow variable length, but keep the "auth → user → order → payment" prefix
            extras = [s for s in ScenarioGenerator.SERVICE_NAMES if s not in chain_services]
            random.shuffle(extras)
            chain_services = (chain_services + extras)[: max(3, chain_length)]

        chain_ids: list[str] = []
        internal: list[Tuple[Alert, bool]] = []

        for i, svc in enumerate(chain_services):
            aid = f"svc-{i+1:03d}"
            chain_ids.append(aid)
            next_svc = chain_services[i + 1] if i + 1 < len(chain_services) else None
            hint = (
                f"Downstream impact observed: {next_svc} reporting dependency errors."
                if next_svc
                else "Downstream impact widespread."
            )
            internal.append(
                (
                    Alert(
                        id=aid,
                        title=f"{svc} failing",
                        severity="critical" if i == 0 else "high",
                        description=f"{svc} error spike. {hint}",
                        source=svc,
                    ),
                    i == 0,  # treat first link as "root cause" internally
                )
            )

        return Scenario(
            name="full_cascade_failure",
            kind="full_cascade_failure",
            max_steps=max(10, len(chain_ids) * 3),
            initial_health=0.45,
            initial_alerts_internal=internal,
            root_cause_alert_id=chain_ids[0],
            cascade_chain_alert_ids=tuple(chain_ids),
        )

