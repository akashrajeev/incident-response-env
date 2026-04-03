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


ScenarioKind = Literal["disk_full", "cascading_db_failure", "full_cascade_failure", "alert_storm"]


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: ScenarioKind
    max_steps: int
    initial_health: float
    initial_alerts_internal: Sequence[Tuple[Alert, bool]]

    # Internal ground truth (never shown to agent)
    correct_action: str = ""
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
        elif task_id == "task_medium":
            return ScenarioGenerator._root_cause(n_services=n_services or random.randint(3, 5))
        elif task_id == "task_expert":
            return ScenarioGenerator._alert_storm()

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
        options = [
            (
                "disk",
                "storage-service",
                "Disk at 99% — scale up required",
                "Storage node out of space. Writes failing intermittently. Immediately scale up storage capacity to resolve.",
                "scale_up",
            ),
            (
                "cpu",
                "compute-node",
                "CPU at 100% — scale up required",
                "Sustained CPU spike. Request timeouts increasing. Scale up compute capacity to relieve pressure.",
                "scale_up",
            ),
            (
                "memory",
                "app-server",
                "Memory exhausted — restart required",
                "Memory pressure critical. OOM risk imminent. Restart the service to recover memory.",
                "restart",
            ),
        ]

        kind, source, title_template, desc, action = random.choice(options)
        n = random.randint(1, 9)
        alert_id = f"{kind}-alert-{n:02d}"

        return Scenario(
            name="disk_full",
            kind="disk_full",
            max_steps=5,
            initial_health=0.55,
            initial_alerts_internal=[
                (
                    Alert(
                        id=alert_id,
                        title=title_template,
                        severity="critical",
                        description=desc,
                        source=source,
                    ),
                    True,
                )
            ],
            root_cause_alert_id=alert_id,
            correct_action=action,
        )

    @staticmethod
    def _root_cause(*, n_services: int) -> Scenario:
        # Required Task 2 scenario: "cascading_db_failure"
        root_services = ["postgres", "mysql", "redis", "mongodb", "elasticsearch"]
        root_src = random.choice(root_services)
        root_id = f"{root_src}-{random.randint(1,5):03d}"

        symptom_ids: list[str] = []
        while len(symptom_ids) < 2:
            sid = f"svc-sym-{random.randint(10, 99)}"
            if sid != root_id and sid not in symptom_ids:
                symptom_ids.append(sid)

        # 3 alerts total: root + 2 downstream symptoms.
        root_alert = Alert(
            id=root_id,
            title=f"{root_src} connection timeout",
            severity="critical",
            description=f"{root_src} is unstable; connection pool exhaustion leads to timeouts affecting dependent services.",
            source=root_src,
        )
        sym1_src = random.choice(["api-gateway", "order-service", "user-service"])
        sym2_src = random.choice(["payment-service", "inventory-service", "notification-service"])
        symptom_alert_1 = Alert(
            id=symptom_ids[0],
            title="Elevated request errors",
            severity="medium",
            description=f"Downstream requests failing due to dependency errors originating in {root_src}.",
            source=sym1_src,
        )
        symptom_alert_2 = Alert(
            id=symptom_ids[1],
            title="Service calls timing out",
            severity="medium",
            description=f"Secondary symptom: failures correlate with {root_src} instability and propagated dependency timeouts.",
            source=sym2_src,
        )

        return Scenario(
            name="cascading_db_failure",
            kind="cascading_db_failure",
            max_steps=10,
            initial_health=0.6,
            initial_alerts_internal=[
                (root_alert, True),
                (symptom_alert_1, False),
                (symptom_alert_2, False),
            ],
            root_cause_alert_id=root_id,
        )

    @staticmethod
    def _cascade_chain(*, chain_length: int) -> Scenario:
        # Required Task 3 scenario: "full_cascade_failure"
        all_svcs = [
            "auth-service",
            "user-service",
            "order-service",
            "payment-service",
            "inventory-service",
            "search-service",
            "notification-service",
            "billing-service",
        ]
        chain = random.sample(all_svcs, k=min(chain_length, len(all_svcs)))

        chain_ids: list[str] = []
        internal: list[Tuple[Alert, bool]] = []
        descriptions = [
            "{svc} is returning errors. Upstream dependencies may be affected.",
            "{svc} health check failing. Dependency issues detected.",
            "{svc} connection pool exhausted. Service degraded.",
            "{svc} error rate elevated. Downstream impact observed.",
            "{svc} latency spike. Cascading effects possible.",
        ]

        for i, svc in enumerate(chain):
            aid = f"inc-{random.randint(100, 999)}"
            while aid in chain_ids:  # keep IDs unique within the chain
                aid = f"inc-{random.randint(100, 999)}"

            chain_ids.append(aid)
            desc = random.choice(descriptions).replace("{svc}", svc)

            internal.append(
                (
                    Alert(
                        id=aid,
                        title=f"{svc} failing",
                        severity="critical" if i == 0 else "high",
                        description=desc,
                        source=svc,
                    ),
                    i == 0,  # internal root-cause flag; not shown to agents
                )
            )

        NOISE_ALERTS = [
            ("High memory usage", "Memory utilization above 85%. Performance may degrade.", "low"),
            ("Slow query detected", "Database query taking >5s. Possible index issue.", "low"),
            ("Certificate expiring", "TLS certificate expires in 7 days.", "low"),
            ("Disk usage warning", "Disk at 75%. Not yet critical.", "low"),
        ]
        n_noise = 1 if chain_length <= 4 else 2
        used_ids = set(chain_ids)
        for title, desc, severity in random.sample(NOISE_ALERTS, n_noise):
            noise_id = f"noise-{random.randint(100,999)}"
            while noise_id in used_ids:
                noise_id = f"noise-{random.randint(100,999)}"
            used_ids.add(noise_id)
            internal.append(
                (
                    Alert(
                        id=noise_id,
                        title=title,
                        severity=severity,
                        description=desc,
                        source="monitoring",
                    ),
                    False,
                )
            )

        return Scenario(
            name="full_cascade_failure",
            kind="full_cascade_failure",
            max_steps=20,
            initial_health=0.45,
            initial_alerts_internal=internal,
            root_cause_alert_id=chain_ids[0] if chain_ids else None,
            cascade_chain_alert_ids=tuple(chain_ids),
        )

    @staticmethod
    def _alert_storm() -> Scenario:
        # Required Task 4 scenario: "alert_storm" (signal vs noise)
        all_svcs = [
            "auth-service",
            "user-service",
            "order-service",
            "payment-service",
            "inventory-service",
            "search-service",
            "notification-service",
            "billing-service",
        ]
        chain_svcs = random.sample(all_svcs, k=3)

        real_ids: list[str] = []
        real_alerts: list[Tuple[Alert, bool]] = []
        for i, svc in enumerate(chain_svcs):
            rid = f"real-{random.randint(100, 999)}"
            while rid in real_ids:
                rid = f"real-{random.randint(100, 999)}"

            real_ids.append(rid)
            next_svc = chain_svcs[i + 1] if i + 1 < len(chain_svcs) else None
            if next_svc:
                desc = f"{svc} is failing. Logs show repeated connection refused errors pointing to {next_svc}."
            else:
                desc = (
                    f"{svc} is failing. No upstream dependency identified. This appears to be the origin of the incident."
                )

            real_alerts.append(
                (
                    Alert(
                        id=rid,
                        title=f"{svc} failing",
                        severity="critical" if i == 0 else "high",
                        description=desc,
                        source=svc,
                    ),
                    i == 0,
                )
            )

        noise_specs = [
            ("low", "Certificate expiring soon", "Routine warning: cert expiry within 7 days.", "cert-expiry"),
            (
                "low",
                "Backup disk usage high",
                "Routine warning: backup disk at 78% capacity; within expected range.",
                "storage-service",
            ),
            (
                "low",
                "CPU nominal",
                "Routine warning: CPU within normal range; no action required.",
                "compute-node",
            ),
            (
                "low",
                "Scheduled maintenance",
                "Routine warning: scheduled maintenance window started; services may flap temporarily.",
                "ops",
            ),
            (
                "low",
                "Log rotation complete",
                "Routine warning: log rotation finished successfully; disk usage stabilized.",
                "storage-service",
            ),
            (
                "low",
                "Memory within threshold",
                "Routine warning: memory within threshold; no OOM risk indicated.",
                "app-server",
            ),
            (
                "low",
                "Disk cleanup pending",
                "Routine warning: disk cleanup pending; recommended during low-traffic periods.",
                "storage-service",
            ),
        ]

        noise_alerts: list[Tuple[Alert, bool]] = []
        for i in range(7):
            nid = f"noise-{i+1:03d}"
            # Noise alerts should still use valid Alert.severity literals.
            severity = "low"
            _, title, description, source = noise_specs[i]
            noise_alerts.append(
                (
                    Alert(
                        id=nid,
                        title=title,
                        severity=severity,
                        description=description,
                        source=source,
                    ),
                    False,
                )
            )

        all_alerts = real_alerts + noise_alerts
        random.shuffle(all_alerts)

        return Scenario(
            name="alert_storm",
            kind="alert_storm",
            max_steps=15,
            initial_health=0.5,
            initial_alerts_internal=all_alerts,
            root_cause_alert_id=real_ids[0] if real_ids else None,
            cascade_chain_alert_ids=tuple(real_ids),
        )


if __name__ == "__main__":
    for task in ["task_easy", "task_medium", "task_hard", "task_expert"]:
        s = ScenarioGenerator.generate(task)
        print(
            f"{task}: {s.kind}, {len(s.initial_alerts_internal)} alerts, "
            f"chain={s.cascade_chain_alert_ids}, correct_action='{s.correct_action}'"
        )

