"""
Synthetic incident scenario generation (Tasks 1-4).

Design principles:
- task_easy:   1 alert, 1 correct action — tight step budget (little room for mistakes)
- task_medium: 1 root + 2 symptoms — investigate then remediate; reduced max_steps
- task_hard:   chain of exactly 5 services + 2 noise alerts — longer chain, tighter budget vs optimal path
- task_expert: chain of exactly 5 services + 3 noise — more distraction, tighter budget

Chain length is FIXED per task (not random). Tighter max_steps makes wasted steps costly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Literal, Sequence, Tuple

try:
    from ..models import Alert
except ImportError:
    from models import Alert


ScenarioKind = Literal["disk_full", "cascading_db_failure", "full_cascade_failure", "alert_storm"]

NOISE_POOL = [
    ("High memory usage",      "Memory utilization above 85%. Performance may degrade slightly.",           "low",  "monitoring"),
    ("Slow query detected",    "Database query taking >5s. Possible index issue, non-critical.",            "low",  "monitoring"),
    ("Certificate expiring",   "TLS certificate expires in 7 days. Routine rotation recommended.",         "low",  "ops"),
    ("Disk usage warning",     "Backup disk at 75%. Not yet critical, monitor trend.",                     "low",  "storage-service"),
    ("Log rotation complete",  "Log rotation finished. Disk usage stabilized within expected range.",      "low",  "ops"),
    ("Scheduled maintenance",  "Maintenance window active. Minor service flaps expected and safe.",        "low",  "ops"),
    ("Memory within threshold","Memory within normal bounds. No OOM risk indicated at this time.",         "low",  "app-server"),
]

# High/critical distractors that are NOT part of the cascade chain.
# These are intended to be tempting but ultimately irrelevant to the ordered chain tasks.
RED_HERRING_POOL = [
    (
        "Payment processor latency spike",
        "Third-party payment processor latency is elevated. This is noisy and intermittent; investigate before taking action.",
        "high",
        "payment-service",
    ),
    (
        "Auth token validation failures",
        "Spike in auth token validation errors. Could be caused by downstream timeouts rather than auth itself.",
        "high",
        "auth-service",
    ),
    (
        "Queue backlog growing",
        "Message queue backlog is growing. Symptoms may clear once upstream services recover.",
        "high",
        "notification-service",
    ),
    (
        "Suspected DDoS traffic",
        "Traffic pattern looks suspicious. Could be a false positive during incident cascades; avoid overreacting.",
        "critical",
        "edge-gateway",
    ),
]

ALL_SERVICES = [
    "auth-service", "user-service", "order-service", "payment-service",
    "inventory-service", "search-service", "notification-service", "billing-service",
]

ROOT_DB_SERVICES = ["postgres", "mysql", "redis", "mongodb", "elasticsearch"]

EASY_OPTIONS = [
    {
        "kind": "disk",
        "source": "storage-service",
        "title": "Disk at 99% — scale up required",
        "description": (
            "Storage node is out of disk space. Write operations are failing intermittently. "
            "Immediately scale up storage capacity to resolve."
        ),
        "correct_action": "scale_up",
    },
    {
        "kind": "cpu",
        "source": "compute-node",
        "title": "CPU at 100% — scale up required",
        "description": (
            "Sustained CPU spike detected. Request timeouts are increasing rapidly. "
            "Scale up compute capacity to relieve the pressure."
        ),
        "correct_action": "scale_up",
    },
    {
        "kind": "memory",
        "source": "app-server",
        "title": "Memory exhausted — restart required",
        "description": (
            "JVM heap at 98%, OOM risk is imminent. Service is degraded. "
            "Restart the service to recover memory and restore normal operation."
        ),
        "correct_action": "restart",
    },
    {
        "kind": "network",
        "source": "load-balancer",
        "title": "Network packet loss — mitigate required",
        "description": (
            "Load balancer detecting 15% packet loss. "
            "Mitigate the network issue to restore stable routing."
        ),
        "correct_action": "mitigate",
    },
]


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: ScenarioKind
    max_steps: int
    initial_health: float
    initial_alerts_internal: Sequence[Tuple[Alert, bool]]
    correct_action: str = ""
    root_cause_alert_id: str | None = None
    cascade_chain_alert_ids: Sequence[str] = ()
    metadata: Dict = field(default_factory=dict)


class ScenarioGenerator:

    @staticmethod
    def generate(task_id: str, seed: int | None = None, **_kwargs) -> Scenario:
        if seed is not None:
            random.seed(seed)

        if task_id == "task_easy":
            return ScenarioGenerator._single_alert()
        if task_id == "task_medium":
            return ScenarioGenerator._root_cause()
        if task_id == "task_expert":
            return ScenarioGenerator._alert_storm()
        # task_hard
        return ScenarioGenerator._cascade_chain(chain_length=6, n_noise=2)

    # ------------------------------------------------------------------
    # Task 1 — easy
    # ------------------------------------------------------------------
    @staticmethod
    def _single_alert() -> Scenario:
        # Weight easy cases so non-scale_up actions happen ~30% of the time.
        # disk/cpu are common; memory/network less common but important.
        opt = random.choices(EASY_OPTIONS, weights=[0.35, 0.35, 0.15, 0.15], k=1)[0]
        n = random.randint(1, 9)
        alert_id = f"{opt['kind']}-alert-{n:02d}"
        return Scenario(
            name="disk_full",
            kind="disk_full",
            max_steps=3,
            initial_health=0.55,
            initial_alerts_internal=[
                (
                    Alert(
                        id=alert_id,
                        title=opt["title"],
                        severity="critical",
                        description=opt["description"],
                        source=opt["source"],
                    ),
                    True,
                )
            ],
            root_cause_alert_id=alert_id,
            correct_action=opt["correct_action"],
        )

    # ------------------------------------------------------------------
    # Task 2 — medium
    # ------------------------------------------------------------------
    @staticmethod
    def _root_cause() -> Scenario:
        root_src = random.choice(ROOT_DB_SERVICES)
        root_id = f"{root_src}-{random.randint(1, 5):03d}"

        sym_srcs = random.sample(
            ["api-gateway", "order-service", "user-service",
             "payment-service", "inventory-service", "notification-service"],
            k=2,
        )
        sym_ids = [f"svc-sym-{random.randint(10, 99)}", f"svc-sym-{random.randint(10, 99)}"]
        while sym_ids[0] == sym_ids[1]:
            sym_ids[1] = f"svc-sym-{random.randint(10, 99)}"

        # One symptom is intentionally "tempting" (high/critical) to induce mistakes.
        tempting_idx = random.randint(0, 1)
        sym0_sev = "high" if tempting_idx == 0 else "medium"
        sym1_sev = "high" if tempting_idx == 1 else "medium"

        alerts = [
            (
                Alert(
                    id=root_id,
                    title=f"{root_src} connection timeout",
                    severity="critical",
                    description=(
                        f"{root_src} is unstable; connection pool exhaustion is causing "
                        "timeouts in dependent services. Resolve this to restore dependent services."
                    ),
                    source=root_src,
                ),
                True,
            ),
            (
                Alert(
                    id=sym_ids[0],
                    title="Service failing — elevated request errors",
                    severity=sym0_sev,
                    description=(
                        f"Downstream requests failing. Errors correlate with {root_src} instability."
                    ),
                    source=sym_srcs[0],
                ),
                False,
            ),
            (
                Alert(
                    id=sym_ids[1],
                    title="Critical dependency timeout",
                    severity=sym1_sev,
                    description=(
                        f"Timeout errors propagating from {root_src} dependency failures."
                    ),
                    source=sym_srcs[1],
                ),
                False,
            ),
        ]
        return Scenario(
            name="cascading_db_failure",
            kind="cascading_db_failure",
            max_steps=6,
            initial_health=0.6,
            initial_alerts_internal=alerts,
            root_cause_alert_id=root_id,
        )

    # ------------------------------------------------------------------
    # Task 3 — hard (chain=5, noise=2)
    # Task 4 — expert (chain=5, noise=3) via _alert_storm
    # ------------------------------------------------------------------
    @staticmethod
    def _cascade_chain(*, chain_length: int, n_noise: int) -> Scenario:
        chain_svcs = random.sample(ALL_SERVICES, k=chain_length)

        chain_ids: list[str] = []
        internal: list[Tuple[Alert, bool]] = []
        used_ids: set[str] = set()

        templates = [
            ("{svc} is failing. Logs show repeated connection refused errors pointing to {next}.", True),
            ("{svc} is timing out under load. Downstream traces implicate {next} as upstream.", True),
            ("{svc} is unhealthy. Dependency checks show {next} is unreachable.", True),
            # Occasionally ambiguous on purpose (hard should be harder than medium).
            ("{svc} is unhealthy. Dependency checks are inconclusive; investigate service interactions.", False),
        ]

        for i, svc in enumerate(chain_svcs):
            aid = f"inc-{random.randint(100, 999)}"
            while aid in used_ids:
                aid = f"inc-{random.randint(100, 999)}"
            used_ids.add(aid)
            chain_ids.append(aid)
            next_svc = chain_svcs[i + 1] if i + 1 < len(chain_svcs) else None
            if next_svc:
                tmpl, has_next = random.choice(templates)
                desc = tmpl.format(svc=svc, next=next_svc) if has_next else tmpl.format(svc=svc, next=next_svc)
            else:
                desc = f"{svc} is failing. No upstream dependency identified - this appears to be the origin."
            internal.append((
                Alert(
                    id=aid,
                    title=f"{svc} failing",
                    severity="critical" if i == 0 else "high",
                    description=desc,
                    source=svc,
                ),
                i == 0,
            ))

        # Low-severity noise
        for title, desc, severity, source in random.sample(NOISE_POOL, n_noise):
            nid = f"noise-{random.randint(100, 999)}"
            while nid in used_ids:
                nid = f"noise-{random.randint(100, 999)}"
            used_ids.add(nid)
            internal.append((
                Alert(id=nid, title=title, severity=severity, description=desc, source=source),
                False,
            ))

        # Add 2 high-severity red herrings that are not in the chain.
        herrings = [h for h in RED_HERRING_POOL if h[2] in ("high", "critical")]
        for i, (title, desc, severity, source) in enumerate(random.sample(herrings, k=2), start=1):
            hid = f"decoy-{i:03d}-{random.randint(100, 999)}"
            while hid in used_ids:
                hid = f"decoy-{i:03d}-{random.randint(100, 999)}"
            used_ids.add(hid)
            internal.append((
                Alert(id=hid, title=title, severity=severity, description=desc, source=source),
                False,
            ))

        return Scenario(
            name="full_cascade_failure",
            kind="full_cascade_failure",
            # 6-link chain needs 12 optimal steps (investigate+resolve). This leaves ~1 mistake of slack.
            max_steps=13,
            initial_health=0.45,
            initial_alerts_internal=internal,
            root_cause_alert_id=chain_ids[0] if chain_ids else None,
            cascade_chain_alert_ids=tuple(chain_ids),
        )

    @staticmethod
    def _alert_storm() -> Scenario:
        """Expert task: longer chain + more/higher-quality distractors."""
        chain_svcs = random.sample(ALL_SERVICES, k=7)
        real_ids: list[str] = []
        real_alerts: list[Tuple[Alert, bool]] = []
        used_ids: set[str] = set()

        templates = [
            "{svc} is failing. Logs show repeated connection refused errors pointing to {next}.",
            "{svc} is failing. Traces suggest upstream dependency on {next} is degraded.",
            "{svc} is unhealthy. Dependency checks are inconclusive; investigate service interactions.",
        ]

        for i, svc in enumerate(chain_svcs):
            rid = f"real-{random.randint(100, 999)}"
            while rid in used_ids:
                rid = f"real-{random.randint(100, 999)}"
            used_ids.add(rid)
            real_ids.append(rid)

            next_svc = chain_svcs[i + 1] if i + 1 < len(chain_svcs) else None
            if next_svc:
                desc = random.choice(templates).format(svc=svc, next=next_svc)
            else:
                desc = f"{svc} is failing. No upstream dependency identified - this appears to be the origin."

            real_alerts.append((
                Alert(
                    id=rid,
                    title=f"{svc} failing",
                    severity="critical" if i == 0 else "high",
                    description=desc,
                    source=svc,
                ),
                i == 0,
            ))

        # Mix low noise with tempting high-severity red herrings.
        noise_sample = random.sample(NOISE_POOL, 3)
        noise_alerts: list[Tuple[Alert, bool]] = []
        for i, (title, desc, severity, source) in enumerate(noise_sample):
            nid = f"noise-{i + 1:03d}"
            noise_alerts.append((
                Alert(id=nid, title=title, severity=severity, description=desc, source=source),
                False,
            ))

        # Add two high-severity red herrings + one critical decoy.
        herrings = [h for h in RED_HERRING_POOL if h[2] == "high"]
        for i, (title, desc, severity, source) in enumerate(random.sample(herrings, k=2), start=1):
            nid = f"decoy-hi-{i:03d}"
            noise_alerts.append((
                Alert(id=nid, title=title, severity=severity, description=desc, source=source),
                False,
            ))
        criticals = [h for h in RED_HERRING_POOL if h[2] == "critical"]
        if criticals:
            title, desc, severity, source = random.choice(criticals)
            noise_alerts.append((
                Alert(id="decoy-crit-001", title=title, severity=severity, description=desc, source=source),
                False,
            ))

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
