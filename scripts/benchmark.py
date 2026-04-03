import os
from typing import Any, Dict, List, Tuple

import requests


ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

_MAX_STEPS = {
    "task_easy": 5,
    "task_medium": 10,
    "task_hard": 20,
    "task_expert": 15,
}


def _extract_obs(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("observation"), dict):
        return payload["observation"]
    return payload


def _extract_reward(payload: Dict[str, Any], obs: Dict[str, Any]) -> float:
    if "reward" in payload and payload["reward"] is not None:
        return float(payload["reward"] or 0.0)
    return float(obs.get("reward") or 0.0)


def _extract_done(payload: Dict[str, Any], obs: Dict[str, Any]) -> bool:
    if "done" in payload and payload["done"] is not None:
        return bool(payload["done"])
    return bool(obs.get("done", False))


def run_episode(task_id: str) -> Tuple[float, int]:
    reset_resp = requests.post(
        f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=15
    )
    try:
        reset_payload = reset_resp.json()
    except Exception:
        print(
            f"  ERROR: non-JSON reset response for {task_id}: {reset_resp.text[:200]}"
        )
        return 0.0, 0
    obs = _extract_obs(reset_payload)

    max_steps = int(_MAX_STEPS.get(task_id, 20))
    steps_taken = 0
    rewards: List[float] = []
    done = bool(obs.get("done", False))

    while not done and steps_taken < max_steps:
        alerts = obs.get("alerts") or []
        alert_id = ""
        if isinstance(alerts, list) and alerts:
            first = alerts[0]
            if isinstance(first, dict):
                alert_id = str(first.get("id") or "")

        action_type = "fix"
        if task_id == "task_easy":
            kind = alert_id.split("-")[0] if alert_id else "disk"
            if kind in ("cpu", "memory"):
                action_type = "restart"
            elif kind == "network":
                action_type = "mitigate"
            else:
                action_type = "scale_up"

        step_payload = requests.post(
            f"{ENV_URL}/step",
            json={
                "action": {
                    "alert_id": alert_id,
                    "action_type": action_type,
                    "notes": "benchmark",
                }
            },
            timeout=15,
        )
        try:
            step_payload = step_payload.json()
        except Exception:
            # Skip this episode rather than crashing.
            print(
                f"  ERROR: non-JSON step response for {task_id}: {step_payload.text[:200]}"
            )
            return 0.0, 0

        obs = _extract_obs(step_payload)
        reward = _extract_reward(step_payload, obs)
        done = _extract_done(step_payload, obs)

        rewards.append(reward)
        steps_taken += 1

        # Continue until the environment reports done (or we hit max_steps).

    score = min(sum(rewards), 1.0) if rewards else 0.0
    return score, steps_taken


def run_benchmark(n: int = 10) -> None:
    tasks = ["task_easy", "task_medium", "task_hard", "task_expert"]
    print(f"=== Benchmark Results (n={n} stub episodes each) ===")

    overall_scores: List[float] = []

    for task_id in tasks:
        scores: List[float] = []
        steps_list: List[int] = []
        for _ in range(n):
            score, steps_taken = run_episode(task_id)
            scores.append(score)
            steps_list.append(steps_taken)
            overall_scores.append(score)

        mean_score = sum(scores) / n if n else 0.0
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        mean_steps = sum(steps_list) / n if n else 0.0

        print(
            f"{task_id}:   mean={mean_score:.3f}  min={min_score:.3f}  max={max_score:.3f}  steps={mean_steps:.1f}"
        )

    overall_mean = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    print(f"=== Overall mean: {overall_mean:.3f} ===")


if __name__ == "__main__":
    run_benchmark()

