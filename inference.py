import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# Sample contract: defaults only for API_BASE_URL and MODEL_NAME; HF_TOKEN has no default.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
# Benchmark name for [START] line (hackathon sample uses env=<benchmark>)
BENCHMARK = os.environ.get("INCIDENT_BENCHMARK", "incident_response_env")
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.1"))
# Stricter bar for reporting "all tasks strong" (e.g. leaderboard psychologics).
STRICT_TASK_SCORE = float(os.environ.get("STRICT_TASK_SCORE", "0.95"))

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) 
doing live incident triage.

You will receive a JSON object with:
- task_id: the current task
- step: current step number
- alerts: list of active alerts, each with id, title, severity, description, source
- resolved_alert_ids: alerts already resolved this episode
- environment_message: feedback from your last action

Your job: decide which alert to address and what action to take.

Rules:
- alert_id MUST be copied exactly from the id field of one active alert
- Read descriptions carefully — they contain dependency and causality clues
- Address root causes before downstream symptoms
- Low/info severity alerts are usually noise unless description clearly shows causality
- Always mention the service name in your notes

Available action_type values:
scale_up, fix, restart, rollback, mitigate, remediate, isolate, block, investigate

Respond ONLY as JSON with no extra text:
{
  "alert_id": "exact id from active alerts list",
  "action_type": "chosen action",
  "notes": "one sentence mentioning the service name and your reasoning"
}"""


def _build_llm_user_payload(*, task_id: str, step: int, obs: Dict[str, Any]) -> str:
    alerts = obs.get("alerts") or []
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "step": step,
        "alerts": alerts,
        "resolved_alert_ids": obs.get("resolved_alerts") or [],
        "environment_message": obs.get("message") or "",
    }
    return json.dumps(payload, ensure_ascii=False)

_TASK_MAX_STEPS = {"task_easy": 5, "task_medium": 10, "task_hard": 20, "task_expert": 15}


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _stub_action(
    task_id: str, obs: Dict[str, Any], episode_alert_ids: List[str]
) -> Dict[str, Any]:
    """Deterministic policy for local runs without an LLM (INFERENCE_STUB=1)."""
    alerts = obs.get("alerts") or []
    resolved = set(
        str(x) for x in (obs.get("resolved_alerts") or []) if x is not None
    )
    active_ids = [
        str(a["id"])
        for a in alerts
        if isinstance(a, dict) and a.get("id")
    ]
    workable = [i for i in episode_alert_ids if i not in resolved]

    if task_id == "task_easy":
        aid = active_ids[0] if active_ids else ""
        kind = aid.split("-")[0] if aid else "disk"
        action_type = (
            "restart"
            if kind in ("cpu", "memory")
            else "scale_up"
            if kind == "disk"
            else "mitigate"
        )
        return {"alert_id": aid, "action_type": action_type, "notes": "stub policy"}

    if task_id == "task_medium":
        aid = active_ids[0] if active_ids else ""
        if not aid:
            pick = _pick_fallback_alert_id(alerts, workable)
            aid = pick or ""
        return {"alert_id": aid, "action_type": "scale_up", "notes": "stub policy"}

    if task_id == "task_expert":
        real_ids = [i for i in episode_alert_ids if not i.startswith("noise")]
        unresolved_real = [i for i in real_ids if i not in resolved]
        aid = unresolved_real[0] if unresolved_real else (active_ids[0] if active_ids else "")
        return {"alert_id": aid, "action_type": "fix", "notes": "stub: addressing real service failure"}

    aid = active_ids[0] if active_ids else ""
    if not aid:
        pick = _pick_fallback_alert_id(alerts, workable)
        aid = pick or ""
    return {"alert_id": aid, "action_type": "fix", "notes": "stub policy"}


def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    *,
    step: int,
    action_str: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP]  step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    *,
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _normalize_step_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "observation" in payload and isinstance(payload["observation"], dict):
        obs = dict(payload["observation"])
        if "done" in payload:
            obs["done"] = payload["done"]
        if "reward" in payload:
            obs["reward"] = payload["reward"]
        return obs
    return payload


def _parse_action(text: str) -> Tuple[Dict[str, Any], str]:
    raw = text.strip()
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except Exception:
                data = {}
        else:
            data = {}

    alert_id = str(data.get("alert_id", "")).strip()
    action_type = str(data.get("action_type", "")).strip() or "investigate"
    raw_notes = data.get("notes", data.get("reasoning", ""))
    notes = str(raw_notes).strip()

    action = {"alert_id": alert_id, "action_type": action_type, "notes": notes}
    return action, notes


def _action_str(action: Dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def _alert_ids_from_obs(alerts: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(alerts, list):
        return out
    for a in alerts:
        if isinstance(a, dict) and a.get("id"):
            out.append(str(a["id"]))
    return out


def _pick_fallback_alert_id(
    alerts: Any, unresolved_ordered: List[str]
) -> Optional[str]:
    """
    Pick a valid unresolved id. Only considers rows in `alerts` whose id is in
    unresolved_ordered - never return an id that leaked into `alerts` from elsewhere.
    Prefer critical among those rows, else first matching row, else first in episode order.
    """
    if not unresolved_ordered:
        return None
    allowed = set(unresolved_ordered)
    if isinstance(alerts, list) and alerts:
        for a in alerts:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id", "")).strip()
            if aid not in allowed:
                continue
            if a.get("severity") == "critical":
                return aid
        for a in alerts:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id", "")).strip()
            if aid in allowed:
                return aid
    return unresolved_ordered[0]


def _sanitize_action(
    action: Dict[str, Any],
    obs: Dict[str, Any],
    episode_alert_ids: List[str],
) -> Dict[str, Any]:
    """
    If the model hallucinates an alert_id from a prior task, repair.

    Only ids that appeared in the initial reset for THIS episode are valid - never trust
    the model to invent ids. Also avoid targeting an id already in resolved_alerts.
    """
    if not episode_alert_ids:
        return action

    alerts = obs.get("alerts", [])
    resolved = set(str(x) for x in (obs.get("resolved_alerts") or []) if x is not None)
    aid = str(action.get("alert_id", "")).strip()

    epi_set = set(episode_alert_ids)
    workable = [i for i in episode_alert_ids if i not in resolved]

    ok = aid in epi_set and aid not in resolved
    if ok:
        return action

    out = dict(action)
    # Preserve reset order (important for cascade chain: earlier IDs before later ones, ...)
    unresolved_for_pick = workable
    chosen = _pick_fallback_alert_id(alerts, unresolved_for_pick)
    if chosen is None:
        return out

    bad = aid or "(empty)"
    note = str(out.get("notes", "")).strip()
    reason = (
        "not part of this episode's alerts"
        if aid not in epi_set
        else "already resolved"
    )
    repair = f"Invalid alert_id {bad!r} ({reason}); using {chosen}."
    out["alert_id"] = chosen
    out["notes"] = f"{repair} {note}".strip()
    return out


def run_episode(
    *,
    task_id: str,
    client: Optional[OpenAI],
    model_name: str,
    use_stub: bool,
) -> float:
    """Run one benchmark task; return episode score in [0, 1] ( capped sum of step rewards)."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    try:
        reset_payload = requests.post(
            f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=15
        ).json()
        obs = _normalize_step_payload(reset_payload)
        episode_alert_ids = _alert_ids_from_obs(obs.get("alerts", []))

        max_loops = int(
            obs.get("max_steps") or _TASK_MAX_STEPS.get(task_id, 20) or 20
        )
        t0 = time.time()
        step = 0

        while not bool(obs.get("done", False)):
            if time.time() - t0 > 60 * 15:
                break
            if step >= max_loops:
                break

            step += 1
            err: Optional[str] = None
            reward = 0.0
            done = False
            action_line = "{}"

            action: Dict[str, Any] = {}
            try:
                alerts = obs.get("alerts", [])

                if use_stub:
                    action = _stub_action(task_id, obs, episode_alert_ids)
                else:
                    assert client is not None
                    user_content = _build_llm_user_payload(
                        task_id=task_id, step=step, obs=obs
                    )
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=0.0,
                    )
                    action, _notes = _parse_action(
                        response.choices[0].message.content or ""
                    )
                    if not action.get("alert_id"):
                        if isinstance(alerts, list) and alerts:
                            first = alerts[0]
                            if isinstance(first, dict) and "id" in first:
                                action["alert_id"] = first["id"]

                action = _sanitize_action(action, obs, episode_alert_ids)

                action_line = _action_str(action)

                step_payload = requests.post(
                    f"{ENV_URL}/step", json={"action": action}, timeout=15
                ).json()
                obs = _normalize_step_payload(step_payload)

                reward = float(obs.get("reward") or 0.0)
                done = bool(obs.get("done", False))
                rewards.append(reward)
                steps_taken = step
            except Exception as exc:
                err = str(exc).replace("\n", " ")
                rewards.append(0.0)
                steps_taken = step
                # Avoid empty action in logs when the LLM or HTTP provider fails (e.g. 402).
                if not action:
                    fb_id = _pick_fallback_alert_id(
                        obs.get("alerts", []),
                        [
                            i
                            for i in episode_alert_ids
                            if i
                            not in set(
                                str(x)
                                for x in (obs.get("resolved_alerts") or [])
                                if x is not None
                            )
                        ],
                    )
                    if fb_id:
                        action = {
                            "alert_id": fb_id,
                            "action_type": "investigate",
                            "notes": "LLM/API error; no step sent.",
                        }
                action_line = _action_str(action) if action else "{}"

            log_step(
                step=step,
                action_str=action_line,
                reward=reward,
                done=done,
                error=err,
            )

            if err is not None:
                break
            if done:
                break

        total = sum(rewards) if rewards else 0.0
        score = min(max(total, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    use_stub = _truthy_env("INFERENCE_STUB")

    if use_stub:
        model_name = os.environ.get("MODEL_NAME", "stub-local")
        client: Optional[OpenAI] = None
    else:
        if not HF_TOKEN or not str(HF_TOKEN).strip():
            raise SystemExit(
                "Set HF_TOKEN before running inference.py (API_BASE_URL and MODEL_NAME "
                "default to the Hugging Face router and a small instruct model), "
                "or set INFERENCE_STUB=1 to run without an LLM."
            )
        client = OpenAI(
            api_key=HF_TOKEN.strip(),
            base_url=API_BASE_URL,
        )
        model_name = MODEL_NAME

    tasks = ["task_easy", "task_medium", "task_hard", "task_expert"]
    episode_scores: List[Tuple[str, float]] = []
    for task in tasks:
        ep_score = run_episode(
            task_id=task,
            client=client,
            model_name=model_name,
            use_stub=use_stub,
        )
        episode_scores.append((task, ep_score))

    # Hackathon evaluators may parse stdout strictly ([START]/[STEP]/[END] only).
    # Set INFERENCE_SUMMARY=1 for an extra aggregate line (local leaderboards).
    if _truthy_env("INFERENCE_SUMMARY"):
        scores_only = [s for _, s in episode_scores]
        mean_score = sum(scores_only) / len(scores_only) if scores_only else 0.0
        min_score = min(scores_only) if scores_only else 0.0
        strict_ok = all(s >= STRICT_TASK_SCORE for s in scores_only)
        parts = ",".join(f"{t}:{v:.3f}" for t, v in episode_scores)
        print(
            f"[SUMMARY] mean_score={mean_score:.3f} min_score={min_score:.3f} "
            f"strict_all_ge_{STRICT_TASK_SCORE:g}={str(strict_ok).lower()} "
            f"per_task={parts}",
            flush=True,
        )


if __name__ == "__main__":
    main()
