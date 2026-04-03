---
title: Incident Response OpenEnv
emoji: 🚨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# Incident Response OpenEnv

Realistic **site reliability / incident triage** environment for [OpenEnv](https://github.com/meta-pytorch/OpenEnv): agents read firing alerts, choose a remediation **`action_type`**, target the correct **`alert_id`**, and receive graded rewards with partial credit. **Four** benchmark tasks (**easy → medium → hard → expert**) cover single-alert triage, root-cause identification among symptoms, ordered cascade resolution, and an **alert storm** with noise vs signal.

## Why this submission stands out

- **Real-world domain** - not a toy grid or guessing game; models must reason about dependencies and severities.
- **Full OpenEnv surface** - typed `Action` / `Observation` / `State`, `reset` / `step` / `state`, `openenv.yaml`, HTTP API, Docker.
- **Meaningful rewards** - sparse success on wrong targets, partial signals for "right direction," chain order on hard tasks, ordered resolution under noise on expert.
- **Reproducible baseline** - root `inference.py` using the official OpenAI client, env vars `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, structured `[START]` / `[STEP]` / `[END]` logs only on stdout by default (no extra lines).

## Tasks

| Task | Difficulty | Focus |
|------|------------|--------|
| `task_easy` | easy | Single alert (e.g. disk / CPU / memory); pick the correct resolution action. |
| `task_medium` | medium | Several alerts; find and fix the **root cause** (e.g. DB), not only symptoms. |
| `task_hard` | hard | **Cascade chain**: investigate and resolve each upstream link in order until the chain clears. |
| `task_expert` | expert | **Alert storm**: many low-severity noise alerts plus a hidden dependency chain; ignore noise and resolve the real chain in order. |

Rewards are always in **\[0, 1]** per step; the baseline caps **episode score** at **1.0** (sum of step rewards, clamped).

## Action & observation

**Action** (`IncidentResponseAction`): `alert_id`, `action_type`, `notes`.

**Observation**: `alerts[]` (id, title, severity, description, source), `resolved_alerts`, `system_health`, `step_number`, `message` (grader feedback), plus top-level `reward` / `done` from the HTTP wrapper.

**State** (`GET /state`): `episode_id`, `step_count`, `task_id`, `max_steps`, `total_reward`, `scenario_name`.

## Quick start (local)

```bash
cd incident_response_env
uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Health:

```bash
curl -s http://127.0.0.1:8000/health
```

Reset & step (example):

```bash
curl -s -X POST http://127.0.0.1:8000/reset -H "Content-Type: application/json" -d "{\"task_id\":\"task_easy\"}"
curl -s -X POST http://127.0.0.1:8000/step -H "Content-Type: application/json" -d "{\"action\":{\"alert_id\":\"disk-alert-07\",\"action_type\":\"scale_up\",\"notes\":\"Relieve disk pressure\"}}"
```

## Baseline inference (`inference.py`)

Required for the hackathon harness: the official **`OpenAI` Python client** with an **OpenAI-compatible** HTTP API (`base_url` + API key). This repo follows the Round 1 sample: use **`HF_TOKEN`** as the client API key and **`API_BASE_URL`** / **`MODEL_NAME`** for the endpoint and model id. Some writeups mention **`OPENAI_API_KEY`**; that is equivalent only if you point the same client at an OpenAI endpoint and set that variable instead—**this project does not read `OPENAI_API_KEY` by default.**

```powershell
$env:ENV_URL       = "http://127.0.0.1:8000"
$env:API_BASE_URL  = "https://router.huggingface.co/v1"
$env:MODEL_NAME    = "<model id>"
$env:HF_TOKEN      = "<hf token>"
uv run python inference.py
```

### Baseline scores (reproducibility)

Episode **score** is the sum of per-step rewards, capped at **1.0**. **`success`** in `[END]` uses `SUCCESS_SCORE_THRESHOLD` (default **0.1**). Exact numbers depend on the **model**, **provider**, and **scenario randomness** unless you fix **`ENV_URL`**, credentials, and (when supported) a seed.

Example run (local env, **`temperature=0`** in `inference.py`, Hugging Face router, model id along the lines of **`llama-3.1-8b-instant`** / **`meta-llama/Llama-3.1-8B-Instruct`** per your **`MODEL_NAME`**):

| Task | Episode score (`[END] score=`) | `success=` |
|------|-------------------------------|------------|
| `task_easy` | 1.000 | `true` |
| `task_medium` | 1.000 | `true` |
| `task_hard` | 1.000 | `true` |
| `task_expert` | 0.070 | `false` |

Re-run after changing the model or env and paste your own row into submissions if organizers ask for measured baselines.

Optional:

- `INFERENCE_STUB=1` - run without an LLM (deterministic policy) for CI or smoke tests.
- `INFERENCE_SUMMARY=1` - print an extra `[SUMMARY]` line (omit for strict stdout parsers).

**Windows / `openenv push`:** if you see `charmap` codec errors, set UTF-8 mode before push: `set PYTHONUTF8=1` (cmd) or `$env:PYTHONUTF8="1"` (PowerShell).

Runtime: keep total wall clock **under 20 minutes**; use a small instruct model if needed.

## Docker

Root **`Dockerfile`** (Hugging Face default) and **`server/Dockerfile`** are kept in sync.

```bash
docker build -t incident-response-env .
# equivalent: docker build -t incident-response-env -f server/Dockerfile .
docker run --rm -p 8000:8000 incident-response-env
```

## Phase 8 - Deploy to Hugging Face Spaces

Create a **Docker** Space on Hugging Face named e.g. `incident-response-env` (must match your repo id if you use CLI defaults).

### Method A - OpenEnv CLI (recommended)

Current OpenEnv packages expose **`openenv push`**, not `openenv deploy` (if your course PDF says `deploy`, use **`push`** with the same repo id).

```bash
# one-time
huggingface-cli login   # paste HF token when prompted

cd incident_response_env
openenv validate
openenv push --repo-id YOUR_HF_USERNAME/incident-response-env
# Windows: try PYTHONUTF8=1 if push fails on encoding
# add --private if required; use --no-interface if you hit Gradio/UI issues
```

`openenv push` reads **`openenv.yaml`** and uploads the environment; the repo root **`Dockerfile`** is used by HF's Docker SDK builder.

### Method B - Manual Git push

```bash
cd incident_response_env
git init
git remote add origin https://huggingface.co/spaces/YOUR_HF_USERNAME/incident-response-env
git add .
git commit -m "OpenEnv incident-response submission"
git push -u origin main
```

Ensure the Space **SDK** is **Docker** on the Hugging Face UI (README front matter already sets `sdk: docker`, `app_port: 8000`).

### After deploy

Public app URL is usually:

`https://YOUR_HF_USERNAME-incident-response-env.hf.space`

Smoke test (no trailing slash issues - use exact host Hugging Face shows):

```bash
curl -sS https://YOUR_HF_USERNAME-incident-response-env.hf.space/health
curl -sS -X POST https://YOUR_HF_USERNAME-incident-response-env.hf.space/reset \
  -H "Content-Type: application/json" -d "{}"
```

Update **`docker_image`** in `openenv.yaml` to `YOUR_HF_USERNAME/incident-response-env` for documentation consistency.

Then run **`openenv validate --url https://...hf.space`** and your organizer's pre-submission script.

## Validate before submit

```bash
openenv validate
# optional: runtime check against a deployed URL
openenv validate --url https://<your-space>.hf.space
```

## Pre-submission checklist (Round 1)

Cross-check with the official dashboard (e.g. Scaler / Meta OpenEnv Round 1):

- [ ] **`inference.py`** at repo root; uses **`OpenAI`** client + **`API_BASE_URL`**, **`MODEL_NAME`**, **`HF_TOKEN`**
- [ ] Stdout: **`[START]`**, **`[STEP]`**, **`[END]`** only (avoid `INFERENCE_SUMMARY` for automated parsing)
- [ ] **`openenv validate`** OK; **`uv.lock`** present if required
- [ ] **Dockerfile** builds in CI
- [ ] **HF Space** up; health + reset respond
- [ ] **≥ 3 tasks** (this repo has **4**) with graders; rewards in **[0, 1]**
- [ ] **README** describes domain, action/observation spaces, tasks, setup, and **example baseline scores** (this file)
- [ ] No secrets in git; rotate any leaked tokens

## Project layout

```
incident_response_env/
|-- Dockerfile             # HF Spaces default path (same image as server/Dockerfile)
|-- .dockerignore
|-- inference.py          # Hackathon baseline (LLM + env HTTP)
|-- openenv.yaml
|-- models.py
|-- client.py             # WebSocket EnvClient wrapper
|-- pyproject.toml
|-- uv.lock
`-- server/
    |-- app.py            # FastAPI app
    |-- environment.py    # reset / step / state
    |-- scenarios.py
    |-- graders.py
    |-- Dockerfile
    `-- requirements.txt
```

## Client (WebSocket)

```python
from incident_response_env import IncidentResponseEnv, IncidentResponseAction

with IncidentResponseEnv(base_url="http://localhost:8000") as env:
    r = env.reset(task_id="task_easy")  # pass kwargs your server accepts
    r = env.step(
        IncidentResponseAction(
            alert_id="disk-alert-07",
            action_type="scale_up",
            notes="Expand storage.",
        )
    )
```

(See `client.py` for `_step_payload` / parsing details.)

## License

See `LICENSE` in the repository.
