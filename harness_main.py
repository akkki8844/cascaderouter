"""Competition harness entrypoint — Track 1, AMD Developer Hackathon: ACT II.

This is what the submitted container runs. Contract (Participant Guide):

  1. Read tasks from /input/tasks.json on startup:
         [ {"task_id": "t1", "prompt": "..."}, ... ]
  2. Answer each task, routing through the Tiered Calibrated Cascade.
  3. Write /output/results.json before exiting:
         [ {"task_id": "t1", "answer": "..."}, ... ]
  4. Exit 0 on success, non-zero on failure.

Runtime environment injected by the harness (never hardcoded):
  FIREWORKS_API_KEY   — the key ALL remote calls must use
  FIREWORKS_BASE_URL  — the URL ALL remote calls must go through
                        (calls bypassing it are not recorded = zero tokens)
  ALLOWED_MODELS      — comma-separated permitted model ids (launch day)

Budgets: container ready <60s, <30s per request, <10 min total runtime,
grading environment 2 vCPU / 4 GB RAM (no Ollama pre-installed — ours is
bundled in the image). Local models score as ZERO tokens; only Fireworks
traffic counts, and the accuracy gate is 80% on a fixed 19-task set. For local testing you can override the I/O paths:
  TASKS_INPUT=eval/sample_input.json RESULTS_OUTPUT=out.json python harness_main.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from router_agent.config import apply_env_overrides, load_config
from router_agent.decision_log import DecisionLogger
from router_agent.router import RoutingAgent
from router_agent.task import task_from_raw

INPUT_PATH = Path(os.environ.get("TASKS_INPUT", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("RESULTS_OUTPUT", "/output/results.json"))
# The 10-minute cap is on the whole run; parallel tasks keep total wall time
# down while each individual request stays under the 30s ceiling. Default 2:
# the published grading environment is 2 vCPU / 4 GB RAM (organizer
# clarification, 2026-07-08) — more workers than cores just deepens the
# CPU-bound Ollama queue and risks per-request timeouts while queued.
MAX_WORKERS = int(os.environ.get("AGENT_MAX_WORKERS", "2"))


def build_agent() -> RoutingAgent:
    cfg = apply_env_overrides(load_config(os.environ.get("AGENT_CONFIG")))
    # Harness posture — differs from local dev in four compliance-driven ways:
    # 1. No cross-run cache: in-memory only, exact-match only, so an unseen
    #    prompt *variant* can never be served a stale answer (rule: "do not
    #    hardcode or cache answers").
    cfg.cache.persist = False
    cfg.cache.similarity_threshold = 1.0
    # 2. No 8B critic in the image (size/CPU budget) — disagreements go
    #    straight to remote verify instead.
    cfg.router.critique_enabled = False
    # 3. Per-request timeout under the competition's 30s ceiling.
    cfg.local.request_timeout = 25.0
    cfg.remote.request_timeout = 25.0
    # 3b. Hard remote-token ceiling: rank is ascending Fireworks tokens, so
    #     the run's billed total must be bounded even on a hostile task set.
    #     Reserve-based, thread-safe; a refused call falls back to the free
    #     local answer (see router._RemoteBudget).
    cfg.router.remote_token_budget = int(
        os.environ.get("REMOTE_TOKEN_BUDGET", "480"))
    # 4. The local tier is the Ollama bundled in this container; config.yaml
    #    already points at localhost:11434. LOCAL_BASE_URL overrides it for
    #    local testing against a differently-hosted server.
    local_url = os.environ.get("LOCAL_BASE_URL", "").strip()
    if local_url:
        cfg.local.backend = "openai_compatible"
        cfg.local.base_url = local_url
    return RoutingAgent(cfg)


def main() -> int:
    start = time.time()
    try:
        raw_tasks = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"FATAL: no task file at {INPUT_PATH} — mount the harness "
              "input volume (or set TASKS_INPUT for local testing).",
              file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"FATAL: {INPUT_PATH} is not valid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(raw_tasks, list):
        print(f"FATAL: {INPUT_PATH} must be a JSON array of tasks.",
              file=sys.stderr)
        return 1

    agent = build_agent()
    logger = DecisionLogger(os.environ.get("DECISIONS_LOG",
                                           "logs/decisions.jsonl"))

    def solve(raw: dict) -> dict:
        """One task -> one result row. Never raises: a single bad task must
        not cost the whole submission (malformed output scores zero)."""
        task_id = str(raw.get("task_id", raw.get("id", "")))
        try:
            task = task_from_raw(raw)
            decision = agent.answer(task)
            try:
                logger.log(decision)
            except Exception:
                pass  # logging must never break the run
            return {"task_id": task.id, "answer": str(decision.answer)}
        except Exception as exc:
            print(f"  task {task_id!r} failed: {exc}", file=sys.stderr)
            return {"task_id": task_id, "answer": ""}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(solve, raw_tasks))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(OUTPUT_PATH)  # atomic: never leave a half-written results file

    elapsed = time.time() - start
    print(f"done: {len(results)} tasks in {elapsed:.1f}s -> {OUTPUT_PATH}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
