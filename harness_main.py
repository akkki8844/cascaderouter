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
traffic counts. The 2026-07-13 resubmission was killed at the 10-minute
limit (local 1B inference is ~10x slower on 2 CPU cores than on the dev
GPU), so v10 paces the whole run against a hard wall-clock deadline:

  - tasks are answered in descending local-cost order, so if time runs
    short it is the cheap-remote tasks that lose their local attempt;
  - a TimeGovernor degrades the router (full guards -> no retry ladders ->
    remote-fast-path) as time-per-remaining-task shrinks;
  - /output/results.json is (re)written atomically after EVERY task, and a
    watchdog flushes it and exits 0 before the kill — a partial score
    always beats TIMEOUT.

For local testing you can override the I/O paths:
  TASKS_INPUT=eval/sample_input.json RESULTS_OUTPUT=out.json python harness_main.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from router_agent.config import apply_env_overrides, load_config
from router_agent.confidence import classify_category
from router_agent.decision_log import DecisionLogger
from router_agent.router import RoutingAgent, TimeGovernor
from router_agent.task import task_from_raw

INPUT_PATH = Path(os.environ.get("TASKS_INPUT", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("RESULTS_OUTPUT", "/output/results.json"))
# Wall-clock budget for the whole harness process. The container is killed at
# 10 minutes, and the scorer's clock may also count container-start overhead
# (image extraction, entrypoint startup ~30s) that we can't see — so leave a
# wide margin: 450s here caps the whole container well under 9 minutes. The
# projected full run is 330-400s on the 2-vCPU grading box, so nothing is
# normally lost; if the box is slower, the watchdog turns the overrun into a
# partial score instead of a TIMEOUT.
RUN_TIME_BUDGET = float(os.environ.get("RUN_TIME_BUDGET_S", "450"))
# Default 2: the run is remote-first, i.e. network-I/O-bound — two in-flight
# Fireworks requests cost the 2-vCPU box nothing and halve the worst-case
# wall clock. (Local decode, when the fallback ever runs, is still serial:
# OLLAMA_NUM_PARALLEL=1 in entrypoint.sh.) Env-overridable.
MAX_WORKERS = int(os.environ.get("AGENT_MAX_WORKERS", "2"))

# Task scheduling order. Two pressures decide it:
#  - BUDGET: factual goes first — its remote escalations are the one spend
#    that buys accuracy (the locals are measurably wrong exactly when they
#    disagree), so those calls must reserve budget before a big code/NER
#    escalation can starve them (observed live: one 300-token NER escalation
#    left the factual calls budget-denied).
#  - TIME: after that, descending local-attempt cost, so if the deadline
#    governor has to degrade the tail of the run to remote-fast-path, the
#    tasks that lose their local attempt are the ones whose remote answers
#    are cheapest (sentiment ~50-160 tokens vs code's ~300-700).
_SCHED_ORDER = {"factual": 0, "code_debugging": 1, "code_generation": 2,
                "math": 3, "logic": 3, "ner": 4, "summarization": 5,
                "sentiment": 6}


def _sched_rank(raw: dict) -> int:
    try:
        return _SCHED_ORDER.get(classify_category(str(raw.get("prompt", ""))), 4)
    except Exception:
        return 4


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
    # 3. Request timeouts. Remote stays under the 30s-per-request ceiling.
    #    Local is the bundled Ollama on 2 CPU cores, where a 300-token
    #    generation legitimately takes ~30-40s: a 25s timeout there just
    #    discards nearly-finished work and retries it, compounding the very
    #    slowness that killed the run — the TimeGovernor bounds total time
    #    instead, shrinking each call's timeout as the deadline nears.
    cfg.local.request_timeout = 40.0
    cfg.remote.request_timeout = 25.0
    # 3b. Hard remote-token ceiling: rank is ascending Fireworks tokens, so
    #     the run's billed total must be bounded even on a hostile task set.
    #     Reserve-based, thread-safe; a refused call falls back to the free
    #     local answer (see router._RemoteBudget). 20000 is a SAFETY ceiling,
    #     not an optimization target: the graded 5.3% run showed a tight cap
    #     (480) denying the remote calls that were the run's only working
    #     tier — the accuracy gate (80%) must never lose to the token rank.
    cfg.router.remote_token_budget = int(
        os.environ.get("REMOTE_TOKEN_BUDGET", "20000"))
    # 3c. Remote-first: the local tier is a last-resort fallback only. On the
    #     2 vCPU / 4 GB grading VM local 1B inference thrashes (both graded
    #     local-first runs died: TIMEOUT, then 5.3%), while remote calls run
    #     in seconds regardless of the box. REMOTE_FIRST=0 restores the
    #     guarded local-first cascade for local experimentation.
    cfg.router.remote_first = os.environ.get("REMOTE_FIRST", "1") != "0"
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

    deadline = start + RUN_TIME_BUDGET
    agent = build_agent()
    agent.governor = TimeGovernor(deadline, len(raw_tasks))
    logger = DecisionLogger(os.environ.get("DECISIONS_LOG",
                                           "logs/decisions.jsonl"))

    def tid_of(raw) -> str:
        if isinstance(raw, dict):
            return str(raw.get("task_id", raw.get("id", "")))
        return ""

    # answers indexed by ORIGINAL input position; pre-seeded empty so every
    # snapshot — including a watchdog flush mid-run — covers every task_id.
    answers: dict[int, str] = {}
    answers_lock = threading.Lock()
    snapshot_lock = threading.Lock()  # two workers writing the same .tmp
    # concurrently is a crash on Windows and a corruption race anywhere

    def snapshot() -> None:
        """Atomically (re)write results.json with everything answered so
        far. Called after every task and by the watchdog: the output file
        is always present and always valid JSON, so a killed run is still
        a scoreable run."""
        with answers_lock:
            rows = [{"task_id": tid_of(raw), "answer": answers.get(i, "")}
                    for i, raw in enumerate(raw_tasks)]
        with snapshot_lock:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = OUTPUT_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(OUTPUT_PATH)  # atomic: never half-written results

    def flush_and_exit() -> None:
        snapshot()
        print(f"WATCHDOG: wall-clock budget ({RUN_TIME_BUDGET:.0f}s) hit — "
              f"flushed {sum(1 for a in answers.values() if a)} answers and "
              "exiting before the container limit.", file=sys.stderr)
        sys.stderr.flush()
        os._exit(0)  # worker threads may be mid-request; the file is written

    watchdog = threading.Timer(max(deadline - time.time(), 1.0),
                               flush_and_exit)
    watchdog.daemon = True
    watchdog.start()

    def solve(item: tuple[int, dict]) -> None:
        """One task -> one stored answer. Never raises: a single bad task
        must not cost the whole submission (malformed output scores zero)."""
        index, raw = item
        try:
            task = task_from_raw(raw)
            decision = agent.answer(task)
            try:
                logger.log(decision)
            except Exception:
                pass  # logging must never break the run
            answer = str(decision.answer)
        except Exception as exc:
            print(f"  task {tid_of(raw)!r} failed: {exc}", file=sys.stderr)
            answer = ""
        with answers_lock:
            answers[index] = answer
        agent.governor.task_done()
        try:
            snapshot()  # a transient write hiccup must not kill the run —
        except Exception:   # the next task's snapshot (or the final one)
            pass            # rewrites the complete file anyway

    ordered = sorted(enumerate(raw_tasks), key=lambda kv: _sched_rank(kv[1]))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(solve, ordered))

    watchdog.cancel()
    snapshot()
    elapsed = time.time() - start
    print(f"done: {len(raw_tasks)} tasks in {elapsed:.1f}s -> {OUTPUT_PATH}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
