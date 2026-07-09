"""Local eval harness — our leaderboard proxy.

Replays a labeled task set through the routing agent and reports the two
numbers the competition scores (remote token count, accuracy) plus the
free-resolution rate (the strategy's real health metric, 11_WINNING_STRATEGY).

Run from the repo root:
    python -m eval.run_eval --tasks eval/tasks_placeholder.jsonl
    python -m eval.run_eval --strategy always_remote   # baseline comparison

Results land in eval_results/ as both JSON (machine) and Markdown (humans /
pitch deck). Task t08 duplicates t01 on purpose to exercise the cache path.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from router_agent.config import load_config
from router_agent.confidence import normalize_answer
from router_agent.decision_log import DecisionLogger
from router_agent.router import RoutingAgent
from router_agent.task import task_from_raw

FREE_ROUTES = {"cache", "local_agreement", "local_critique", "local_only"}


def _as_number(text: str):
    try:
        return float(text.replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def _spelled_to_num(text: str):
    """Convert spelled-out numbers to digits (zero -> 0, one -> 1, etc)."""
    spelled = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    t = text.strip().lower()
    return spelled.get(t)


def is_correct(answer: str, expected) -> bool:
    """Grade against one expected value or a list of acceptable aliases.

    Matches on: normalized equality, whole-word containment (so expected '8'
    does NOT match answer '48'), or numeric equivalence ('18.0' == '$18').
    Spelled numbers ('three') match digits ('3').
    """
    candidates = expected if isinstance(expected, list) else [expected]
    na = normalize_answer(answer)
    n_ans = _as_number(na)
    # Try spelled-to-numeric conversion
    if n_ans is None:
        s = _spelled_to_num(na)
        if s is not None:
            n_ans = float(s)

    for exp in candidates:
        ne = normalize_answer(str(exp))
        if not ne:
            continue
        if na == ne:
            return True
        # Whole-word containment (avoid '8' matching '48')
        if re.search(rf"(?<![0-9.]){re.escape(ne)}(?![0-9.])", na):
            return True
        # Numeric equivalence
        n_exp = _as_number(ne)
        if n_exp is None:
            s = _spelled_to_num(ne)
            if s is not None:
                n_exp = float(s)
        if n_ans is not None and n_exp is not None and n_ans == n_exp:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local eval harness.")
    parser.add_argument("--tasks", default="eval/tasks_placeholder.jsonl")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--strategy", default=None,
                        help="override config strategy for this run "
                             "(always_remote|always_local|heuristic|cascade)")
    parser.add_argument("--outdir", default="eval_results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.strategy:
        cfg.router.strategy = args.strategy
    # eval runs must not poison or read the production cache between strategies
    cfg.cache.path = str(Path(args.outdir) / f"cache_{cfg.router.strategy}.jsonl")
    cache_file = Path(cfg.cache.path)
    if cache_file.exists():
        cache_file.unlink()

    agent = RoutingAgent(cfg)
    logger = DecisionLogger(cfg.logging.decisions_path)

    rows = []
    for line in Path(args.tasks).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        expected = raw.pop("expected", "")
        task = task_from_raw(raw)
        decision = agent.answer(task)
        correct = is_correct(decision.answer, expected)
        decision.features["correct"] = correct
        logger.log(decision)
        rows.append({
            "id": task.id, "route": decision.route, "correct": correct,
            "remote_tokens": decision.remote_tokens,
            "local_tokens": decision.local_tokens,
            "latency_s": decision.latency_s,
            "answer": decision.answer, "expected": expected,
        })
        print(f"  {task.id}: route={decision.route:16s} "
              f"remote_tokens={decision.remote_tokens:5d} correct={correct}")

    n = len(rows)
    accuracy = sum(r["correct"] for r in rows) / n if n else 0.0
    remote_tokens = sum(r["remote_tokens"] for r in rows)
    free = sum(1 for r in rows if r["route"] in FREE_ROUTES)
    report = {
        "strategy": cfg.router.strategy,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": n,
        "accuracy": round(accuracy, 4),
        "remote_tokens_total": remote_tokens,
        "free_resolution_rate": round(free / n, 4) if n else 0.0,
        "route_distribution": _distribution(rows),
        "rows": rows,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = outdir / f"eval_{cfg.router.strategy}_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    md_path = outdir / f"eval_{cfg.router.strategy}_{stamp}.md"
    md_path.write_text(_markdown(report), encoding="utf-8")

    print(f"\nstrategy={cfg.router.strategy}  tasks={n}  "
          f"accuracy={accuracy:.1%}  remote_tokens={remote_tokens}  "
          f"free_resolution={free}/{n}")
    print(f"report: {json_path}")


def _distribution(rows: list[dict]) -> dict:
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["route"]] = dist.get(r["route"], 0) + 1
    return dist


def _markdown(report: dict) -> str:
    lines = [
        f"# Eval report — strategy `{report['strategy']}`",
        "",
        f"- **Tasks:** {report['tasks']}",
        f"- **Accuracy:** {report['accuracy']:.1%}",
        f"- **Remote tokens (score cost):** {report['remote_tokens_total']}",
        f"- **Free-resolution rate:** {report['free_resolution_rate']:.1%}",
        f"- **Run at:** {report['timestamp']}",
        "",
        "| Route | Count |",
        "|---|---|",
    ]
    for route, count in sorted(report["route_distribution"].items()):
        lines.append(f"| {route} | {count} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
