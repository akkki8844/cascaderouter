"""CLI entrypoint for the Hybrid Token-Efficient Routing Agent.

Usage:
    python main.py answer --prompt "What is the capital of France?"
    python main.py batch --input tasks.jsonl --output answers.jsonl
    python main.py answer --prompt "..." --strategy always_remote

The batch input format is JSONL, one task per line:
    {"id": "t1", "prompt": "..."}
Adapt router_agent/task.py on Day 1 if the harness's real format differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from router_agent.config import load_config
from router_agent.decision_log import DecisionLogger
from router_agent.router import RoutingAgent
from router_agent.task import task_from_raw


def build_agent(args) -> tuple[RoutingAgent, DecisionLogger]:
    cfg = load_config(args.config)
    if args.strategy:
        cfg.router.strategy = args.strategy
    return RoutingAgent(cfg), DecisionLogger(cfg.logging.decisions_path)


def cmd_answer(args) -> None:
    agent, logger = build_agent(args)
    task = task_from_raw(args.prompt)
    decision = agent.answer(task)
    logger.log(decision)
    print(json.dumps({
        "id": task.id,
        "answer": decision.answer,
        "route": decision.route,
        "remote_tokens": decision.remote_tokens,
    }, ensure_ascii=False))


def cmd_batch(args) -> None:
    agent, logger = build_agent(args)
    in_path, out_path = Path(args.input), Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_remote = 0
    with out_path.open("w", encoding="utf-8") as out:
        for line in in_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            task = task_from_raw(json.loads(line))
            decision = agent.answer(task)
            logger.log(decision)
            total_remote += decision.remote_tokens
            out.write(json.dumps({"id": task.id, "answer": decision.answer},
                                 ensure_ascii=False) + "\n")
    print(f"done: {out_path}  (remote tokens spent: {total_remote})",
          file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(prog="routing-agent")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--strategy", default=None,
                        help="override strategy: always_remote | always_local "
                             "| heuristic | cascade")
    sub = parser.add_subparsers(dest="command", required=True)

    p_answer = sub.add_parser("answer", help="answer a single task")
    p_answer.add_argument("--prompt", required=True)
    p_answer.set_defaults(func=cmd_answer)

    p_batch = sub.add_parser("batch", help="answer a JSONL file of tasks")
    p_batch.add_argument("--input", required=True)
    p_batch.add_argument("--output", default="answers.jsonl")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
