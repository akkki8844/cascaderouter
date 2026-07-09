"""Task object + input adapter.

The harness's task format is unknown until kickoff. This is the ONLY module
that should need rewriting on Day 1 — everything downstream consumes the
normalized Task object, never the raw input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    prompt: str
    metadata: dict = field(default_factory=dict)


def task_from_raw(raw) -> Task:
    """Normalize whatever the harness sends into a Task.

    Currently handles:
      - plain string -> Task with generated id
      - dict with 'prompt' (and optional 'id', anything else goes to metadata)
      - JSON string of the above
    Extend/replace this on Day 1 once the real format is revealed.
    """
    if isinstance(raw, Task):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{"):
            try:
                return task_from_raw(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return Task(id=f"task-{abs(hash(raw)) % 10**8}", prompt=raw)
    if isinstance(raw, dict):
        prompt = raw.get("prompt") or raw.get("question") or raw.get("input")
        if prompt is None:
            raise ValueError(f"Cannot find a prompt field in task: {raw!r}")
        # 'task_id' is the official competition harness field (/input/tasks.json);
        # 'id' is our internal/eval format — accept both.
        task_id = str(raw.get("task_id") or raw.get("id")
                      or f"task-{abs(hash(prompt)) % 10**8}")
        metadata = {k: v for k, v in raw.items()
                    if k not in ("task_id", "id", "prompt", "question", "input")}
        return Task(id=task_id, prompt=str(prompt), metadata=metadata)
    raise TypeError(f"Unsupported task input type: {type(raw)}")
