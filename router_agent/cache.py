"""Within-run dedup cache (03_ARCHITECTURE §2).

If the incoming batch contains literally repeated tasks, a hit returns the
prior answer at $0 and zero latency. If tasks are all unique it simply never
fires — cheap insurance, deliberately dependency-light (fuzzy string
similarity, no embedding model required).

Competition-compliance note: the rules forbid hardcoding or caching answers
across runs ("evaluation uses unseen prompt variants"). In harness mode the
cache therefore runs with persist=False (in-memory, this run only) and an
exact-match threshold of 1.0 so a prompt *variant* — same wording, different
numbers — can never be served a stale answer. The image ships with no cache
file (.dockerignore excludes cache/).
"""

from __future__ import annotations

import json
import threading
from difflib import SequenceMatcher
from pathlib import Path

from .confidence import normalize_answer


class TaskCache:
    def __init__(self, path: str, similarity_threshold: float = 0.95,
                 enabled: bool = True, persist: bool = True):
        self.path = Path(path)
        self.similarity_threshold = similarity_threshold
        self.enabled = enabled
        self.persist = persist
        self._lock = threading.Lock()
        self._entries: list[dict] = []
        if enabled and persist and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        self._entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def lookup(self, prompt: str) -> str | None:
        if not self.enabled:
            return None
        key = normalize_answer(prompt)
        if not key:
            return None
        best_sim, best_answer = 0.0, None
        with self._lock:
            entries = list(self._entries)
        for entry in entries:
            if self.similarity_threshold >= 1.0:
                # exact-match mode (harness): no fuzzy matching at all
                if key == entry["key"]:
                    return entry["answer"]
                continue
            sim = SequenceMatcher(None, key, entry["key"]).ratio()
            if sim > best_sim:
                best_sim, best_answer = sim, entry["answer"]
        if self.similarity_threshold < 1.0 and best_sim >= self.similarity_threshold:
            return best_answer
        return None

    def store(self, prompt: str, answer: str) -> None:
        if not self.enabled:
            return
        entry = {"key": normalize_answer(prompt), "answer": answer}
        with self._lock:
            self._entries.append(entry)
            if not self.persist:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
