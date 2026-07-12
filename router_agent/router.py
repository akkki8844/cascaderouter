"""The Tiered Calibrated Cascade router (03_ARCHITECTURE).

Route order (strategy: cascade, v4 accuracy-first revision):
  1. within-run dedup cache      -> hit: return, 0 tokens
  2. category classifier         -> pick the remote plan per category:
     - math / logic              -> cross-family self-consistency vote
     - code gen / debug          -> strongest code model, single call
     - sentiment / summarization -> Gemma-first, single call
     - factual / NER             -> strongest general model, single call
  3. local models                -> emergency fallback ONLY (all remote
                                    tiers failed or returned blank)

Why no free local answering tier anymore: the real grading run (2026-07-11)
scored 57.9% (gate: 80%) because two 1B-class local models confidently
agreed on wrong answers on nearly half the tasks. Tokens saved below the
accuracy gate score zero — accuracy buys placement, tokens only break ties
among gate-passers.

Also implements the fallback strategies from 02_MVP_PLAN so every stage of
the build is submittable: always_remote (v0), always_local (v1),
heuristic (v2).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .cache import TaskCache
from .config import Config
from .confidence import (
    answers_agree,
    classify_category,
    heuristic_difficulty,
    is_computational,
    majority_vote,
    normalize_answer,
)
from .models import (
    ANSWER_SYSTEM_PROMPT,
    CODEDEBUG_SYSTEM_PROMPT,
    CODEGEN_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    NER_SYSTEM_PROMPT,
    REASONING_ANSWER_SYSTEM_PROMPT,
    SENTIMENT_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    MockClient,
    OpenAICompatibleClient,
    extract_json_field,
)
from .task import Task

# Category -> (system prompt, generation budget) for the categories that
# skip the short-answer cascade and go straight to one remote call.
# Code/summaries are long-form (string agreement between two local models is
# meaningless there). NER is here for a different reason: it is graded on
# COMPLETENESS, and validated testing showed a 1B local dropping an entity
# ("Tim Cook") while the critic confidently blessed the incomplete answer —
# completeness is exactly what a critic can't check, so local trust signals
# don't apply.
_LONGFORM = {
    "code_generation": (CODEGEN_SYSTEM_PROMPT, 2000),
    "code_debugging": (CODEDEBUG_SYSTEM_PROMPT, 2000),
    "summarization": (SUMMARY_SYSTEM_PROMPT, 800),
    "ner": (NER_SYSTEM_PROMPT, 800),
}

# Category -> system prompt for the short-answer local tier. Anything not
# listed uses the generic ANSWER prompt.
_LOCAL_PROMPTS = {
    "sentiment": SENTIMENT_SYSTEM_PROMPT,
}


@dataclass
class Decision:
    task_id: str
    prompt: str
    route: str                 # cache | local_agreement | local_critique | remote_tier1 | remote_tier2 | local_only | remote_only
    answer: str
    remote_tokens: int         # the ONLY number that costs us on the leaderboard
    local_tokens: int          # logged for interest; scores as zero
    latency_s: float
    features: dict = field(default_factory=dict)  # calibration inputs

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class RoutingAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.local.backend == "mock":
            self._local_a = MockClient(persona="model-a")
            self._local_b = MockClient(persona="model-b")
            self._critic = MockClient(persona="critic")
        else:
            client = OpenAICompatibleClient(cfg.local.base_url,
                                            cfg.local.api_key,
                                            cfg.local.request_timeout)
            self._local_a = self._local_b = self._critic = client
        if cfg.remote.backend == "mock":
            self._remote = MockClient(persona="remote")
        else:
            self._remote = OpenAICompatibleClient(cfg.remote.base_url,
                                                  cfg.remote.api_key,
                                                  cfg.remote.request_timeout)
        self.cache = TaskCache(cfg.cache.path,
                               cfg.cache.similarity_threshold,
                               cfg.cache.enabled,
                               cfg.cache.persist)

    # ---------- public API ----------

    def answer(self, task: Task) -> Decision:
        start = time.time()
        strategy = self.cfg.router.strategy
        if strategy == "always_remote":
            decision = self._always_remote(task)
        elif strategy == "always_local":
            decision = self._always_local(task)
        elif strategy == "heuristic":
            decision = self._heuristic(task)
        else:
            decision = self._cascade(task)
        decision.latency_s = round(time.time() - start, 3)
        return decision

    # ---------- strategies ----------

    def _always_remote(self, task: Task) -> Decision:
        category = classify_category(task.prompt)
        features = {"category": category}
        if category in _LONGFORM:
            system, budget = _LONGFORM[category]
        else:
            system, budget = ANSWER_SYSTEM_PROMPT, None
        return self._remote_single(task, system, "remote_only", features,
                                   max_tokens=budget)

    def _always_local(self, task: Task) -> Decision:
        resp = self._local_chat(self._local_a, self.cfg.local.model_a,
                                ANSWER_SYSTEM_PROMPT, task.prompt)
        answer = extract_json_field(resp.text, "answer") or resp.text.strip()
        return Decision(task.id, task.prompt, "local_only", str(answer),
                        remote_tokens=0, local_tokens=resp.total_tokens,
                        latency_s=0.0)

    def _heuristic(self, task: Task) -> Decision:
        difficulty = heuristic_difficulty(
            task.prompt,
            self.cfg.router.heuristic_max_prompt_chars,
            self.cfg.router.heuristic_hard_markers)
        if difficulty >= self.cfg.router.escalation_threshold:
            decision = self._always_remote(task)
        else:
            decision = self._always_local(task)
        decision.features["heuristic_difficulty"] = difficulty
        return decision

    def _cascade(self, task: Task) -> Decision:
        """v4 accuracy-first cascade.

        The real 19-task grading run (scored 2026-07-11) came back 57.9% —
        below the 80% accuracy gate, which zeroes the submission regardless
        of token count. Root cause: the free dual-local-agreement tier (two
        1–1.5B models) answered ~46% of tasks and was only reliable on the
        toy-easy proxy set, and self-consistency voted with three Gemma
        variants (correlated errors). Below the gate, saved tokens are worth
        nothing, so v4 routes EVERY graded category to a strong remote model
        (category-matched, cross-family voting for math/logic) and demotes
        the local models to an emergency fallback when all remote tiers fail.
        """
        features: dict = {}

        # 1. cache (within-run dedup only in harness mode — see cache.py)
        cached = self.cache.lookup(task.prompt)
        if cached is not None:
            return Decision(task.id, task.prompt, "cache", cached,
                            remote_tokens=0, local_tokens=0, latency_s=0.0,
                            features={"cache_hit": True})

        category = classify_category(task.prompt)
        features["category"] = category
        if category in ("math", "logic"):
            return self._remote_self_consistency(task)
        if category in _LONGFORM:
            return self._remote_longform(task, category, features)

        # factual / sentiment: one short remote call to the strongest general
        # model. Completion is a few tokens; the accuracy delta over a 1B
        # local model is the whole ballgame.
        system = _LOCAL_PROMPTS.get(category, ANSWER_SYSTEM_PROMPT)
        return self._remote_single(task, system, f"remote_{category}",
                                   features, max_tokens=800,
                                   category=category)

    # ---------- model-role selection ----------

    def _tier_order(self, category: str | None) -> list[str]:
        """Order the allowed models best-first for a category.

        kimi-k2p7-code is the strongest general instruct model on the allowed
        list (and the obvious first choice for code); minimax-m3 is the
        reasoning model; the Gemma checkpoints lead the categories a 31B
        instruct model is near-certain on (sentiment, summarization) so the
        submission keeps genuine Gemma usage for the bonus prize without
        betting the accuracy gate on it.
        """
        tiers = list(self.cfg.remote.tiers)

        def bump(*subs: str) -> list[str]:
            hits = [m for m in tiers if any(s in m.lower() for s in subs)]
            rest = [m for m in tiers if m not in hits]
            return (hits + rest) if hits else tiers

        if category in ("code_generation", "code_debugging"):
            return bump("code", "kimi")
        if category in ("sentiment", "summarization"):
            return bump("gemma")
        # factual, ner, math/logic votes, generic fallback
        return bump("kimi", "code")

    def _remote_longform(self, task: Task, category: str,
                         features: dict) -> Decision:
        """Single remote call for long-form categories (code, summaries).

        No voting: majority vote needs string-comparable answers, and two
        correct code solutions rarely match textually. One strong-model call
        with a category-tuned prompt is both the cheapest and the most
        accurate option here.
        """
        system, budget = _LONGFORM[category]
        return self._remote_single(task, system, f"remote_{category}",
                                   features, max_tokens=budget,
                                   category=category)

    def _remote_single(self, task: Task, system: str, route: str,
                       features: dict, max_tokens: int | None = None,
                       category: str | None = None) -> Decision:
        """One remote generation, escalating through the category-ordered
        tier list on failure OR on an empty extracted answer (a truncated
        reasoning channel can return 200 OK with blank content — an empty
        answer is a guaranteed zero, so it must never be final)."""
        remote_tokens = 0
        answer = ""
        for i, tier_model in enumerate(self._tier_order(category)):
            try:
                resp = self._remote.chat(
                    tier_model, system, task.prompt,
                    max_tokens=max_tokens or self.cfg.remote.max_tokens,
                    temperature=self.cfg.remote.temperature)
            except Exception as exc:
                features[f"tier{i + 1}_error"] = str(exc)[:200]
                continue
            remote_tokens += resp.total_tokens
            answer = str(extract_json_field(resp.text, "answer")
                         or resp.text.strip())
            if answer:
                features["model"] = tier_model
                break
        if not answer:
            answer = self._local_last_resort(task, system, features)
            route += "_local_fallback"
        self.cache.store(task.prompt, answer)
        return Decision(task.id, task.prompt, route, answer,
                        remote_tokens=remote_tokens, local_tokens=0,
                        latency_s=0.0, features=features)

    def _local_last_resort(self, task: Task, system: str,
                           features: dict) -> str:
        """Free local draft, used ONLY when every remote tier failed or came
        back blank — a plausible local answer beats a certain-zero blank."""
        try:
            resp = self._local_chat(self._local_a, self.cfg.local.model_a,
                                    system, task.prompt)
            return str(extract_json_field(resp.text, "answer")
                       or resp.text.strip())
        except Exception as exc:
            features["local_fallback_error"] = str(exc)[:200]
            return ""

    def _remote_self_consistency(self, task: Task) -> Decision:
        """Cross-FAMILY majority vote for math/logic.

        Sampling the same model (or the same weights re-quantized) repeatedly
        only catches random slips — a systematic reasoning bias reproduces in
        every sample, which is exactly what sank the Gemma-only vote plan on
        the real grading run. v4 votes with genuinely independent models:
        the strongest general instruct model, the dedicated reasoning model,
        then a Gemma checkpoint as tie-breaker. Early stop when the first two
        families agree on a non-blank answer (2-of-2 cross-family agreement
        is a stronger signal than 3-of-3 same-family ever was)."""
        general = self._tier_order(None)  # kimi-first
        reasoning = [m for m in self.cfg.remote.tiers
                     if any(s in m.lower() for s in ("minimax", "m3", "r1"))]
        gemma = [m for m in self.cfg.remote.tiers if "gemma" in m.lower()]
        plan = [general[0]]
        if reasoning and reasoning[0] != plan[0]:
            plan.append(reasoning[0])
        elif len(general) > 1:
            plan.append(general[1])
        plan.append(gemma[0] if gemma else general[0])
        plan.append(general[0])  # 4th vote breaks a 1-1-1 split toward kimi

        answers: list[str] = []
        remote_tokens = 0
        for i, model in enumerate(plan):
            # Reasoning-channel models (minimax-m3, gemma4-class) burn budget
            # on hidden reasoning before the JSON content — too small a
            # max_tokens returns 200 OK with EMPTY content (observed on
            # r42/r44 at 320 tokens). 1200 still clears the hardest multi-step
            # problems in our grading-style set without truncating (validated
            # 19/19), while capping any run-away chain sooner than the old
            # 2000. temp 0.3 keeps the reasoning direct rather than rambling;
            # usage bills actual tokens, not the cap.
            try:
                resp = self._remote.chat(model, REASONING_ANSWER_SYSTEM_PROMPT,
                                         task.prompt,
                                         max_tokens=max(self.cfg.remote.max_tokens, 1200),
                                         temperature=0.3)
            except Exception:
                answers.append("")  # failed vote counts as blank, not fatal
                continue
            remote_tokens += resp.total_tokens
            answers.append(str(extract_json_field(resp.text, "answer")
                               or resp.text.strip()))
            if i == 1:
                agree, _ = answers_agree(answers[0], answers[1])
                if agree and normalize_answer(answers[0]):
                    break
        answer = majority_vote(answers)
        if not normalize_answer(answer):
            answer = self._local_last_resort(
                task, REASONING_ANSWER_SYSTEM_PROMPT, {})
        self.cache.store(task.prompt, answer)
        return Decision(task.id, task.prompt, "remote_self_consistency",
                        answer, remote_tokens=remote_tokens, local_tokens=0,
                        latency_s=0.0,
                        features={"computational": True, "votes": answers,
                                  "vote_models": plan[:len(answers)]})

    # ---------- plumbing ----------

    def _local_chat(self, client, model, system, user):
        return client.chat(model, system, user,
                           max_tokens=self.cfg.local.max_tokens,
                           temperature=self.cfg.local.temperature)

    def _remote_chat(self, model, system, user):
        return self._remote.chat(model, system, user,
                                 max_tokens=self.cfg.remote.max_tokens,
                                 temperature=self.cfg.remote.temperature)
