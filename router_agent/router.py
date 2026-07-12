"""The Tiered Calibrated Cascade router (03_ARCHITECTURE).

Route order (strategy: cascade, v5 token-lean revision):
  1. within-run dedup cache      -> hit: return, 0 tokens
  2. category classifier         -> pick the plan per category:
     - sentiment                 -> dual-local agreement (0 tokens), remote
                                    only on disagreement or blank
     - math / logic              -> ONE call to the strongest general model
                                    (escalates through tiers only on blank)
     - code gen / debug / NER    -> minimax-first (terse JSON completions;
                                    measured ~1/2 to 1/3 the billed tokens
                                    of kimi for equal answers), kimi next
                                    tier on blank/failure
     - summarization             -> Gemma-first, single call
     - factual                   -> strongest general model, single call
  3. local models                -> emergency fallback ONLY (all remote
                                    tiers failed or returned blank)

Scoring reality (live leaderboard, 2026-07-13): rank is ascending REMOTE
tokens subject to a 50% accuracy floor — entries at 52.6% rank normally, so
the floor is lenient, but tokens decide placement. v4 routed everything to
strong remote models and voted on math (8,618 tokens, 19/19 on the proxy
set); v5 keeps a strong remote model on every graded category (the 57.9%
real run showed 1B locals answering broadly is fatal) while cutting the
redundant math vote, moving NER/debug to the cheap-completion model, and
letting the free local pair handle only sentiment — their one reliably
safe category — with a remote fallback on any disagreement.

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
        """v5 token-lean cascade (see module docstring for the full plan).

        Every graded category still gets a strong remote model — the 57.9%
        real run proved broad local answering is fatal — but redundancy the
        floor no longer justifies is gone: math/logic take ONE strong-model
        call instead of a cross-family vote (the vote never changed the
        first model's answer in validation), and sentiment goes to the free
        local pair first because a bare polarity label is the one output two
        1B models can be trusted to cross-check, with remote on any doubt.
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
        if category == "sentiment":
            return self._local_sentiment(task, features)
        if category in ("math", "logic"):
            return self._remote_reasoning_single(task, features)
        if category in _LONGFORM:
            return self._remote_longform(task, category, features)

        # factual: one short remote call to the strongest general model.
        # Completion is a few tokens; the accuracy delta over a 1B local
        # model is the whole ballgame.
        system = _LOCAL_PROMPTS.get(category, ANSWER_SYSTEM_PROMPT)
        return self._remote_single(task, system, f"remote_{category}",
                                   features, max_tokens=800,
                                   category=category)

    def _local_sentiment(self, task: Task, features: dict) -> Decision:
        """Free tier for sentiment ONLY: both local models label the text;
        a fuzzy match between two different-lineage 1B models on a bare
        polarity label is a genuine agreement signal (unlike free-form
        answers, where correlated confident-wrongs sank the 57.9% run).
        Any disagreement, error, or blank goes to the remote path."""
        local_tokens = 0
        labels = []
        for client, model in ((self._local_a, self.cfg.local.model_a),
                              (self._local_b, self.cfg.local.model_b)):
            try:
                resp = self._local_chat(client, model,
                                        SENTIMENT_SYSTEM_PROMPT, task.prompt)
            except Exception as exc:
                features["local_sentiment_error"] = str(exc)[:200]
                break
            local_tokens += resp.total_tokens
            labels.append(str(extract_json_field(resp.text, "answer")
                              or resp.text.strip()))
        if len(labels) == 2 and normalize_answer(labels[0]):
            agree, _ = answers_agree(labels[0], labels[1])
            if agree:
                self.cache.store(task.prompt, labels[0])
                features["labels"] = labels
                return Decision(task.id, task.prompt, "local_sentiment",
                                labels[0], remote_tokens=0,
                                local_tokens=local_tokens, latency_s=0.0,
                                features=features)
        features["local_disagreement"] = labels
        decision = self._remote_single(task, SENTIMENT_SYSTEM_PROMPT,
                                       "remote_sentiment", features,
                                       max_tokens=800, category="sentiment")
        decision.local_tokens = local_tokens
        return decision

    def _remote_reasoning_single(self, task: Task, features: dict) -> Decision:
        """ONE strong-model call for math/logic, tier-escalating only on
        blank/failure. Replaces the v4 cross-family vote: in every validated
        run the second vote merely confirmed the first model's answer (and
        the reasoning model was measured WRONG on a trick question the
        general model got right), so the confirmation call was pure token
        cost. The reasoning-tuned prompt and generous cap stay — truncated
        reasoning channels return blank, and blank is a guaranteed zero."""
        features["computational"] = True
        return self._remote_single(
            task, REASONING_ANSWER_SYSTEM_PROMPT, "remote_reasoning",
            features, max_tokens=max(self.cfg.remote.max_tokens, 1200),
            category="math")

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

        if category in ("code_generation", "code_debugging", "ner"):
            # minimax-m3 answers these with terse, deterministic JSON
            # completions — measured ~290-380 total tokens/task vs kimi's
            # volatile 400-940 for answers the judge grades identically
            # (identical output across repeated codegen runs at temp 0).
            # Falls back to the kimi order when no minimax checkpoint is on
            # the allowed list; kimi is the next tier for a blank/failure.
            return bump("minimax", "m3")
        if category in ("sentiment", "summarization"):
            return bump("gemma")
        # factual, math/logic, generic fallback
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

    # ---------- plumbing ----------

    def _local_chat(self, client, model, system, user):
        return client.chat(model, system, user,
                           max_tokens=self.cfg.local.max_tokens,
                           temperature=self.cfg.local.temperature)

    def _remote_chat(self, model, system, user):
        return self._remote.chat(model, system, user,
                                 max_tokens=self.cfg.remote.max_tokens,
                                 temperature=self.cfg.remote.temperature)
