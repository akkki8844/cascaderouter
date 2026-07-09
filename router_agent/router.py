"""The Tiered Calibrated Cascade router (03_ARCHITECTURE).

Route order (strategy: cascade, the target v3 submission):
  1. semantic cache        -> hit: return, 0 tokens
  2. dual local models     -> agree: return, 0 tokens
  3. local critique        -> confident verdict: return, 0 tokens
  4. remote tier 1 (Gemma) -> verify-or-fix the best local draft
  5. remote tier 2         -> only if tier 1 signals it could not verify

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
    "code_generation": (CODEGEN_SYSTEM_PROMPT, 900),
    "code_debugging": (CODEDEBUG_SYSTEM_PROMPT, 900),
    "summarization": (SUMMARY_SYSTEM_PROMPT, 400),
    "ner": (NER_SYSTEM_PROMPT, 300),
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
        features: dict = {}

        # 1. cache (within-run dedup only in harness mode — see cache.py)
        cached = self.cache.lookup(task.prompt)
        if cached is not None:
            return Decision(task.id, task.prompt, "cache", cached,
                            remote_tokens=0, local_tokens=0, latency_s=0.0,
                            features={"cache_hit": True})

        # 1.5 capability-category dispatch (the eight Track 1 categories).
        #  - math/logic bypass local-agreement trust: small local models can
        #    confidently agree on the same wrong answer (correlated errors),
        #    so agreement isn't a valid free-tier signal — go straight to
        #    remote self-consistency voting.
        #  - code/summarization/NER skip the cascade: one remote call with a
        #    category-tuned prompt (see _LONGFORM for the per-category why).
        #  - factual/sentiment stay on the free dual-local path.
        category = classify_category(task.prompt)
        features["category"] = category
        if category in ("math", "logic"):
            return self._remote_self_consistency(task)
        if category in _LONGFORM:
            return self._remote_longform(task, category, features)

        # 2. dual local models (both free)
        local_prompt = _LOCAL_PROMPTS.get(category, ANSWER_SYSTEM_PROMPT)
        try:
            resp_a = self._local_chat(self._local_a, self.cfg.local.model_a,
                                      local_prompt, task.prompt)
            resp_b = self._local_chat(self._local_b, self.cfg.local.model_b,
                                      local_prompt, task.prompt)
        except Exception as exc:
            # Local tier unreachable/failed (e.g. Ollama not up in the judge
            # VM) — never fail the task over the free tier: answer remote.
            features["local_error"] = str(exc)[:200]
            return self._remote_single(task, local_prompt, "remote_fallback",
                                       features)
        ans_a = str(extract_json_field(resp_a.text, "answer")
                    or resp_a.text.strip())
        ans_b = str(extract_json_field(resp_b.text, "answer")
                    or resp_b.text.strip())
        local_tokens = resp_a.total_tokens + resp_b.total_tokens

        agree, similarity = answers_agree(
            ans_a, ans_b, self.cfg.router.agreement_fuzzy_threshold)
        features.update({"agreement": agree, "similarity": round(similarity, 3),
                         "prompt_chars": len(task.prompt)})

        if agree:
            self.cache.store(task.prompt, ans_a)
            return Decision(task.id, task.prompt, "local_agreement", ans_a,
                            remote_tokens=0, local_tokens=local_tokens,
                            latency_s=0.0, features=features)

        # 3. local critique (still free)
        draft = ans_a
        if self.cfg.router.critique_enabled:
            critic_user = (
                f"Task: {task.prompt}\n\n"
                f"Candidate A: {ans_a}\n"
                f"Candidate B: {ans_b}")
            try:
                crit = self._local_chat(self._critic,
                                        self.cfg.local.critic_model,
                                        CRITIC_SYSTEM_PROMPT, critic_user)
            except Exception as exc:
                crit = None
                features["critic_error"] = str(exc)[:200]
            if crit is not None:
                local_tokens += crit.total_tokens
                verdict, conf = extract_json_field(crit.text,
                                                   "verdict", "confidence")
                features.update({"critique_verdict": verdict,
                                 "critique_confidence": conf})
                if verdict in ("A", "B"):
                    draft = ans_a if verdict == "A" else ans_b
                    if str(conf).lower() == "high":
                        self.cache.store(task.prompt, draft)
                        return Decision(task.id, task.prompt, "local_critique",
                                        draft, remote_tokens=0,
                                        local_tokens=local_tokens,
                                        latency_s=0.0, features=features)

        # 4/5. remote verify-not-regenerate, tiered (Gemma first)
        verify_user = f"Task: {task.prompt}\n\nDraft answer: {draft}"
        remote_tokens = 0
        answer = draft
        route = "remote_tier1"
        for i, tier_model in enumerate(self.cfg.remote.tiers):
            try:
                resp = self._remote_chat(tier_model, VERIFY_SYSTEM_PROMPT,
                                         verify_user)
            except Exception as exc:
                features[f"tier{i + 1}_error"] = str(exc)[:200]
                continue  # next tier; worst case we keep the local draft
            remote_tokens += resp.total_tokens
            correct, fixed = extract_json_field(resp.text, "correct", "answer")
            route = f"remote_tier{i + 1}"
            if fixed:
                answer = str(fixed)
            if correct is not None:  # tier gave a usable verdict -> stop
                features[f"tier{i + 1}_verdict"] = bool(correct)
                break
            # verdict unparseable -> try next tier, if any

        self.cache.store(task.prompt, answer)
        return Decision(task.id, task.prompt, route, answer,
                        remote_tokens=remote_tokens, local_tokens=local_tokens,
                        latency_s=0.0, features=features)

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
                                   features, max_tokens=budget)

    def _remote_single(self, task: Task, system: str, route: str,
                       features: dict, max_tokens: int | None = None) -> Decision:
        """One remote generation on tier 1, falling back through the tier
        list if a call fails. Used for long-form categories and as the
        safety net when the local tier is unreachable."""
        remote_tokens = 0
        answer = ""
        for i, tier_model in enumerate(self.cfg.remote.tiers):
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
                break
        self.cache.store(task.prompt, answer)
        return Decision(task.id, task.prompt, route, answer,
                        remote_tokens=remote_tokens, local_tokens=0,
                        latency_s=0.0, features=features)

    def _remote_self_consistency(self, task: Task) -> Decision:
        """Fresh-generate across remote tiers at temperature>0 and take the
        majority vote. Sampling the SAME model repeatedly only catches random
        slips — a systematic reasoning bias in that model gets reproduced by
        every sample. Spreading votes across independently-trained remote
        models (Gemma + Llama3) catches cases where one model is confidently,
        consistently wrong (03_ARCHITECTURE §3 revision)."""
        tiers = self.cfg.remote.tiers
        # 3 votes from tier0 (the stronger primary model), 1 from tier1, so
        # a genuine capability gap between tiers doesn't tie 2-2 -- tier0 is
        # trusted more, tier1 only needed to catch tier0-specific mistakes.
        # Early stop: if the first two tier0 votes already agree on a
        # non-blank answer, further votes almost never flip the majority --
        # skip them and save ~half the remote tokens on easy math.
        tier1 = tiers[1] if len(tiers) > 1 else tiers[0]
        plan = [tiers[0], tiers[0], tiers[0], tier1]
        answers: list[str] = []
        remote_tokens = 0
        for i, model in enumerate(plan):
            # gemma4/qwen3-class models emit a hidden 'reasoning' channel
            # before the JSON content -- too small a budget lets reasoning
            # consume the whole response and leaves content empty (observed:
            # 320 tokens produced empty votes on r42/r44; 900+ is reliable).
            try:
                resp = self._remote.chat(model, REASONING_ANSWER_SYSTEM_PROMPT,
                                         task.prompt,
                                         max_tokens=max(self.cfg.remote.max_tokens, 900),
                                         temperature=0.5)
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
        self.cache.store(task.prompt, answer)
        return Decision(task.id, task.prompt, "remote_self_consistency",
                        answer, remote_tokens=remote_tokens, local_tokens=0,
                        latency_s=0.0,
                        features={"computational": True, "votes": answers})

    # ---------- plumbing ----------

    def _local_chat(self, client, model, system, user):
        return client.chat(model, system, user,
                           max_tokens=self.cfg.local.max_tokens,
                           temperature=self.cfg.local.temperature)

    def _remote_chat(self, model, system, user):
        return self._remote.chat(model, system, user,
                                 max_tokens=self.cfg.remote.max_tokens,
                                 temperature=self.cfg.remote.temperature)
