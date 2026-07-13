"""The Tiered Calibrated Cascade router (03_ARCHITECTURE).

Route order (strategy: cascade, v6 max-free revision):
  1. within-run dedup cache      -> hit: return, 0 tokens
  2. category classifier         -> pick the plan per category:
     - sentiment / factual /     -> dual-local agreement first (0 tokens);
       math / logic                 disagreement, blank, or error escalates
                                    to ONE strong-model call (reasoning
                                    prompt + blank-proof cap for math/logic)
     - summarization             -> single local summary (0 tokens), guarded
                                    by a mechanical length-constraint check;
                                    violations escalate remote Gemma-first
     - code gen / debug / NER    -> minimax-first (terse JSON completions;
                                    measured ~1/2 to 1/3 the billed tokens
                                    of kimi for equal answers), kimi next
                                    tier on blank/failure
  3. local models                -> emergency fallback ONLY (all remote
                                    tiers failed or returned blank)

Scoring reality (live leaderboard, 2026-07-13): rank is ascending REMOTE
tokens above a ~50% accuracy floor — tokens decide placement, accuracy just
has to clear the bar. The evolution: v3 let the local pair answer anything
they agreed on and scored 57.9% (locals answering code/NER/hard-math is
fatal); v4 sent everything remote with math voting (19/19, 8,618 tokens);
v5 cut the never-dissenting vote and moved to measured-cheapest remote
models (19/19, 4,603); v6 re-admits the free local tier — but ONLY in the
categories the probe data shows agreement is trustworthy (short answers:
sentiment/factual/math/logic) or a mechanical guard exists (summaries),
never code or NER, and every doubt still escalates to the same strong
remote path v5 validated.

Also implements the fallback strategies from 02_MVP_PLAN so every stage of
the build is submittable: always_remote (v0), always_local (v1),
heuristic (v2).
"""

from __future__ import annotations

import json
import re
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

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers_match(a: str, b: str) -> bool:
    """True when both answers contain exactly ONE number and it's the same —
    catches '1989' vs '{ "year": 1989 }' that string agreement misses.
    Requiring a single number keeps work-dumps ('240 + 180 = 420') from
    fake-matching a bare final answer."""
    na, nb = _NUMBER_RE.findall(str(a)), _NUMBER_RE.findall(str(b))
    return len(na) == 1 and len(nb) == 1 and float(na[0]) == float(nb[0])


_CAP_WORD_RE = re.compile(r"\b[A-Z][a-z]+")


def _covers_capitalized_phrases(prompt: str, answer: str) -> bool:
    """True when every mid-sentence capitalized word in the source text
    (proper-noun candidates — sentence-initial words are skipped) appears in
    the answer. This is the mechanical completeness gate for local NER:
    small models' failure mode is DROPPING entities, and a dropped entity
    always drops its capitalized words."""
    lowered = answer.lower()
    for match in _CAP_WORD_RE.finditer(prompt):
        head = prompt[:match.start()].rstrip()
        if not head or head[-1] in ".!?:":
            continue  # sentence-initial word, not a proper-noun signal
        if match.group(0).lower() not in lowered:
            return False
    return True


_WORD_LIMIT_RE = re.compile(
    r"(?:no more than|at most|maximum of|max(?:imum)?|in|within|under)\s+"
    r"(\d+)\s+words", re.IGNORECASE)
_ONE_SENTENCE_RE = re.compile(r"\b(?:one|1|a single)\s+sentence", re.IGNORECASE)


def _meets_length_constraint(prompt: str, answer: str) -> bool:
    """Mechanical check of explicit length constraints in a summarization
    task — a judge fails a 20-word answer to a '15 words max' task no matter
    how good the summary is, so constraint violations must escalate."""
    m = _WORD_LIMIT_RE.search(prompt)
    if m and len(answer.split()) > int(m.group(1)):
        return False
    if _ONE_SENTENCE_RE.search(prompt):
        # crude sentence count: terminators followed by more text
        if len(re.findall(r"[.!?](?=\s+\S)", answer.strip())) >= 1:
            return False
    return True


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
            return self._local_pair_first(task, features, "sentiment",
                                          SENTIMENT_SYSTEM_PROMPT,
                                          SENTIMENT_SYSTEM_PROMPT, 800)
        if category == "factual":
            return self._local_pair_first(task, features, "factual",
                                          ANSWER_SYSTEM_PROMPT,
                                          ANSWER_SYSTEM_PROMPT, 800)
        if category in ("math", "logic"):
            features["computational"] = True
            # v5 dropped the remote confirmation vote (it never changed the
            # first model's answer in any validated run; the reasoning model
            # was measured WRONG on a trick question the general model got
            # right). v6 puts the free pair in front: agreement on a bare
            # number between two 1B lineages is right more often than not,
            # and every disagreement still escalates to the strong model
            # with the reasoning prompt and blank-proof 1200 cap.
            return self._local_pair_first(
                task, features, "reasoning",
                REASONING_ANSWER_SYSTEM_PROMPT, REASONING_ANSWER_SYSTEM_PROMPT,
                max(self.cfg.remote.max_tokens, 1200), local_max_tokens=400)
        if category == "summarization":
            return self._local_summary(task, features)
        if category == "ner":
            return self._local_ner(task, features)
        if category in _LONGFORM:
            return self._remote_longform(task, category, features)
        system = _LOCAL_PROMPTS.get(category, ANSWER_SYSTEM_PROMPT)
        return self._remote_single(task, system, f"remote_{category}",
                                   features, max_tokens=800,
                                   category=category)

    def _local_pair_first(self, task: Task, features: dict, kind: str,
                          local_system: str, remote_system: str,
                          remote_max: int,
                          local_max_tokens: int | None = None) -> Decision:
        """Free tier: both local models answer; accept ONLY when the two
        different-lineage 1B models agree on the short answer (fuzzy match,
        plus a single-number match so '1989' agrees with '{"year": 1989}').
        Agreement between independent lineages is a genuine signal on short
        answers; any disagreement, error, or blank escalates to the remote
        path, so a local miss costs latency, never an answer."""
        local_tokens = 0
        answers = []
        for client, model in ((self._local_a, self.cfg.local.model_a),
                              (self._local_b, self.cfg.local.model_b)):
            try:
                resp = client.chat(model, local_system, task.prompt,
                                   max_tokens=(local_max_tokens
                                               or self.cfg.local.max_tokens),
                                   temperature=self.cfg.local.temperature)
            except Exception as exc:
                features["local_error"] = str(exc)[:200]
                break
            local_tokens += resp.total_tokens
            answers.append(str(extract_json_field(resp.text, "answer")
                               or resp.text.strip()))
        if len(answers) == 2 and normalize_answer(answers[0]):
            agree, _ = answers_agree(answers[0], answers[1])
            if not agree:
                agree = _numbers_match(answers[0], answers[1])
            if agree:
                answer = min(answers, key=len)  # tersest phrasing of the match
                self.cache.store(task.prompt, answer)
                features["local_answers"] = answers
                return Decision(task.id, task.prompt, f"local_{kind}",
                                answer, remote_tokens=0,
                                local_tokens=local_tokens, latency_s=0.0,
                                features=features)
        features["local_disagreement"] = answers
        route_cat = "math" if kind == "reasoning" else kind
        decision = self._remote_single(task, remote_system,
                                       f"remote_{kind}", features,
                                       max_tokens=remote_max,
                                       category=route_cat)
        decision.local_tokens = local_tokens
        return decision

    def _local_summary(self, task: Task, features: dict) -> Decision:
        """Free tier for summarization: the stronger local model (qwen2.5
        1.5B) writes the summary. Long-form answers can't be cross-checked
        by string agreement, so the guard is constraint compliance instead:
        an explicit word limit or one-sentence requirement in the task is
        checked mechanically. A violation gets ONE free local retry with the
        limit spelled out; a second violation, blank, or error escalates to
        the remote (Gemma-first) path."""
        local_tokens = 0
        prompts = [task.prompt,
                   task.prompt + "\nIMPORTANT: obey the length limit "
                                 "EXACTLY — count your words."]
        for attempt, user in enumerate(prompts):
            try:
                resp = self._local_chat(self._local_b,
                                        self.cfg.local.model_b,
                                        SUMMARY_SYSTEM_PROMPT, user)
            except Exception as exc:
                features["local_error"] = str(exc)[:200]
                break
            local_tokens += resp.total_tokens
            answer = str(extract_json_field(resp.text, "answer")
                         or resp.text.strip())
            if answer and len(answer) >= 20 and _meets_length_constraint(
                    task.prompt, answer):
                self.cache.store(task.prompt, answer)
                features["local_attempts"] = attempt + 1
                return Decision(task.id, task.prompt,
                                "local_summarization", answer,
                                remote_tokens=0, local_tokens=local_tokens,
                                latency_s=0.0, features=features)
            features[f"local_rejected_{attempt + 1}"] = answer[:200]
        decision = self._remote_single(task, SUMMARY_SYSTEM_PROMPT,
                                       "remote_summarization", features,
                                       max_tokens=800,
                                       category="summarization")
        decision.local_tokens = local_tokens
        return decision

    def _local_ner(self, task: Task, features: dict) -> Decision:
        """Free tier for NER, guarded by mechanical COMPLETENESS: the one
        failure mode of small models on NER is silently dropping entities
        (observed live: a 1B dropping 'Tim Cook'), so the local answer is
        accepted only if every capitalized phrase in the source text
        reappears in it. Judges grade NER on completeness; an answer that
        provably names every candidate is safe to take for free, and
        anything less escalates to the remote (minimax-first) path."""
        try:
            resp = self._local_chat(self._local_a, self.cfg.local.model_a,
                                    NER_SYSTEM_PROMPT, task.prompt)
            answer = str(extract_json_field(resp.text, "answer")
                         or resp.text.strip())
            local_tokens = resp.total_tokens
        except Exception as exc:
            features["local_error"] = str(exc)[:200]
            answer, local_tokens = "", 0
        if answer and _covers_capitalized_phrases(task.prompt, answer):
            self.cache.store(task.prompt, answer)
            return Decision(task.id, task.prompt, "local_ner", answer,
                            remote_tokens=0, local_tokens=local_tokens,
                            latency_s=0.0, features=features)
        features["local_incomplete"] = answer[:200]
        decision = self._remote_single(task, NER_SYSTEM_PROMPT,
                                       "remote_ner", features,
                                       max_tokens=800, category="ner")
        decision.local_tokens = local_tokens
        return decision

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
