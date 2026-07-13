"""The Tiered Calibrated Cascade router (03_ARCHITECTURE).

Route order (strategy: cascade, v8 min-token revision):
  1. within-run dedup cache      -> hit: return, 0 tokens
  2. category classifier         -> pick the plan per category:
     - sentiment / factual       -> dual-local agreement first (0 tokens);
                                    disagreement, blank, or error escalates
                                    to ONE strong-model call
     - math / logic              -> dual-local agreement first; on
                                    disagreement the STRONGER local's
                                    distilled answer stands (probe-verified
                                    it wins every measured disagreement) —
                                    remote only on blank/error
     - summarization             -> local summary (0 tokens) behind a
                                    mechanical length-constraint check, with
                                    escalating free retries that restate the
                                    limit; violations escalate Gemma-first
     - NER                       -> local behind a completeness + type-
                                    sanity guard, with a free hint-retry
                                    naming the exact entities the first
                                    attempt missed; failures go minimax-first
     - code generation           -> local behind compile + function-name
                                    guards; failures go minimax-first
     - code debugging            -> BOTH locals produce forced-code fixes;
                                    accepted only when both compile, both
                                    change the buggy code, and the two fixes
                                    agree BEHAVIORALLY (executed on a probe
                                    battery in a sandboxed subprocess);
                                    anything less goes minimax-first
  3. local models                -> emergency fallback ONLY (all remote
                                    tiers failed, returned blank, or were
                                    refused by the remote token budget)

Hard remote budget (v9): every remote call must first reserve its worst
case against a run-wide ceiling on billed Fireworks tokens (harness
default 480 — the leaderboard ranks by ascending remote tokens, so the
scored number is bounded by construction, no matter what the hidden task
set looks like). A refused call gets the free local answer instead:
over budget we risk one answer, never the rank.

Scoring reality (live leaderboard, 2026-07-13): rank is ascending REMOTE
tokens above a ~50% accuracy floor — tokens decide placement, accuracy just
has to clear the bar. The evolution: v3 let the local pair answer anything
they agreed on and scored 57.9% (locals answering code/NER/hard-math is
fatal); v4 sent everything remote with math voting (19/19, 8,618 tokens);
v5 cut the never-dissenting vote and moved to measured-cheapest remote
models (19/19, 4,603); v6/v7 re-admitted the free local tier behind hard
mechanical guards (17/19, 2,646); v8 closes the remaining paid routes the
probes proved the locals can win — behavioral dual-agreement on debug
fixes, hint-retries on NER completeness, hard-budget retries on summary
limits, stronger-local fallback on math/logic splits — leaving remote
spend only where the locals are measurably wrong.

Also implements the fallback strategies from 02_MVP_PLAN so every stage of
the build is submittable: always_remote (v0), always_local (v1),
heuristic (v2).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
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
    CODEDEBUG_FIX_SYSTEM_PROMPT,
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


def _missing_capitalized_words(prompt: str, answer: str) -> list[str]:
    """Mid-sentence capitalized words of the source text (proper-noun
    candidates — sentence-initial words are skipped) that do NOT appear in
    the answer. This is the mechanical completeness gate for local NER:
    small models' failure mode is DROPPING entities, and a dropped entity
    always drops its capitalized words. The list doubles as the hint for
    the free local retry ('you missed: ...')."""
    lowered = answer.lower()
    missing: list[str] = []
    for match in _CAP_WORD_RE.finditer(prompt):
        head = prompt[:match.start()].rstrip()
        if not head or head[-1] in ".!?:":
            continue  # sentence-initial word, not a proper-noun signal
        word = match.group(0)
        if word.lower() not in lowered and word not in missing:
            missing.append(word)
    return missing


def _covers_capitalized_phrases(prompt: str, answer: str) -> bool:
    return not _missing_capitalized_words(prompt, answer)


_ENTITY_TYPE_WORDS = {
    "person", "people", "organization", "organisation", "company",
    "location", "place", "city", "country", "date", "time", "event",
    "product", "group",
}


def _has_sane_entity_types(answer: str) -> bool:
    """A complete NER answer can still be judge-fatal if its type labels are
    gibberish (observed live: a 1.5B labeling entities '(NP)', '(Pro)',
    '(J)'). Require at least two recognizable type words before a local NER
    answer counts."""
    words = set(re.findall(r"[a-z]+", answer.lower()))
    return len(words & _ENTITY_TYPE_WORDS) >= 2


def _distill_short_answer(category: str | None, raw_text: str) -> str:
    """Reduce a local model's possibly-verbose reasoning output to the short
    answer a judge grades: the 'answer' JSON field when present, else the
    final number of the work (math) or the concluding sentence (logic)."""
    ans, work = extract_json_field(raw_text, "answer", "work")
    text = str(ans if ans is not None else
               (work if work is not None else raw_text)).strip()
    if not text:
        return ""
    if category == "math":
        nums = _NUMBER_RE.findall(text)
        if nums:
            return nums[-1]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences[-1].strip() if sentences else text


def _extract_code(text: str) -> str | None:
    """The Python code portion of a model answer (or task prompt): from the
    first import/from/class that precedes the first def, else from the
    first def. None when there is no def at all."""
    if "def " not in text:
        return None
    start = text.index("def ")
    for prefix in ("import ", "from ", "class "):
        idx = text.find(prefix)
        if 0 <= idx < start:
            start = idx
    return text[start:]


# Runs in a sandboxed subprocess (crash/hang isolation): loads both models'
# fixes, then calls every function name they both define on a small battery
# of generic probe inputs. Prints YES only when at least one call succeeded
# on both fixes and every comparable call returned the same value.
_BEHAVIOR_CHECK_SCRIPT = r"""
import inspect, json, sys
payload = json.loads(sys.stdin.read())

def load(src):
    ns = {}
    exec(compile(src, "<fix>", "exec"), ns)
    return {k: v for k, v in ns.items()
            if callable(v) and not k.startswith("_")}

try:
    fa, fb = load(payload["a"]), load(payload["b"])
except Exception:
    print("NO"); sys.exit(0)
BATTERY = {
    1: [(3,), (7,), (0,), ("hello world",), ([3, 1, 2, 2],)],
    2: [(3, 4), (10, 3), ("hello world", "o"), ([3, 1, 2], 2)],
}
agreed = 0
for name in [k for k in fa if k in fb]:
    try:
        arity = len(inspect.signature(fa[name]).parameters)
    except Exception:
        continue
    for args in BATTERY.get(arity, []):
        try:
            ra = fa[name](*args)
        except Exception:
            continue  # input outside this function's domain — skip
        try:
            rb = fb[name](*args)
        except Exception:
            print("NO"); sys.exit(0)  # one runs where the other crashes
        if ra != rb:
            print("NO"); sys.exit(0)
        agreed += 1
print("YES" if agreed else "NO")
"""


def _debug_fixes_agree(code_a: str, code_b: str) -> bool:
    """Behavioral cross-check of two independent local fixes, executed in a
    killable subprocess (a bad fix can loop forever — 8s hard timeout)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _BEHAVIOR_CHECK_SCRIPT],
            input=json.dumps({"a": code_a, "b": code_b}),
            capture_output=True, text=True, timeout=8)
    except Exception:
        return False
    return proc.stdout.strip() == "YES"


_FUNC_NAME_RE = re.compile(r"(?:called|named)\s+`?(\w+)\s*\(", re.IGNORECASE)


def _code_passes_guards(prompt: str, answer: str) -> bool:
    """Mechanical acceptance test for locally generated code: it must
    contain a definition, compile as Python (the default/overwhelmingly
    requested language here), and define the function name when the task
    names one. Non-Python requests fail the compile check and escalate —
    conservative by design."""
    if "def " not in answer or len(answer) < 20:
        return False
    code = answer[answer.index("def "):]
    for prefix in ("import", "from", "class"):
        idx = answer.find(prefix + " ")
        if 0 <= idx < answer.index("def "):
            code = answer[idx:]
            break
    try:
        compile(code, "<local-codegen>", "exec")
    except SyntaxError:
        return False
    m = _FUNC_NAME_RE.search(prompt)
    if m and f"def {m.group(1)}" not in answer:
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


# When the run-wide remote budget is active, each category's remote calls get
# a clamped completion cap (billed completion tokens can never exceed
# max_tokens) and a minimum viable grant below which the call is refused
# outright — a code answer truncated at 100 tokens bills real money for
# garbage, so it's cheaper to not make the call at all.
_BUDGET_COMPLETION: dict[str, tuple[int, int]] = {
    "code_generation": (700, 400),
    "code_debugging": (700, 400),
    "ner": (300, 150),
    "summarization": (300, 120),
    "math": (400, 120),
}
_BUDGET_COMPLETION_DEFAULT = (160, 48)  # short answers: factual/sentiment/etc.


class _RemoteBudget:
    """Hard run-wide ceiling on billed Fireworks tokens (leaderboard rank is
    ascending remote tokens — this converts 'probably cheap' into 'cannot
    exceed the cap on ANY task set').

    Reserve-then-settle: before a remote call, a conservative estimate
    (prompt chars/3 + margin, plus the FULL completion grant — max_tokens
    hard-bounds billed completion) is reserved under a lock, so two worker
    threads can't race past the limit; after the call the reservation is
    replaced by the actual billed usage. A call that doesn't fit is refused
    and the router falls back to the free local answer instead."""

    def __init__(self, limit: int):
        self.limit = limit  # <= 0 disables (local dev)
        self._lock = threading.Lock()
        self._spent = 0
        self._reserved = 0

    @property
    def active(self) -> bool:
        return self.limit > 0

    def reserve(self, prompt_chars: int, completion_cap: int,
                completion_min: int) -> tuple[int, int] | None:
        """Try to fit one call. Returns (reservation, granted_completion)
        or None when the call must not be made. prompt_chars/3 overcounts
        real tokenization (~4 chars/token) on purpose — the estimate must
        be an upper bound for the ceiling to be a guarantee."""
        est_prompt = prompt_chars // 3 + 40
        with self._lock:
            room = self.limit - self._spent - self._reserved
            granted = min(completion_cap, room - est_prompt)
            if granted < completion_min:
                return None
            reservation = est_prompt + granted
            self._reserved += reservation
            return reservation, granted

    def settle(self, reservation: int, actual: int) -> None:
        with self._lock:
            self._reserved -= reservation
            self._spent += actual

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent


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
        self._budget = _RemoteBudget(
            getattr(cfg.router, "remote_token_budget", 0))

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
        if category == "code_generation":
            return self._local_codegen(task, features)
        if category == "code_debugging":
            return self._local_codedebug(task, features)
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
        path, so a local miss costs latency, never an answer.

        Exception (v8): on a math/logic split the STRONGER local's distilled
        answer stands instead of paying for a remote call — in every
        measured disagreement on the validation set the 1.5B was right and
        the 1B wrong, so the split itself identifies which model to trust.
        Factual splits still escalate: there both locals were measured wrong
        together, so the split means neither can be trusted."""
        local_tokens = 0
        answers: list[str] = []
        raws: list[str] = []
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
            raws.append(resp.text)
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
        if kind == "reasoning" and len(answers) == 2:
            answer = _distill_short_answer(features.get("category"), raws[1])
            if normalize_answer(answer):
                self.cache.store(task.prompt, answer)
                return Decision(task.id, task.prompt, f"local_{kind}_solo",
                                answer, remote_tokens=0,
                                local_tokens=local_tokens, latency_s=0.0,
                                features=features)
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
        checked mechanically. A violation gets free local retries with the
        limit restated ever harder: first the limit spelled out, then a
        hard budget BELOW the task's limit, and finally a rewrite pass that
        asks the model to SHORTEN its own previous answer — compressing an
        existing sentence is far easier for a small model than regenerating
        under a limit. Retries cost local tokens only (score zero); a final
        violation, blank, or error escalates to the remote (Gemma-first)
        path."""
        local_tokens = 0
        limit = _WORD_LIMIT_RE.search(task.prompt)
        budget = max(int(limit.group(1)) - 3, 5) if limit else None
        prompts = [task.prompt,
                   task.prompt + "\nIMPORTANT: obey the length limit "
                                 "EXACTLY — count your words."]
        if budget:
            prompts.append(task.prompt + f"\nHARD RULE: your summary must "
                                         f"be {budget} words or fewer. "
                                         "Count every word.")
        elif _ONE_SENTENCE_RE.search(task.prompt):
            prompts.append(task.prompt + "\nHARD RULE: reply with EXACTLY "
                                         "ONE sentence.")
        last_answer = ""
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
            if answer and len(answer) >= 20:
                last_answer = answer
            if (budget and last_answer
                    and attempt == len(prompts) - 1 and len(prompts) < 5):
                prompts.append(  # rewrite pass: shorten the previous try
                    f"Shorten this to at most {budget} words, keeping the "
                    f"main point. Reply with the shortened text only:\n"
                    f"{last_answer}")
        decision = self._remote_single(task, SUMMARY_SYSTEM_PROMPT,
                                       "remote_summarization", features,
                                       max_tokens=800,
                                       category="summarization")
        decision.local_tokens = local_tokens
        return decision

    def _local_codegen(self, task: Task, features: dict) -> Decision:
        """Free tier for code GENERATION, guarded mechanically: the local
        answer is accepted only if it (a) actually compiles and (b) defines
        the function name the task asks for. A 1.5B model writes correct
        code for the straightforward functions these tasks ask for, and
        compile + name checks filter the failure modes we can detect
        without a judge; anything doubtful escalates to the remote
        (minimax-first) path. Code DEBUGGING stays remote: the same local
        model describes the bug without emitting the corrected code, which
        an intent judge fails."""
        try:
            resp = self._local_b.chat(self.cfg.local.model_b,
                                      CODEGEN_SYSTEM_PROMPT, task.prompt,
                                      max_tokens=600,
                                      temperature=self.cfg.local.temperature)
            answer = str(extract_json_field(resp.text, "answer")
                         or resp.text.strip())
            local_tokens = resp.total_tokens
        except Exception as exc:
            features["local_error"] = str(exc)[:200]
            answer, local_tokens = "", 0
        if answer and _code_passes_guards(task.prompt, answer):
            self.cache.store(task.prompt, answer)
            return Decision(task.id, task.prompt, "local_code_generation",
                            answer, remote_tokens=0,
                            local_tokens=local_tokens, latency_s=0.0,
                            features=features)
        features["local_rejected"] = answer[:200]
        system, budget = _LONGFORM["code_generation"]
        decision = self._remote_single(task, system,
                                       "remote_code_generation", features,
                                       max_tokens=budget,
                                       category="code_generation")
        decision.local_tokens = local_tokens
        return decision

    def _local_codedebug(self, task: Task, features: dict) -> Decision:
        """Free tier for code DEBUGGING (v8). Asking a small model to
        'identify and fix' yields prose about the bug; forcing code-only
        output yields an actual fix — which unlocks the strongest guard in
        the router: BOTH local lineages produce a fix, and the fixes are
        accepted only when each compiles, each actually changes the buggy
        code, and the two independently-written fixes agree BEHAVIORALLY
        when executed side by side on a probe battery (sandboxed subprocess,
        hard timeout). Two independent models converging on the same
        input/output behavior is far stronger evidence than either model's
        word. Anything less escalates to the remote (minimax-first) path."""
        buggy = _extract_code(task.prompt)
        local_tokens = 0
        fixes: list[str] = []
        if buggy is not None:
            buggy_norm = "".join(buggy.split())
            for client, model in ((self._local_a, self.cfg.local.model_a),
                                  (self._local_b, self.cfg.local.model_b)):
                try:
                    resp = client.chat(model, CODEDEBUG_FIX_SYSTEM_PROMPT,
                                       task.prompt, max_tokens=500,
                                       temperature=self.cfg.local.temperature)
                except Exception as exc:
                    features["local_error"] = str(exc)[:200]
                    break
                local_tokens += resp.total_tokens
                answer = str(extract_json_field(resp.text, "answer")
                             or resp.text.strip())
                code = _extract_code(answer)
                if code is None or "".join(code.split()) == buggy_norm:
                    features[f"local_rejected_{model}"] = answer[:150]
                    break  # no code, or returned the bug unchanged
                try:
                    compile(code, "<local-debug>", "exec")
                except SyntaxError:
                    features[f"local_rejected_{model}"] = answer[:150]
                    break
                fixes.append(code)
        if len(fixes) == 2 and _debug_fixes_agree(fixes[0], fixes[1]):
            answer = fixes[1]  # the stronger local's phrasing of the fix
            self.cache.store(task.prompt, answer)
            features["behavioral_agreement"] = True
            return Decision(task.id, task.prompt, "local_code_debugging",
                            answer, remote_tokens=0,
                            local_tokens=local_tokens, latency_s=0.0,
                            features=features)
        system, budget = _LONGFORM["code_debugging"]
        decision = self._remote_single(task, system,
                                       "remote_code_debugging", features,
                                       max_tokens=budget,
                                       category="code_debugging")
        decision.local_tokens = local_tokens
        return decision

    def _local_ner(self, task: Task, features: dict) -> Decision:
        """Free tier for NER, guarded by mechanical COMPLETENESS: the one
        failure mode of small models on NER is silently dropping entities
        (observed live: a 1B dropping 'Tim Cook'), so a local answer is
        accepted only if every capitalized phrase in the source text
        reappears in it — plus a type-sanity check, because a 'complete'
        answer whose type labels are gibberish still fails a judge.

        v8: a rejected attempt gets targeted free retries — an incomplete
        answer is retried with the exact capitalized words it missed
        (probe-verified this recovers dropped entities); a complete but
        untyped/garbage-typed answer is retried with its own name list and
        an order to label each one. Both local lineages get a turn before
        any remote call. Local attempts cost zero; only a full local
        failure pays for the remote (minimax-first) path."""
        local_tokens = 0
        for client, model in ((self._local_a, self.cfg.local.model_a),
                              (self._local_b, self.cfg.local.model_b)):
            user = task.prompt
            for attempt in range(3):
                try:
                    resp = client.chat(model, NER_SYSTEM_PROMPT, user,
                                       max_tokens=300,
                                       temperature=self.cfg.local.temperature)
                except Exception as exc:
                    features["local_error"] = str(exc)[:200]
                    break
                local_tokens += resp.total_tokens
                answer = str(extract_json_field(resp.text, "answer")
                             or resp.text.strip())
                if not answer:
                    break
                missing = _missing_capitalized_words(task.prompt, answer)
                if not missing and _has_sane_entity_types(answer):
                    features["local_model"] = model
                    features["local_attempts"] = attempt + 1
                    self.cache.store(task.prompt, answer)
                    return Decision(task.id, task.prompt, "local_ner",
                                    answer, remote_tokens=0,
                                    local_tokens=local_tokens,
                                    latency_s=0.0, features=features)
                features[f"local_rejected_{model}_{attempt + 1}"] = answer[:150]
                if missing:
                    user = (task.prompt +
                            "\nYour previous answer MISSED these names "
                            f"from the text: {', '.join(missing)}. List "
                            "EVERY named entity including those, each with "
                            "its type (person, organization, location, "
                            "date, event, or product).")
                else:  # complete but untyped — relabel its own list
                    user = (task.prompt +
                            f"\nThese are the entities: {answer}\n"
                            "Answer again with these SAME names, labelling "
                            "each with its type, in the form: "
                            "Entity (person); Entity (organization); "
                            "Entity (location); Entity (date); ...")
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
        answer is a guaranteed zero, so it must never be final).

        Every attempt must first fit inside the run-wide remote token budget
        (when active): the reservation covers the whole worst case, so the
        run's billed Fireworks total can never exceed the cap. A refused
        call falls through to the free local last resort — over budget we
        risk an answer, never the rank."""
        remote_tokens = 0
        answer = ""
        for i, tier_model in enumerate(self._tier_order(category)):
            want = max_tokens or self.cfg.remote.max_tokens
            reservation = 0
            if self._budget.active:
                cap, floor = _BUDGET_COMPLETION.get(
                    category or "", _BUDGET_COMPLETION_DEFAULT)
                grant = self._budget.reserve(
                    len(system) + len(task.prompt), min(want, cap), floor)
                if grant is None:
                    features["budget_denied"] = True
                    break
                reservation, want = grant
            try:
                resp = self._remote.chat(
                    tier_model, system, task.prompt,
                    max_tokens=want,
                    temperature=self.cfg.remote.temperature)
            except Exception as exc:
                # A fast failure (404 model, refused connection) bills
                # nothing; a timeout MAY have generated server-side, so its
                # full reservation stays spent — the guarantee never leans
                # on an optimistic guess.
                billed_guess = (reservation
                                if "time" in str(exc).lower() else 0)
                if self._budget.active:
                    self._budget.settle(reservation, billed_guess)
                features[f"tier{i + 1}_error"] = str(exc)[:200]
                continue
            if self._budget.active:
                self._budget.settle(reservation, resp.total_tokens)
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
