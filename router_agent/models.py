"""Model clients: local (free tier) and remote (Fireworks, paid tier).

Both speak the OpenAI-compatible /v1/chat/completions protocol, so one client
class covers Ollama, vLLM (ROCm or CPU), and Fireworks — only base_url,
api_key, and model id differ. A mock backend lets the whole pipeline run
end-to-end with zero services, for CI/smoke tests and pre-kickoff dev.

Structured output: every call instructs the model to reply as JSON
{"answer": "..."} and requests json_object response format where the backend
supports it. This strips preamble/filler on every tier — the highest
ROI-per-effort optimization in the plan (03_ARCHITECTURE §7).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

ANSWER_SYSTEM_PROMPT = (
    # Terse by design: this prompt rides on EVERY short-answer call and
    # prompt tokens are ~72% of billed spend, so every word here is paid for
    # on the whole task set. Keep only the JSON contract + brevity order.
    'Reply ONLY as JSON {"answer":"<answer>"}. Answer immediately, no '
    "reasoning or prose. Be as short as correctness allows — one word, name, "
    "or number; use digits; no units/symbols unless the task asks."
)

REASONING_ANSWER_SYSTEM_PROMPT = (
    'Solve the math/logic problem. Reply ONLY as JSON {"work":"<brief '
    'steps>","answer":"<final value>"}. Keep work tight — watch units, order '
    "of operations, off-by-one; no restating the problem. In answer put ONLY "
    "the short final number or word (no units/symbols unless asked)."
)

CRITIC_SYSTEM_PROMPT = (
    "You are a strict answer judge. Given a task and two candidate answers, "
    "decide which is correct. For math or logic problems, verify the computation. "
    "Reply ONLY with a JSON object of the form "
    '{"verdict": "A" | "B" | "both" | "neither", "confidence": "high" | "low"}. '
    "No explanation."
)

VERIFY_SYSTEM_PROMPT = (
    "You verify and correct draft answers. Given a task and a draft answer, "
    "for any calculation, sequence, or logic problem, verify step by step. "
    "Reply ONLY with a JSON object: "
    '{"correct": true, "answer": "<the draft>"} if the draft is correct, or '
    '{"correct": false, "answer": "<the corrected answer>"} if not. '
    "Keep the answer as short as correctness allows. No explanation."
)

# --- Track 1 category prompts ----------------------------------------------
# The participant guide fixes eight capability categories graded by an
# LLM-Judge on *intent*, not exact-match — so long-form categories must
# produce genuinely complete answers (real code, real summaries), while
# short-form ones stay terse to keep scored remote tokens down.

SENTIMENT_SYSTEM_PROMPT = (
    'Classify sentiment. Reply ONLY as JSON {"answer":"<label>"} using the '
    "task's label set, else positive/negative/neutral/mixed. Label only, no "
    "justification."
)

SUMMARY_SYSTEM_PROMPT = (
    "Summarize the passage faithfully, obeying any length/format limit "
    "EXACTLY (a word limit is hard; one sentence means exactly one). Add "
    'nothing not in the passage. Reply ONLY as JSON {"answer":"<summary>"}. '
    "No markdown."
)

NER_SYSTEM_PROMPT = (
    "Extract EVERY named entity with its type (person, organization, "
    "location, date, plus any type the task names). Follow the task's output "
    'format if given, else "Entity (type); Entity (type)". Reply ONLY as JSON '
    '{"answer":"<entities>"}. No duplicates, no markdown.'
)

CODEGEN_SYSTEM_PROMPT = (
    "Write correct, complete code meeting the spec exactly — match the "
    "requested names, signatures, and language (default Python). Handle edge "
    'cases. Reply ONLY as JSON {"answer":"<code>"} with the code a plain JSON '
    "string using \\n newlines. No markdown fences, no prose."
)

CODEDEBUG_FIX_SYSTEM_PROMPT = (
    # Local-tier variant: small models asked to "identify and fix" drift into
    # prose ABOUT the bug and never emit the fix. Forcing code-only output is
    # what makes the mechanical (compile + behavioral-agreement) guard
    # possible at all — probe-verified both bundled locals then produce
    # correct, executable fixes.
    "You fix buggy code. Reply ONLY with a JSON object "
    '{"answer": "<the FULL corrected code>"} — the corrected code itself, '
    "as a plain JSON string with \\n newlines. No explanation, no markdown."
)

CODEDEBUG_SYSTEM_PROMPT = (
    # No prose bug description: the judge grades the corrected code and the
    # remote guard only needs compilable code, so a description just burns
    # completion tokens.
    "Fix the bug(s) in the given code. Reply ONLY as JSON "
    '{"answer":"<full corrected code>"} — the corrected code as a plain JSON '
    "string with \\n newlines. No description, no markdown, no prose."
)


@dataclass
class ModelResponse:
    text: str                # the extracted answer text
    raw_text: str            # full raw completion
    prompt_tokens: int
    completion_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def extract_json_field(raw: str, *fields: str):
    """Pull field(s) from a possibly-messy JSON completion.

    Falls back to using the raw text as the answer when the model failed to
    emit valid JSON — never crash the pipeline over formatting.
    """
    text = raw.strip()
    # reasoning models (minimax-m3 class) may prepend a <think>...</think>
    # block whose braces would confuse the JSON-span regex — drop it first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    # last flat JSON object in the text (reasoning prose often precedes it)
    tail = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if tail and tail[-1] not in candidates:
        candidates.append(tail[-1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                if len(fields) == 1:
                    return obj.get(fields[0])
                return tuple(obj.get(f) for f in fields)
        except json.JSONDecodeError:
            continue
    return None if len(fields) == 1 else tuple(None for _ in fields)


class OpenAICompatibleClient:
    """Chat client for any OpenAI-compatible endpoint (Ollama, vLLM, Fireworks)."""

    def __init__(self, base_url: str, api_key: str,
                 timeout: float | None = None):
        from openai import OpenAI  # imported lazily so mock mode needs no SDK
        # In harness mode config sets this to 40s local / 25s remote; local
        # CPU dev leaves it generous.
        # max_retries=0: the SDK's automatic retry REPEATS a timed-out request
        # — on the CPU-slow grading VM that silently doubles every slow call's
        # wall-clock cost, which is runtime-budget poison. The router already
        # has its own recovery (tier escalation, local fallback), so a failed
        # call should surface immediately, not be retried blind.
        self._client = OpenAI(base_url=base_url, api_key=api_key or "none",
                              timeout=timeout or 300.0, max_retries=0)

    def chat(self, model: str, system: str, user: str,
             max_tokens: int = 512, temperature: float = 0.0,
             timeout: float | None = None) -> ModelResponse:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if timeout is not None:
            # per-call override: under wall-clock pressure the router shrinks
            # this below the client default so one slow call can't eat the
            # whole run budget
            kwargs["timeout"] = timeout
        from openai import APIConnectionError, APITimeoutError
        try:
            resp = self._client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs)
        except (APITimeoutError, APIConnectionError):
            # A timeout/dead endpoint won't be cured by dropping
            # response_format — re-running it would double the wall-clock
            # cost of every slow call. Surface it; the router escalates.
            raise
        except Exception:
            # Some backends/models reject response_format — retry without it.
            resp = self._client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return ModelResponse(
            text=raw,
            raw_text=raw,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=model,
        )


class MockClient:
    """Deterministic offline stand-in so the pipeline runs with no services.

    Behavior: echoes a canned answer derived from the prompt. Prompts
    containing the word 'hard' make the two mock local models disagree, which
    exercises the critique + escalation paths in tests.
    """

    def __init__(self, persona: str = "generic"):
        self.persona = persona

    def chat(self, model: str, system: str, user: str,
             max_tokens: int = 512, temperature: float = 0.0,
             timeout: float | None = None) -> ModelResponse:
        lowered = user.lower()
        if "verdict" in system.lower():
            # low confidence on 'riddle' tasks -> exercises the remote path
            conf = "low" if "riddle" in lowered else "high"
            payload = {"verdict": "A", "confidence": conf}
        elif "draft answer" in lowered or "verify" in system.lower():
            payload = {"correct": True, "answer": _mock_answer(user, "remote")}
        else:
            variant = self.persona if "hard" in lowered else "agree"
            payload = {"answer": _mock_answer(user, variant)}
        raw = json.dumps(payload)
        tokens = max(1, len(user) // 4)
        return ModelResponse(text=raw, raw_text=raw,
                             prompt_tokens=tokens,
                             completion_tokens=max(1, len(raw) // 4),
                             model=f"mock/{model}")


def _mock_answer(prompt: str, variant: str) -> str:
    base = f"mock-answer-{abs(hash(prompt.strip().lower())) % 10**6}"
    if variant in ("agree", "remote"):
        return base
    # personas produce entirely unrelated strings so the fuzzy matcher
    # registers a genuine disagreement on 'hard' prompts
    return f"{variant}-{str(abs(hash(variant + prompt)) % 10**6)}"


def make_local_client(cfg) -> object:
    if cfg.backend == "mock":
        return None  # caller builds per-persona mocks
    return OpenAICompatibleClient(cfg.base_url, cfg.api_key)


def make_remote_client(cfg) -> object:
    if cfg.backend == "mock":
        return MockClient(persona="remote")
    key = cfg.api_key
    if not key:
        raise RuntimeError(
            f"Remote backend is '{cfg.backend}' but ${cfg.api_key_env} is not set. "
            "Set it in .env or the environment, or use backend: mock.")
    return OpenAICompatibleClient(cfg.base_url, key)
