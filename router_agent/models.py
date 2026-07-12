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
    "You are a precise task solver. For any calculation, sequence, or logic "
    "problem, think through it step by step before answering. Reply ONLY with "
    'a JSON object of the form {"answer": "<your answer>"}. No explanation, '
    "no markdown, no extra keys. Keep the answer as short as correctness allows: "
    "a single word, name, or number whenever possible. Use digits for numbers "
    "(e.g. 42, not forty-two) and do not add units or symbols unless the task "
    "asks for them."
)

REASONING_ANSWER_SYSTEM_PROMPT = (
    "You are a precise math and logic solver. Reply ONLY with a JSON object "
    'of the form {"work": "<brief step-by-step reasoning>", "answer": "<final '
    'answer>"}. Do the reasoning in the "work" field FIRST, computing carefully '
    "(watch unit conversions, e.g. distance/time-in-hours for speed, order of "
    "operations, off-by-one errors), then put ONLY the final short numeric or "
    'word answer in "answer" — no units or symbols unless asked for. No '
    "markdown, no extra keys."
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
    "You classify sentiment. Reply ONLY with a JSON object of the form "
    '{"answer": "<label>"}. Use exactly the label set the task asks for; '
    "if none is given, use positive, negative, neutral, or mixed. The "
    "answer is the bare label only — no justification, no markdown, no "
    "extra keys."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are a precise summarizer. Obey the length and format constraint in "
    "the task EXACTLY (e.g. one sentence means exactly one sentence; a word "
    "limit is a hard limit). Cover the passage's main point faithfully; do "
    "not add opinions or facts not in the passage. Reply ONLY with a JSON "
    'object of the form {"answer": "<the summary>"}. No markdown, no extra keys.'
)

NER_SYSTEM_PROMPT = (
    "You extract named entities. Find EVERY entity in the text and label its "
    "type (person, organization, location, date — plus any other type the "
    "task asks for). If the task specifies an output format, follow it; "
    'otherwise use "Entity (type); Entity (type); ...". Reply ONLY with a '
    'JSON object of the form {"answer": "<the labelled entities>"}. '
    "No duplicates, no markdown, no extra keys."
)

CODEGEN_SYSTEM_PROMPT = (
    "You are an expert programmer. Write correct, complete, well-structured "
    "code that satisfies the specification exactly — match the requested "
    "function/class names, signatures, and language (default to Python if "
    "unspecified). Handle edge cases. Reply ONLY with a JSON object of the "
    'form {"answer": "<the complete code>"} where the code is a plain JSON '
    "string with \\n newlines — no markdown fences, no prose before or "
    "after the code."
)

CODEDEBUG_SYSTEM_PROMPT = (
    "You are an expert code reviewer. Identify the bug(s) in the given code, "
    "then provide the corrected implementation. Reply ONLY with a JSON "
    'object of the form {"answer": "<one-sentence bug description, then the '
    'full corrected code>"} — the code as a plain JSON string with \\n '
    "newlines, no markdown fences."
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
        # In harness mode config sets this to 25s (under the competition's
        # 30s-per-request ceiling); local CPU dev leaves it generous.
        # max_retries=1 so a flaky call can't silently eat the runtime budget.
        self._client = OpenAI(base_url=base_url, api_key=api_key or "none",
                              timeout=timeout or 300.0, max_retries=1)

    def chat(self, model: str, system: str, user: str,
             max_tokens: int = 512, temperature: float = 0.0) -> ModelResponse:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            resp = self._client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs)
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
             max_tokens: int = 512, temperature: float = 0.0) -> ModelResponse:
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
