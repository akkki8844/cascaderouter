"""Free-tier confidence signals: answer agreement + heuristic difficulty.

Agreement between two independently-trained local models is the primary
confidence signal (03_ARCHITECTURE §3) — much harder to be 'confidently
wrong' on than a single model's self-report. All of this runs locally = $0.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize_answer(text: str) -> str:
    """Normalize for comparison: lowercase, collapse whitespace, strip
    punctuation at the edges, drop articles. Conservative on purpose —
    tighten/loosen once the real task domain (and grading style) is known."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?\"'`")
    text = re.sub(r"\b(a|an|the)\b\s*", "", text)
    return text.strip()


def answers_agree(a: str, b: str, fuzzy_threshold: float = 0.90) -> tuple[bool, float]:
    """Exact-after-normalization match, falling back to fuzzy similarity.

    Returns (agree, similarity). Fuzzy match catches trivially-different
    phrasings of the same short answer without a semantic model dependency.
    """
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return False, 0.0
    if na == nb:
        return True, 1.0
    sim = SequenceMatcher(None, na, nb).ratio()
    return sim >= fuzzy_threshold, sim


_COMPUTE_PATTERN = re.compile(
    r"\bcalculate\b|\bcompute\b|\bsequence\b|\bpercent\b|%|"
    r"\d+\s*(?:plus|minus|times|multiplied|divided|to the power)|"
    r"[+\-*/^]\s*\d|\d\s*[+\-*/^]|square root|remainder|"
    r"how many|what day|discount|speed|kilometers per hour|"
    r"brothers?|sisters?|siblings?|machines?|widgets?|"
    r"next number|double|half of",
    re.IGNORECASE,
)


def is_computational(prompt: str) -> bool:
    """True for arithmetic/sequence/logic tasks small local models get wrong
    even when they agree with each other (correlated arithmetic errors).

    These get force-routed to remote fresh-generation with self-consistency
    instead of trusting local agreement (03_ARCHITECTURE §3 revision after
    real-model eval showed local models confidently agreeing on wrong sums).
    """
    return bool(_COMPUTE_PATTERN.search(prompt))


# --- Track 1 capability-category classifier -------------------------------
# The participant guide fixes eight evaluation categories. Each maps to the
# cheapest route that still clears the accuracy gate (accuracy is a hard
# gate: below-threshold submissions are excluded from the leaderboard, so
# categories the 1B locals are weak at go remote even though it costs tokens).

_CODE_MARKER = re.compile(
    r"```|\bdef\s+\w+|\bclass\s+\w+|\bfunction\b|\breturn\b|\bpython\b|"
    r"\bjavascript\b|\bjava\b|\bc\+\+\b|\bcode\b|\bsnippet\b|\bscript\b|"
    r"\bimplement(?:ation)?\b|\balgorithm\b|\bregex\b|\bsql\b",
    re.IGNORECASE)
_DEBUG_MARKER = re.compile(
    r"\bbug(?:gy|s)?\b|\bfix\b|\bdebug\b|\berror\b|\bbroken\b|\bwrong\b|"
    r"\bincorrect\b|\bfails?\b|\bdoesn'?t work\b|\bnot work(?:ing)?\b|"
    r"\bunexpected\b|\bcrash",
    re.IGNORECASE)
_SUMMARY_MARKER = re.compile(
    r"\bsummar(?:y|ise|ize|isation|ization)\b|\btl;?dr\b|\bcondense\b|"
    r"\bin (?:one|1|two|2|a single) sentences?\b|\bin \d+ words\b|"
    r"\bshorten\b|\bparaphrase\b|\bmain (?:point|idea)s?\b",
    re.IGNORECASE)
_SENTIMENT_MARKER = re.compile(
    r"\bsentiment\b|\bpositive,? (?:or )?negative\b|"
    r"\bpositive or negative\b|\btone of\b|\bemotion(?:al)?\b|"
    r"\bopinion (?:expressed|of the)\b|\bhow does the (?:author|reviewer|writer) feel\b",
    re.IGNORECASE)
_NER_MARKER = re.compile(
    r"\bnamed entit(?:y|ies)\b|\bentit(?:y|ies)\b|"
    r"\bextract(?:\s+\w+){0,4}\s+(?:names?|people|persons?|organi[sz]ations?|"
    r"locations?|places|dates?|companies)\b|"
    r"\b(?:identify|list|label)(?:\s+\w+){0,4}\s+(?:names?|people|persons?|"
    r"organi[sz]ations?|locations?|places|dates?|companies)\b",
    re.IGNORECASE)
_LOGIC_MARKER = re.compile(
    r"\briddle\b|\bpuzzle\b|\bconstraints?\b|\bknights? and knaves?\b|"
    r"\bseated?\b|\bsits? (?:next to|between|left|right)\b|\btruth[- ]?teller\b|"
    r"\balways lies\b|\bif and only if\b|\bdeduce\b|\bwho (?:is|owns|has|am i)\b|"
    r"\btaller than\b|\bolder than\b|\byounger than\b|\bexactly one\b|"
    r"\bbrothers?\b|\bsisters?\b|\bsiblings?\b",
    re.IGNORECASE)


def classify_category(prompt: str) -> str:
    """Map a task to one of the eight Track 1 capability categories.

    Returns one of: code_debugging, code_generation, summarization, ner,
    sentiment, math, logic, factual. Order matters: sentiment/summarization
    are checked before code because everyday passages trip the code markers
    ("upgraded to first CLASS" tripped the class-keyword regex on the eval,
    h18), while sentiment/summary instruction verbs almost never appear in
    genuine code tasks. Code is still checked before math/logic because code
    snippets are full of arithmetic operators.
    """
    if _SENTIMENT_MARKER.search(prompt):
        return "sentiment"
    if _SUMMARY_MARKER.search(prompt):
        return "summarization"
    if _CODE_MARKER.search(prompt):
        if _DEBUG_MARKER.search(prompt):
            return "code_debugging"
        return "code_generation"
    if _NER_MARKER.search(prompt):
        return "ner"
    if _LOGIC_MARKER.search(prompt):
        return "logic"
    if is_computational(prompt):
        return "math"
    return "factual"


def majority_vote(answers: list[str]) -> str:
    """Most common normalized answer, breaking ties by first occurrence.

    Blank votes (a model truncated before emitting content -- e.g. a
    'reasoning'-channel model that ran out of budget) are ignored unless
    every vote is blank, so a single real answer beats several empties.
    """
    counts: dict[str, int] = {}
    first_seen: dict[str, str] = {}
    for a in answers:
        key = normalize_answer(a)
        if not key:
            continue
        if key not in first_seen:
            first_seen[key] = a
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return next((a for a in answers if a), "")
    best_key = max(counts, key=lambda k: counts[k])
    return first_seen[best_key]


def heuristic_difficulty(prompt: str, max_chars: int, hard_markers: list[str]) -> float:
    """Cheap static difficulty estimate in [0, 1] for the v2 heuristic router.

    0 = looks easy (route local), 1 = looks hard (route remote). Deliberately
    dumb — it exists as the fast Day-1 fallback, not the final design.
    """
    score = 0.0
    if len(prompt) > max_chars:
        score += 0.4
    lowered = prompt.lower()
    if any(marker in lowered for marker in hard_markers):
        score += 0.5
    # code/math markers tend to be harder for 1-2B models
    if re.search(r"[=+\-*/^]{2,}|```|\bdef |\bclass |\d+\s*[+\-*/]\s*\d+", prompt):
        score += 0.2
    return min(score, 1.0)
