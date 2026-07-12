# lablab.ai Submission Kit — Track 1

Copy-paste content for the lablab.ai submission form. Team: **Veritas**
(AMD pod team id: team-3195).

---

## Project Title

**CascadeRouter — Hybrid Token-Efficient Routing Agent**

## Short Description (one-liner)

A routing agent that classifies every task into its competition category and sends it to the allowed model best suited for it — cross-family self-consistency votes for math/logic, a code specialist for code, Gemma for sentiment and summaries — validated live at 19/19 correct on a grading-style task set.

## Long Description

CascadeRouter is a Track 1 agent built on one principle learned the hard way:
**accuracy is a gate, not a score — a token you saved on a wrong answer is worth
exactly nothing.**

Our first submission routed nearly half of all tasks to free bundled 1B-class
local models and scored 57.9%: below the 80% accuracy gate, so every saved token
was wasted. v4 is the redesign that follows from that measurement. Every task is
classified into one of the eight competition categories by a zero-cost
classifier, then answered by the *strongest allowed model for that category*:

- **Math & logic** → cross-family self-consistency: one vote each from the
  strongest general instruct model, the dedicated reasoning model, and Gemma 4 —
  independently trained families, so a systematic bias in one can't fake a
  majority. Early-stopped as soon as the first two families agree (2-of-2
  cross-family agreement is a stronger signal than any same-family majority,
  and ~40% cheaper).
- **Code generation & debugging** → one call to the code-specialist model
  (`kimi-k2p7-code`) with a category-tuned prompt.
- **Sentiment & summarization** → Gemma-first single call (Best Use of Gemma),
  with the stronger general models behind it if it fails.
- **Factual & NER** → one terse call to the strongest general model; short-form
  answers are a bare label/name/number, so completions cost single-digit tokens.
- **Never-blank guarantee** → reasoning-channel models can return HTTP 200 with
  empty content when truncated; an empty answer is a guaranteed zero. Every
  route escalates through the remaining allowed models on failure *or* blank
  output, ending at the bundled local models (Llama 3.2 1B + Qwen2.5 1.5B) as a
  last resort — a plausible local answer beats a certain-zero blank.

`ALLOWED_MODELS` is read at runtime, never hardcoded. Every tier emits
structured JSON — no preamble or filler tokens anywhere. A within-run dedup
cache answers repeated prompts once (in-memory, exact-match; nothing
precomputed or persisted).

The submission is a single self-contained ~4 GB linux/amd64 image: agent +
Ollama + both fallback models baked into the layers. It reads
`/input/tasks.json`, writes `/output/results.json`, honors all env vars, runs
tasks on a worker pool with 25 s per-request timeouts, and degrades gracefully
(any tier dies → next tier; a single bad task → empty answer, never a crashed
batch).

Validation on the real Fireworks API, on a 19-task set mirroring the grading
distribution: **19/19 correct, 8,822 remote tokens, 27 s wall time** — including
a live demonstration of the escalation ladder recovering through three
unavailable model tiers without losing a single answer (receipts committed in
`eval_results/hard_v4b_*`).

## Technology & Category Tags

`Gemma` · `Fireworks AI` · `AMD Developer Cloud` · `Ollama` · `Kimi K2` ·
`MiniMax` · `Python` · `Docker`

## Links (fill in)

- Public GitHub repository: `https://github.com/akkki8844/cascaderouter`
- Docker image: `ghcr.io/akkki8844/cascaderouter:latest`
- Application URL / demo platform: the Docker image is the application
  (harness-run); link the GHCR package page.

---

## Video Presentation script (~2 min)

1. **(0:00–0:20) Hook** — "Track 1 scores two things: accuracy first, tokens
   second. Our first submission optimized tokens and failed the accuracy gate
   at 57.9%. Here's what we rebuilt — and why it now scores 19 out of 19."
2. **(0:20–0:50) The idea** — show the routing diagram (README): 8-category
   classifier; the right model for each category; cross-family voting for
   math/logic. "Three models from different families that agree beat one
   model's confidence — and beat three copies of the same model every time."
3. **(0:50–1:20) Demo** — run the harness on the 19-task eval set, show
   `results.json` appearing in 27 seconds and the per-task decision log:
   category, model chosen, tokens per route.
4. **(1:20–1:45) Receipts** — 19/19 correct at 8,822 tokens; the live
   escalation log showing three unavailable tiers recovered without losing an
   answer; the classifier fix caught by evals ("first class" ≠ code).
   Engineering by measurement.
5. **(1:45–2:00) Close** — Gemma answering sentiment and summarization via
   Fireworks, fully self-contained image, all Participant Guide budgets
   honored. "Accuracy is the gate. Tokens are the score. CascadeRouter
   optimizes both, in that order."

## Slide deck outline (7 slides)

1. Title — CascadeRouter, team, track.
2. The scoring game — accuracy gate first, then fewest tokens; a saved token on
   a wrong answer is worth nothing (our 57.9% lesson).
3. Architecture — the routing diagram (from README): right model per category.
4. Cross-family self-consistency — why independent model families beat repeated
   samples of one family; early-stop at first cross-family agreement.
5. Reliability engineering — never-blank guarantee, escalation ladder proven
   live through three dead tiers, within-run dedup cache.
6. Results — 19/19 correct · 8,822 tokens · 27 s on a grading-style set.
7. Compliance — self-contained amd64 image, env contract, budgets, Gemma via
   Fireworks on sentiment/summarization.
