# lablab.ai Submission Kit — Track 1

Copy-paste content for the lablab.ai submission form. Team: **Veritas**
(AMD pod team id: team-3195).

---

## Project Title

**CascadeRouter — Hybrid Token-Efficient Routing Agent**

## Short Description (one-liner)

A routing agent that classifies every task into its competition category and sends it to the measured-cheapest allowed model that answers it correctly — one call per task, free local models where they're provably safe — validated live at 19/19 correct and 4,603 remote tokens on a grading-style task set.

## Long Description

CascadeRouter is a Track 1 agent built on two measurements, not vibes:
**accuracy below the floor is worth nothing, and every token above the
cheapest-correct call is worth less than nothing.**

Our first submission routed nearly half of all tasks to free bundled 1B-class
local models and scored 57.9% — the free tier can't be trusted with graded
answers. Our second rebuilt every category around the strongest remote models
and scored 19/19 on our grading-style validation set, but at 8,618 tokens.
v5 is the synthesis: we measured, per category and per model on the real
Fireworks API, exactly what a correct answer costs — then kept only the
cheapest correct option:

- **Math & logic** → ONE call to the strongest general model with a
  reasoning-tuned prompt. v4's cross-family confirmation vote never changed
  the first model's answer in any validated run, so it was pure token cost:
  dropping it halved math tokens with identical answers (the general model
  alone answered every trick question correctly — the dedicated reasoning
  model measurably didn't).
- **Code generation, debugging & NER** → `minimax-m3` first: its terse,
  deterministic JSON completions cost ~290–380 tokens/task where
  `kimi-k2p7-code` bills a volatile 400–940 for answers an LLM judge grades
  identically. Kimi is the next tier if minimax fails or returns blank.
- **Sentiment** → the two bundled 1B local models (different training
  lineages) answer for **zero tokens**, accepted only when both agree on the
  bare polarity label — the one output small models can genuinely
  cross-check. Any disagreement or blank goes remote.
- **Summarization** → Gemma-first single call (Best Use of Gemma), with the
  stronger general models behind it if it fails.
- **Factual** → one terse call to the strongest general model; short-form
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
distribution: **19/19 correct, 4,603 remote tokens, 28 s wall time** — a 46%
token cut over the accuracy-first v4 with zero lost answers, plus a live
demonstration of the escalation ladder recovering through unavailable model
tiers (receipts committed in `eval_results/hard_v5_*`; v4 receipts in
`eval_results/hard_v4c_*` for the before/after).

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

1. **(0:00–0:20) Hook** — "Track 1 ranks by fewest tokens above an accuracy
   floor. Our first submission saved tokens and scored 57.9%. Our second
   scored 19 out of 19 — at twice the necessary cost. Here's how we measured
   our way to both: 19/19 at 4,603 tokens."
2. **(0:20–0:50) The idea** — show the routing diagram (README): 8-category
   classifier; the measured-cheapest correct model per category. "We priced
   every category on every allowed model. The 'code specialist' bills 2–3× the
   tokens of the reasoning model for answers the judge grades identically —
   you only learn that by measuring."
3. **(0:50–1:20) Demo** — run the harness on the 19-task eval set, show
   `results.json` appearing in 28 seconds and the per-task decision log:
   category, model chosen, tokens per route — including sentiment answered by
   the bundled local pair for zero tokens.
4. **(1:20–1:45) Receipts** — 19/19 correct at 4,603 tokens, down 46% from our
   accuracy-first rebuild with zero lost answers; the live escalation log
   recovering through unavailable tiers; the confirmation vote we deleted
   because it never changed an answer. Engineering by measurement.
5. **(1:45–2:00) Close** — Gemma answering summarization via Fireworks, fully
   self-contained image, all Participant Guide budgets honored. "Accuracy is
   the floor. Tokens are the score. CascadeRouter measures both."

## Slide deck outline (7 slides)

1. Title — CascadeRouter, team, track.
2. The scoring game — fewest tokens wins above an accuracy floor; our 57.9%
   lesson (free models fail the floor) and our 8,618-token lesson (unmeasured
   redundancy doubles the bill).
3. Architecture — the routing diagram (from README): measured-cheapest correct
   model per category.
4. Measurement over vibes — per-category token pricing on the real API; the
   code specialist that bills 2–3× the reasoning model for judge-equivalent
   answers; the math vote that never changed an answer.
5. Reliability engineering — never-blank guarantee, escalation ladder proven
   live through dead tiers, dual-local agreement gate on sentiment, within-run
   dedup cache.
6. Results — 19/19 correct · 4,603 tokens · 28 s on a grading-style set (46%
   cheaper than our accuracy-first v4, same answers).
7. Compliance — self-contained amd64 image, env contract, budgets, Gemma via
   Fireworks on summarization.
