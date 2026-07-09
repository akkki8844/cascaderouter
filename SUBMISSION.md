# lablab.ai Submission Kit — Track 1

Copy-paste content for the lablab.ai submission form. Team: **Veritas**
(AMD pod team id: team-3195).

---

## Project Title

**CascadeRouter — Hybrid Token-Efficient Routing Agent**

## Short Description (one-liner)

A routing agent that answers tasks with two free local models that must agree, escalates math and logic to early-stopping self-consistency votes, and pays Fireworks tokens only when a cheaper path can't clear the accuracy gate — validated at 100% category accuracy.

## Long Description

CascadeRouter is a Track 1 agent built on one principle: **the cheapest token is the one you never send.**

Every task is first classified into one of the eight competition categories with a
zero-cost classifier, then routed down the cheapest path that still clears the
accuracy gate:

- **Factual & sentiment** → two bundled 1B-class local models (Llama 3.2 1B +
  Qwen2.5 1.5B, different training lineages) answer independently. If they agree,
  that's the answer — **zero Fireworks tokens**. Agreement between unrelated models
  is a far stronger free confidence signal than any single model's self-rating.
- **Math & logic** → straight to remote self-consistency: up to 4 reasoning votes
  through the cheapest allowed Fireworks model, early-stopped as soon as two agree
  (−37% tokens at unchanged accuracy in our evals). Small local models fail
  arithmetic in *correlated* ways, so local agreement is deliberately skipped here.
- **Code generation, code debugging, summarization, NER** → one remote call with a
  category-tuned prompt. NER lives here because it is graded on *completeness*,
  which a local critic cannot verify — we watched a 1B model drop "Tim Cook" from
  an extraction and an 8B critic confidently approve it. Measured, not guessed.
- **Escalation is verify-not-regenerate**: when locals disagree, the remote model
  is sent the local draft and asked to confirm or fix it — confirm/fix responses
  are far shorter than regeneration, so escalations cost fewer scored tokens.

`ALLOWED_MODELS` is read at runtime and auto-ordered **Gemma-first, largest-first**,
so every escalation lands on Gemma 4 31B IT via Fireworks (Best Use of Gemma).
Every tier emits structured JSON — no filler tokens anywhere. The escalation
threshold is calibrated against our own decision logs, not hand-tuned.

The submission is a single self-contained ~4 GB linux/amd64 image: agent + Ollama +
both local models baked into the layers. It reads `/input/tasks.json`, writes
`/output/results.json`, honors all env vars, runs tasks on a worker pool with 25 s
per-request timeouts, and degrades gracefully (local tier dies → remote-only;
a single bad task → empty answer, never a crashed batch).

Validation on real models: **8/8 (100%) across all eight categories at 2,597
remote tokens** on our category smoke set, and 100% / 16,730 tokens / 23 of 50
tasks at $0 on our 50-task proxy set.

## Technology & Category Tags

`Gemma` · `Fireworks AI` · `AMD Developer Cloud` · `Ollama` · `Llama 3.2` ·
`Qwen` · `Python` · `Docker`

## Links (fill in)

- Public GitHub repository: `https://github.com/akkki8844/cascaderouter`
- Docker image: `ghcr.io/akkki8844/cascaderouter:latest`
- Application URL / demo platform: the Docker image is the application
  (harness-run); link the GHCR package page.

---

## Video Presentation script (~2 min)

1. **(0:00–0:20) Hook** — "Track 1 scores two things: accuracy first, tokens
   second. Most agents pay for every answer. Ours refuses to pay unless it has to."
2. **(0:20–0:50) The idea** — show the routing diagram: 8-category classifier;
   free dual-local agreement; self-consistency for math/logic; category-tuned
   single calls for code/summarization/NER. "Two tiny models that agree are
   worth more than one big model's confidence."
3. **(0:50–1:20) Demo** — run `docker compose run agent` on the sample input,
   show `results.json` appearing and the per-task decision log: which tasks were
   free, which escalated, token counts per route.
4. **(1:20–1:45) Receipts** — eval screenshots: 8/8 categories, 100% on the
   50-task proxy set, the NER failure case that made us re-route it (engineering
   by measurement).
5. **(1:45–2:00) Close** — Gemma-first ordering via Fireworks, fully
   self-contained image, all Participant Guide budgets honored. "Accuracy is the
   gate. Tokens are the score. CascadeRouter optimizes both, in that order."

## Slide deck outline (7 slides)

1. Title — CascadeRouter, team, track.
2. The scoring game — accuracy gate first, then fewest tokens; local = $0.
3. Architecture — the routing diagram (from README).
4. Why dual-local agreement works (and where it doesn't — correlated math errors,
   NER completeness).
5. Token-saving techniques — verify-not-regenerate, early-stop voting, JSON-only
   output, calibrated threshold.
6. Results — 8/8 categories · 100% proxy accuracy · 23/50 tasks free.
7. Compliance — self-contained amd64 image, env contract, budgets, Gemma-first.
