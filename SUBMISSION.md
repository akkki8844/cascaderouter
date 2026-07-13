# lablab.ai Submission Kit — Track 1

Copy-paste content for the lablab.ai submission form. Team: **Veritas**
(AMD pod team id: team-3195).

---

## Project Title

**CascadeRouter — Hybrid Token-Efficient Routing Agent**

## Short Description (one-liner)

A routing agent that answers every task with the cheapest source that can be mechanically verified — free local models behind compile checks, dual-model agreement, and completeness guards, escalating to the measured-cheapest strong remote model the moment a guard fails — validated live at 89.5% with 9 of 19 tasks answered for zero tokens.

## Long Description

CascadeRouter is a Track 1 agent built on a principle we earned across three
real submissions: **small free models can't be trusted — but they don't need
to be trusted, they need to be checked.**

Our v3 answered ~46% of tasks with free bundled 1B-class local models on
trust (two models agreeing, any category) and scored 57.9%. Our v4 swung to
all-remote and scored 19/19 on our grading-style validation set — at 8,618
tokens. v7 is the synthesis: every task is first offered to the free local
models, but their answer only counts if it passes a **hard mechanical guard**
specific to the category's known failure mode:

- **Code generation** → local answer accepted only if it *compiles* and
  defines the exact function the task names. A 1.5B model writes correct
  code for straightforward functions; the guard catches it when it doesn't.
- **Sentiment, factual, math & logic** → accepted only when two local models
  from *different training lineages* (Llama 3.2 1B + Qwen2.5 1.5B) agree on
  the short answer (numeric-aware matching, so `1989` agrees with
  `{"year": 1989}`).
- **NER** → accepted only if every capitalized phrase of the source text
  reappears in the answer — small models' NER failure mode is silently
  *dropping* entities, and a dropped entity always drops its capital letters.
- **Summarization** → accepted only if explicit word/sentence limits in the
  task are mechanically satisfied (one free local retry before going remote).
- **Code debugging** → never local: probed live, the local model describes
  the bug without emitting corrected code — no cheap guard can save that.

Every guard failure escalates to **one call to the measured-cheapest strong
remote model for that category** — we priced every category on every allowed
model on the real Fireworks API: `minimax-m3`'s terse deterministic JSON for
code and NER (~290–380 tokens/task vs the code-specialist's volatile 400–940
for judge-equivalent answers), `kimi-k2p7-code` for factual and math/logic
(it alone answered every trick question correctly), Gemma-first for sentiment
and summaries (Best Use of Gemma). Blank or failed calls escalate through the
remaining allowed tiers and end at the local models as a never-blank last
resort. `ALLOWED_MODELS` is read at runtime, never hardcoded; every tier
emits structured JSON; a within-run dedup cache answers repeated prompts once
(in-memory, exact-match; nothing precomputed or persisted).

The submission is a single self-contained ~4 GB linux/amd64 image: agent +
Ollama + both local models baked into the layers. It reads
`/input/tasks.json`, writes `/output/results.json`, honors all env vars, runs
tasks on a worker pool with 25 s per-request timeouts, and degrades gracefully
(any tier dies → next tier; a single bad task → empty answer, never a crashed
batch).

Validation on the real Fireworks API, on a 19-task set mirroring the grading
distribution: **17/19 correct (89.5%), 2,646 remote tokens, 50 s wall time —
9 of 19 tasks answered for zero tokens**, a 69% token cut from our all-remote
v4 (19/19, 8,618) for two proxy-set misses (receipts committed in
`eval_results/hard_v5_*`; the v4/v5 receipts remain for the before/after).

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
   floor. Free 1B models save every token but scored us 57.9%. Strong remote
   models scored 19/19 but cost 8,618 tokens. The answer isn't picking a
   side — it's *verification*: 89.5% at 2,646 tokens, half the tasks free."
2. **(0:20–0:50) The idea** — show the routing diagram (README): every task
   goes to the free local models first, but their answer only counts if it
   passes a mechanical guard — generated code must compile and define the
   requested function, NER must preserve every capitalized phrase, summaries
   must meet the stated word limit, short answers need two model lineages to
   agree. "We don't trust small models. We check them."
3. **(0:50–1:20) Demo** — run the harness on the 19-task eval set, show
   `results.json` appearing in ~50 seconds and the per-task decision log:
   9 tasks at zero tokens, and every guard escalation paying for exactly one
   measured-cheapest remote call.
4. **(1:20–1:45) Receipts** — 89.5% at 2,646 tokens (69% below our all-remote
   build); the guard catching the local model dropping a named entity and
   escalating; the code-debug probe that kept debugging remote. Engineering
   by measurement.
5. **(1:45–2:00) Close** — Gemma on summaries via Fireworks, fully
   self-contained image, all Participant Guide budgets honored. "Free when
   verifiable. Cheap when not. Never blank."

## Slide deck outline (7 slides)

1. Title — CascadeRouter, team, track.
2. The scoring game — fewest tokens wins above an accuracy floor; our 57.9%
   lesson (trusted free models) and our 8,618-token lesson (all-remote
   overkill).
3. Architecture — the routing diagram (from README): local-first behind
   mechanical guards, measured-cheapest remote on escalation.
4. The guards — compile + function-name for code, dual-lineage agreement for
   short answers, capitalized-phrase completeness for NER, word-limit checks
   for summaries; debugging never local (probed: no guard exists).
5. Measurement over vibes — per-category token pricing on the real API; the
   code specialist that bills 2–3× the reasoning model for judge-equivalent
   answers; the math vote that never changed an answer.
6. Results — 17/19 (89.5%) · 2,646 tokens · 9/19 tasks free · 50 s (69%
   cheaper than the all-remote 19/19 build).
7. Compliance — self-contained amd64 image, env contract, budgets, Gemma via
   Fireworks on summarization.
