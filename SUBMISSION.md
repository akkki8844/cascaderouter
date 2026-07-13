# lablab.ai Submission Kit — Track 1

Copy-paste content for the lablab.ai submission form. Team: **Veritas**
(AMD pod team id: team-3195).

---

## Project Title

**CascadeRouter — Hybrid Token-Efficient Routing Agent**

## Short Description (one-liner)

A routing agent that answers every task with the cheapest source that can be mechanically verified — free local models behind compile checks, dual-model agreement, behavioral cross-execution of code fixes, and completeness guards with targeted free retries — plus TWO hard run-wide ceilings: a token budget that makes billing more than 480 remote tokens IMPOSSIBLE on any task set, and a wall-clock deadline governor that makes finishing inside the 10-minute container limit equally guaranteed. Validated live at 89.5% with 17 of 19 tasks answered for ZERO tokens (347 total remote tokens).

## Long Description

CascadeRouter is a Track 1 agent built on a principle we earned across three
real submissions: **small free models can't be trusted — but they don't need
to be trusted, they need to be checked.**

Our v3 answered ~46% of tasks with free bundled 1B-class local models on
trust (two models agreeing, any category) and scored 57.9%. Our v4 swung to
all-remote and scored 19/19 on our grading-style validation set — at 8,618
tokens. v8 is the synthesis: every task is first offered to the free local
models, but their answer only counts if it passes a **hard mechanical guard**
specific to the category's known failure mode — and a rejected attempt gets
*targeted free retries* that tell the model exactly what it got wrong:

- **Code debugging** → the strongest guard in the router: BOTH local
  lineages produce forced-code-only fixes, accepted only when each compiles,
  each actually changes the buggy code, and the two independently written
  fixes **agree behaviorally when executed side by side** on a probe battery
  (sandboxed subprocess, hard timeout). Two independent models converging on
  the same input/output behavior is far stronger evidence than either
  model's word.
- **Code generation** → local answer accepted only if it *compiles* and
  defines the exact function the task names.
- **Sentiment & factual** → accepted when two local models from *different
  training lineages* (Llama 3.2 1B + Qwen2.5 1.5B) agree on the short
  answer (numeric-aware matching, so `1989` agrees with `{"year": 1989}`).
  A factual split escalates, because there the locals were measured wrong
  *together*.
- **Math & logic** → one call to the stronger local, distilled to the bare
  final answer. We measured (v8) that on every disagreement between the two
  locals the 1.5B was right — so the final answer always equaled the
  1.5B's, and the second model call bought no information while costing
  the largest slice of CPU time on the 2-vCPU grading VM.
- **NER** → accepted only if every capitalized phrase of the source text
  reappears in the answer with sane type labels; an incomplete attempt is
  retried with the exact words it missed, an untyped one with its own list
  and an order to label it.
- **Summarization** → accepted only if explicit word/sentence limits are
  mechanically satisfied; retries state a harder limit, then a rewrite pass
  shortens the model's own previous attempt.

On top of the guards sit **two hard run-wide ceilings**. First, a remote-
token budget: every remote call must first reserve its worst case
(over-counted prompt estimate plus the full completion cap, which
`max_tokens` hard-bounds) against a thread-safe run-wide ceiling of 480
billed Fireworks tokens. A call that doesn't fit is never made — the task
takes the free local answer instead. The number the leaderboard ranks by is
therefore **bounded by construction, no matter what the hidden task set
looks like**; a worst-case drill with the budget forced to 1 token
completed the full run at 78.9% and exactly 0 remote tokens.

Second, a **wall-clock deadline governor**: the grading VM (2 vCPU / 4 GB)
kills containers at 10 minutes, and local 1B inference there runs ~10×
slower than on a GPU dev box — we learned this from a real resubmission
that TIMED OUT. The run is now paced against a hard deadline: tasks execute
factual-first then descending local-cost (budget-critical escalations
reserve first; any degraded tail is the cheapest-remote tasks), the router
steps down from full guards → no retry ladders → one fast remote call as
time-per-remaining-task shrinks, `results.json` is atomically rewritten
after EVERY task, and a watchdog flushes and exits cleanly before the kill.
A drill with the wall clock forced to 55 seconds still produced a complete,
valid, budget-respecting results file — a partial score always beats
TIMEOUT. Ollama is pinned to keep both bundled models resident and decode
one request at a time (parallel decode on 2 cores only adds timeout risk),
with both models warmed at startup.

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
`/input/tasks.json`, writes `/output/results.json` after every task, honors
all env vars, bounds every request with deadline-aware timeouts, and degrades
gracefully (any tier dies → next tier; a single bad task → empty answer,
never a crashed batch).

Validation on the real Fireworks API, on a 19-task set mirroring the grading
distribution: **17/19 correct (89.5%), 347 remote tokens — 17 of 19 tasks
answered for zero tokens** (earlier revisions reproduced at 348 and 375
tokens). That is a 96% token cut from our all-remote v4 (19/19, 8,618) for
two proxy-set misses, and the only remote spend left is the two factual
tasks both local models are *measurably wrong* about — every other category
resolves free behind its guard, the hard budget caps the total at 480 even
on a hostile task set, and the deadline governor caps total runtime even on
a hostile CPU (receipts committed in `eval_results/hard_v5_*`).

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
   side — it's *verification*: 89.5% at just 347 tokens, 17 of 19 tasks
   completely free — and a hard budget makes billing more than 480 tokens
   *impossible*, on any task set."
2. **(0:20–0:50) The idea** — show the routing diagram (README): every task
   goes to the free local models first, but their answer only counts if it
   passes a mechanical guard — generated code must compile, debug fixes from
   two model lineages must agree when *executed side by side*, NER must
   preserve every capitalized phrase, summaries must meet the stated word
   limit — and a rejected attempt gets a free retry that names exactly what
   it got wrong. "We don't trust small models. We check them."
3. **(0:50–1:20) Demo** — run the harness on the 19-task eval set, show
   `results.json` appearing task by task (written after every answer) and the per-task decision log:
   17 tasks at zero tokens; the only two remote calls are the two factual
   tasks the local models are measurably wrong about — and each call first
   reserved its worst case against the hard 480-token run budget.
4. **(1:20–1:45) Receipts** — 89.5% at 347 tokens (96% below our all-remote
   build); the behavioral cross-execution guard accepting two independent
   correct factorial fixes; the worst-case drill: budget forced to 1 token,
   the run still finishes at 78.9% and ZERO remote tokens. Engineering by
   measurement.
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
4. The guards — behavioral cross-execution of both lineages' debug fixes
   (sandboxed), compile + function-name for codegen, dual-lineage agreement
   for short answers, capitalized-phrase completeness + type sanity for NER,
   word-limit checks for summaries — plus targeted free retries that tell
   the model exactly what its last attempt got mechanically wrong.
5. Measurement over vibes — per-category token pricing on the real API; the
   code specialist that bills 2–3× the reasoning model for judge-equivalent
   answers; the math vote that never changed an answer; the split data
   showing the 1.5B wins every math/logic disagreement.
6. Results — 17/19 (89.5%) · 347 tokens · 17/19 tasks free (96%
   cheaper than the all-remote 19/19 build; only the two tasks the locals
   are measurably wrong about cost anything; a hard reserve-then-settle
   budget makes exceeding 480 billed tokens impossible on any task set).
7. Compliance — self-contained amd64 image, env contract, budgets, Gemma via
   Fireworks on summarization.
