# Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II — Track 1** · Team: **Veritas**

An AI agent that answers every task with the **cheapest source that can be mechanically trusted** — free bundled local models behind hard acceptance guards (dual-model agreement, behavioral cross-execution of code fixes, compile checks, completeness checks, length-constraint checks) with targeted free retries, escalating to the measured-cheapest strong remote model only where the guards prove the free answer can't be trusted. The leaderboard ranks by fewest tokens above an accuracy floor; the router spends remote tokens only on the tasks the local models are *measurably wrong* about.

**Internal validation on real Fireworks models (19-task set mirroring the grading distribution, no mocks): 17/19 correct (89.5%), 390 remote tokens, 76 s wall time — 17 of 19 tasks answered for zero tokens** (`eval_results/hard_v5_decisions.jsonl`; reproduced twice at 375/390 tokens). See `AMDPLAN/IMPLEMENTATION.md` for the full build record and `AMDPLAN/RUN.md` for a copy-paste run guide.

## The submission contract (Participant Guide compliance)

The submitted artifact is a **single self-contained Docker image** (agent + Ollama + both local models baked in) that:

1. reads `/input/tasks.json` (`[{"task_id": ..., "prompt": ...}, ...]`) on startup,
2. answers every task via the cascade below,
3. writes `/output/results.json` (`[{"task_id": ..., "answer": ...}, ...]`) and exits 0.

The judging harness injects — and the container reads at runtime, never hardcodes:

| Env var | Use |
|---|---|
| `FIREWORKS_API_KEY` | key for ALL remote calls |
| `FIREWORKS_BASE_URL` | ALL remote calls go through this URL (bypassing it = zero recorded tokens) |
| `ALLOWED_MODELS` | the only permitted model ids; we auto-order them **Gemma-first** (bonus prize), largest first. Official Track 1 list: `gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it-nvfp4`, `minimax-m3`, `kimi-k2p7-code` |

Budgets honored: container ready < 60 s (Ollama starts in ~2 s, models are pre-baked), < 30 s per request (25 s client timeout), < 10 min total (tasks run on a worker pool sized for the published 2 vCPU / 4 GB grading environment), image ≪ 10 GB (~4 GB — also under the organizers' ~5 GB pull-reliability guidance), `linux/amd64` build. No answer caching across runs: the within-run dedup cache runs in-memory and exact-match only. A failing local tier degrades to remote-only instead of failing the run; a failing single task yields an empty answer instead of a crashed container.

## How it works

Each task is first classified into one of the eight Track 1 capability categories, then offered to the **free bundled local models — but their answer only counts if it passes a hard mechanical guard** specific to the category, with targeted *free* retries that tell the model exactly what its previous attempt got mechanically wrong. Guard failures escalate to the measured-cheapest strong remote model for that category, with further tier escalation on failure or blank output. Placement is fewest tokens above an accuracy floor, so every routing choice below is backed by per-category measurements on the real Fireworks API, not vibes.

```
task ──▶ within-run dedup cache ──hit──▶ answer (0 tokens)
              │ miss
              ▼
      category classifier (8 Track 1 categories)
              │
    ┌──────────────┬───────────────┬───────────────┬───────────────┐
    │              │               │               │               │
sentiment /     code gen       summarization      NER          code debug
factual /          │               │               │               │
math / logic       ▼               ▼               ▼               ▼
    │         local qwen;     local qwen;     both locals;    BOTH locals
    ▼         accept ONLY     accept ONLY if  accept ONLY if  emit forced-
dual-local    if it compiles  explicit word/  every capital-  code fixes;
agreement     + defines the   sentence limit  ized source     accept ONLY if
(2 lineages   requested       is met; free    word appears +  both compile,
must agree;   function        retries state   sane type       change the bug,
numeric-                      a HARDER limit, labels; free    and AGREE when
aware)                        then shorten    retries name    EXECUTED side
    │                         the previous    the exact       by side on a
    │                         attempt         missed words    probe battery
    │                                                         (sandboxed)
    ├─ math/logic split: the stronger local's distilled answer
    │  stands (probe-verified it wins every measured split) — free
    │ guard fails / factual split / blank / error
    ▼
ONE call to the measured-cheapest strong remote model per category:
kimi (factual, math/logic w/ reasoning prompt) · minimax-m3 (code, NER:
terse JSON, ~1/2–1/3 kimi's tokens) · Gemma-first (sentiment, summaries)
              │ (any tier fails or returns blank)
              ▼
   next allowed model … ▶ local models (last resort — never return blank)
```

Key techniques (see `AMDPLAN/03_ARCHITECTURE.md` for the full design rationale):

- **Free answers only behind mechanical guards** — the v3 lesson (57.9% real score) is that 1B local models can't be *trusted*; the v7/v8 insight is they don't need to be — they need to be *checked*. Dual-lineage agreement for short answers, `compile()` + function-name checks for generated code, capitalized-phrase completeness + type-sanity for NER, word/sentence-limit checks for summaries, and for code debugging the strongest guard in the router: both lineages produce forced-code fixes that are accepted only when they compile, actually change the buggy code, and **agree behaviorally when executed side by side** on a probe battery in a sandboxed, hard-timeout subprocess. Every guard failure escalates; a local miss costs latency, never an answer. Result: 17 of 19 validation tasks answered for zero tokens with no measured accuracy loss.
- **Targeted free retries** — a guard doesn't just reject, it says *why*, and the retry feeds that back: an incomplete NER answer is retried with the exact capitalized words it missed; a complete-but-untyped one is retried with its own list and an order to label it; an over-long summary first gets a HARDER stated limit, then a rewrite pass shortening its own previous attempt (compression is far easier for a small model than constrained generation). Local retries score zero tokens.
- **Category-aware, cost-measured remote selection** — when a guard does escalate, the call goes to the allowed model that delivers a correct answer for the fewest measured tokens: `minimax-m3`'s terse deterministic JSON for code and NER (~290–380 tokens/task vs kimi's volatile 400–940 for judge-equivalent answers), `kimi-k2p7-code` for factual and math/logic (it alone answered every trick question correctly), Gemma-4 checkpoints leading sentiment and summarization.
- **Splits identify the winner** — on a math/logic disagreement the stronger local's distilled answer stands for free (probe-verified it won every measured split), while a factual split still escalates (there both locals were measured wrong *together*). The only remote tokens in the final validation run are the two factual tasks the locals provably can't answer.
- **Single-call math/logic** — v4's cross-family confirmation vote never changed the first model's answer in any validated run, so it's gone: one strong-model call with a reasoning-tuned prompt, tier escalation only on blank.
- **Never-blank guarantee** — reasoning-channel models can return HTTP 200 with empty content when truncated; an empty answer is a guaranteed zero. Every route escalates through the remaining allowed models on failure *or* blank output, ending at the bundled local models as a last resort.
- **Structured JSON output on every tier** — no preamble or filler tokens, ever; short-form categories answer with a bare label/number/name.
- **Within-run dedup cache** — repeated prompts inside one grading run are answered once (exact-match, in-memory only; nothing is precomputed or persisted).
- **Lesson learned the hard way** — v3 answered ~46% of tasks with free 1B-class local models on *trust* (agreement alone, any category) and scored 57.9% on the real grading run. v8 re-earns the free tier with *verification*: local answers count only when a mechanical guard proves the failure modes we know about aren't present. Even code debugging — prose-prone small models' worst category — went free once we found its guard: force code-only output and cross-execute both lineages' fixes.

## Build & push the submission image

**Canonical path — GitHub Actions (no local Docker needed):** every push to `main`
runs `.github/workflows/docker.yml`, which builds the image for `linux/amd64` and
pushes it to `ghcr.io/<owner>/<repo>:latest`. Set the GHCR package to **public**
before submitting (repo → Packages → package settings → Change visibility).

Manual alternative — the judging VM runs `linux/amd64`; build for it explicitly (required on Apple Silicon, harmless elsewhere):

```bash
docker buildx build --platform linux/amd64 -t <registry>/<team>-routing-agent:latest --push .
```

The build bakes `llama3.2:1b` and `qwen2.5:1.5b` into the image, so the judging VM pulls nothing at runtime.

### Test the image exactly like the harness will

```bash
mkdir -p harness_input harness_output
cp eval/sample_input.json harness_input/tasks.json
cp .env.example .env      # fill in FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS
docker compose run --rm agent
cat harness_output/results.json
```

Without Docker (dev machine, Ollama running locally):

```bash
pip install -r requirements.txt
TASKS_INPUT=eval/sample_input.json RESULTS_OUTPUT=out.json python harness_main.py
```

## Local development & evaluation

Development runs entirely on local Ollama — the "remote" tier is simulated by stronger local models (`gemma4:latest`, staying in the Gemma family like the real remote tier) so routing logic can be developed and measured without spending API credits:

```bash
ollama pull llama3.2:1b qwen2.5:1.5b llama3:latest gemma4:latest gemma:latest
python -m eval.run_eval --config config.yaml --tasks eval/tasks_real.jsonl --strategy cascade
python -m eval.run_eval --tasks eval/tasks_categories.jsonl   # 8-category smoke set
python -m eval.run_eval --strategy always_remote              # baseline comparison
```

The eval prints per-task route/token/correctness lines plus the three metrics that matter — accuracy, remote tokens, free-resolution rate — and saves JSON + Markdown reports to `eval_results/`. Fit the escalation threshold with `python -m eval.calibrate --accuracy-floor 0.9`.

**Single task / batch (dev CLI):**
```bash
python main.py answer --prompt "What is the capital of France?"
python main.py batch --input tasks.jsonl --output answers.jsonl
python main.py --strategy always_remote answer --prompt "..."   # baselines: always_local | heuristic | cascade
```

## Project layout

```
harness_main.py             SUBMISSION entrypoint (/input/tasks.json -> /output/results.json)
entrypoint.sh               container boot: start bundled Ollama, then harness_main
main.py                     dev CLI (answer / batch)
config.yaml                 all models, thresholds, strategies — nothing hardcoded
router_agent/
  task.py                   input adapter (accepts harness task_id + dev id)
  models.py                 OpenAI-compatible client (Ollama/vLLM/Fireworks) + category prompts + mock backend
  confidence.py             agreement signals + 8-category classifier
  cache.py                  within-run dedup cache (in-memory, exact-match in harness mode)
  router.py                 the tiered calibrated cascade (all 4 strategies)
  decision_log.py           JSONL decision logging (calibration training data)
eval/
  run_eval.py               leaderboard-proxy harness
  calibrate.py              threshold fitting (constrained optimization)
  tasks_real.jsonl          50-task labeled proxy set
  tasks_categories.jsonl    8-task smoke set, one per competition category
  sample_input.json         harness-format sample input
Dockerfile, docker-compose.yml
```

## Submission compliance checklist

- ✅ **Single self-contained Docker image** — Ollama + local models baked in; `linux/amd64`; ~4 GB ≪ 10 GB cap.
- ✅ **Harness I/O contract** — `/input/tasks.json` → `/output/results.json`, exit 0/non-zero, valid JSON always (atomic write).
- ✅ **Env-var contract** — `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS` read at runtime; no hardcoded model ids; no bundled `.env`.
- ✅ **All eight capability categories** routed and prompted for explicitly — validated 8/8 on real models (`eval/tasks_categories.jsonl`).
- ✅ **No cross-run caching / hardcoded answers** — in-memory exact-match dedup only; no cache file ships in the image.
- ✅ **Budgets** — ready < 60 s, < 30 s/request (25 s timeouts), < 10 min total (parallel workers), English-only outputs.
- ✅ **Gemma via Fireworks** — `ALLOWED_MODELS` auto-ordered Gemma-first for every escalation.
- ✅ **Public GitHub repo with README setup + usage instructions** — this file; MIT licensed (`LICENSE`).
- ⚠️ Before submitting: confirm the GHCR package is public and run one full harness-style test with the launch-day env values (see `AMDPLAN/07_SUBMISSION_CHECKLIST.md`).

## License

MIT — see [LICENSE](LICENSE).
