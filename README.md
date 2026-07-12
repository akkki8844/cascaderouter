# Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II — Track 1** · Team: **Veritas**

An AI agent that routes every task to the **allowed model best suited to its category** — code models for code, cross-family reasoning votes for math/logic, Gemma for sentiment/summarization — spending remote tokens exactly where they buy accuracy, and nowhere else. Track 1's accuracy threshold is a hard gate: a cheap-but-wrong submission scores nothing, so the router is accuracy-first by design (v4; the v3 free-local-first cascade scored 57.9% on the real grading run and taught us that lesson empirically).

**Internal validation on real Fireworks models (19-task set mirroring the grading distribution, no mocks): 19/19 correct, 8,822 remote tokens, 27 s wall time** (`eval_results/hard_v4b_decisions.jsonl`). See `AMDPLAN/IMPLEMENTATION.md` for the full build record and `AMDPLAN/RUN.md` for a copy-paste run guide.

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

Each task is first classified into one of the eight Track 1 capability categories, then routed to the model best suited to that category. Accuracy is a **gate** — officially 80% on a fixed 19-task set: below it a submission scores nothing regardless of token count. The v4 router is therefore *accuracy-first*: every graded answer comes from a strong remote model, with the local models kept only as an emergency fallback, and tokens are saved through category-matched single calls, early-stop voting, and terse structured output rather than through free-but-unreliable local answering.

```
task ──▶ within-run dedup cache ──hit──▶ answer (0 tokens)
              │ miss
              ▼
      category classifier (8 Track 1 categories)
              │
    ┌─────────┼──────────────────┬────────────────────────┐
    │         │                  │                        │
math / logic  code gen /         sentiment /              factual / NER
    │         code debug         summarization                │
    ▼         │                  │                            ▼
cross-family  ▼                  ▼                     single call,
self-consistency: strongest code model  Gemma-first    strongest general
strongest general (kimi-k2p7-code),     single call    model, terse
+ reasoning model  single call,         (Gemma bonus   category prompt
+ Gemma vote;      category-tuned       prize; strong
early-stop when    prompt               models behind
first two agree                         it if it fails)
              │ (any tier fails or returns blank)
              ▼
   next allowed model … ▶ local models (last resort — never return blank)
```

Key techniques (see `AMDPLAN/03_ARCHITECTURE.md` for the full design rationale):

- **Category-aware model selection** — each of the eight competition categories is answered by the allowed model best suited to it: `kimi-k2p7-code` for code generation/debugging and general short answers, cross-family voting for math/logic, Gemma-4 checkpoints leading sentiment and summarization.
- **Cross-family self-consistency** — math/logic votes come from *independently trained* models (general instruct + dedicated reasoning model + Gemma), because repeated samples of one model family reproduce that family's systematic biases. Early stop when the first two families agree (2-of-2 cross-family agreement is stronger than any same-family majority, and ~40% cheaper).
- **Never-blank guarantee** — reasoning-channel models can return HTTP 200 with empty content when truncated; an empty answer is a guaranteed zero. Every route escalates through the remaining allowed models on failure *or* blank output, ending at the bundled local models as a last resort.
- **Structured JSON output on every tier** — no preamble or filler tokens, ever; short-form categories answer with a bare label/number/name.
- **Within-run dedup cache** — repeated prompts inside one grading run are answered once (exact-match, in-memory only; nothing is precomputed or persisted).
- **Lesson learned the hard way** — v3 answered ~46% of tasks with free 1B-class local models and scored 57.9% on the real grading run: below the gate, every saved token was worth nothing. v4 spends tokens where they buy accuracy.

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
