# Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II — Track 1** · Team: [TEAM NAME]

An AI agent that completes tasks using the **cheapest possible path**: two free local models vote first, and only tasks they can't confidently resolve pay for a remote call via **Fireworks AI** (Gemma tier first). Local tokens score as **zero** on the Track 1 leaderboard — so the agent's job is to know when it doesn't need the expensive model.

**Internal validation on real models (our own 50-task proxy set, no mocks): 100% accuracy, 16,730 remote tokens, 23/50 tasks resolved at $0.** See `AMDPLAN/IMPLEMENTATION.md` for the full build record and `AMDPLAN/RUN.md` for a copy-paste run guide.

| Strategy | Accuracy | Remote tokens | Free resolution |
|---|---|---|---|
| always_local | 78% | 0 | 50/50 |
| heuristic | 80% | 1,572 | 39/50 |
| always_remote | 96% | 7,383 | 0/50 |
| **cascade (this design)** | **100%** | 16,730 | 23/50 |

Cascade is the only strategy reaching 100% — single-shot remote misses the trap logic puzzles that need multi-vote reasoning.

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
| `ALLOWED_MODELS` | the only permitted model ids; we auto-order them **Gemma-first** (bonus prize), largest first |

Budgets honored: container ready < 60 s (Ollama starts in ~2 s, models are pre-baked), < 30 s per request (25 s client timeout), < 10 min total (tasks run on a 4-worker pool), image ≪ 10 GB (~4 GB), `linux/amd64` build. No answer caching across runs: the within-run dedup cache runs in-memory and exact-match only. A failing local tier degrades to remote-only instead of failing the run; a failing single task yields an empty answer instead of a crashed container.

## How it works

Each task is first classified into one of the eight Track 1 capability categories, then routed down the cheapest path that still clears the accuracy gate (accuracy is a **gate**: below-threshold submissions are excluded from the leaderboard entirely, so weak-local categories go remote on purpose):

```
task ──▶ within-run dedup cache ──hit──▶ answer (0 tokens)
              │ miss
              ▼
      category classifier (8 Track 1 categories)
              │
    ┌─────────┼──────────────────────────────┐
    │         │                              │
factual /   math / logic              code gen / code debug /
sentiment     │                       summarization / NER
    │         ▼                              │
    │   remote self-consistency              ▼
    │   (up to 4 reasoning votes,     single remote call,
    │    early-stop when the first    category-tuned prompt
    │    two agree)                   (completeness/long-form)
    ▼
Local Model A ─┐
               ├─ agree? ──yes──▶ answer (0 tokens)
Local Model B ─┘
    │ disagree
    ▼
Local critique pass ──confident──▶ answer (0 tokens)
    │ unresolved
    ▼
Fireworks remote, tier 1 (Gemma): "verify or FIX this draft"
    │ unusable verdict
    ▼
Fireworks remote, tier 2: next allowed model
```

Key techniques (see `AMDPLAN/03_ARCHITECTURE.md` for the full design rationale):

- **Category-aware routing** — the eight competition categories map to three route families. Factual/sentiment stay on the free local tier; math/logic get remote self-consistency voting; code, summarization, and NER get one remote call with a category-tuned prompt (code/summaries because string-agreement between two local models is meaningless on long-form text; NER because it's graded on *completeness*, which a local critic can't verify — validated: a 1B local dropped an entity and the critic confidently approved it).
- **Dual-local-model agreement** — two small models from different training lineages (Llama 3.2 + Qwen2.5) vote; agreement is a far stronger free confidence signal than any single model's self-rating.
- **Computational bypass** — math/logic prompts skip local agreement entirely: small models fail arithmetic in *correlated* ways (they agree on the same wrong answer), so agreement is not a valid signal there.
- **Verify-not-regenerate escalation** — the remote model is sent our local draft and asked to confirm/fix it, not re-answer from scratch. Confirm/fix responses are shorter → fewer scored tokens.
- **Structured JSON output on every tier** — no preamble or filler tokens, ever, local or remote.
- **Early-stop self-consistency** — up to 4 reasoning votes, stopped after 2 when they agree (−37% remote tokens at unchanged accuracy in our validation).
- **Calibrated escalation threshold** — fit against our own eval logs (`eval/calibrate.py`), not hand-picked.

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
- ⚠️ Before submitting: fill in `[TEAM NAME]`, push the image to a public registry, and run one full harness-style test with the launch-day env values (see `AMDPLAN/07_SUBMISSION_CHECKLIST.md`).

## License

MIT — see [LICENSE](LICENSE).
