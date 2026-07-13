#!/bin/sh
# Container entrypoint — Track 1 submission.
# Starts the bundled Ollama server (the FREE local tier: its tokens score as
# zero), waits briefly for readiness, then runs the harness contract.
# If Ollama fails to come up, the run proceeds anyway: the router falls back
# to remote-only per task, which costs tokens but never fails the submission.
# Total startup stays well under the 60-second readiness budget.

# Tuning for the published grading VM (2 vCPU / 4 GB RAM). The 2026-07-13
# resubmission hit the 10-minute container limit there; these settings remove
# the two silent CPU-box failure modes on Ollama's side:
#  - model unload/reload thrash: keep both 1B-class models resident forever
#    (llama3.2:1b + qwen2.5:1.5b ~2.2 GB, fits in 4 GB with room for Python)
#  - parallel decode on 2 cores: two interleaved requests each run ~2x slower
#    and start tripping per-request timeouts; one at a time is strictly better
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=1
# Longest real prompt (NER hint retry) is well under 1k tokens; halving the
# default context halves KV-cache RAM per loaded model.
export OLLAMA_CONTEXT_LENGTH=2048

ollama serve >/tmp/ollama.log 2>&1 &

i=0
while [ $i -lt 25 ]; do
    if curl -sf http://127.0.0.1:11434/ >/dev/null 2>&1; then
        echo "ollama ready after ${i}s" >&2
        break
    fi
    i=$((i + 1))
    sleep 1
done

# Warm both models in the background (sequentially — one disk/RAM spike at a
# time) so the weights are resident before the first task needs them instead
# of during it. keep_alive -1 pins them for the whole run.
{
    curl -s http://127.0.0.1:11434/api/generate \
        -d '{"model":"qwen2.5:1.5b","prompt":"hi","options":{"num_predict":1},"keep_alive":-1}' \
        >/dev/null 2>&1
    curl -s http://127.0.0.1:11434/api/generate \
        -d '{"model":"llama3.2:1b","prompt":"hi","options":{"num_predict":1},"keep_alive":-1}' \
        >/dev/null 2>&1
} &

exec python harness_main.py
