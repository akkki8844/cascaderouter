#!/bin/sh
# Container entrypoint — Track 1 submission.
# Starts the bundled Ollama server (the FREE local tier: its tokens score as
# zero), waits briefly for readiness, then runs the harness contract.
# If Ollama fails to come up, the run proceeds anyway: the router falls back
# to remote-only per task, which costs tokens but never fails the submission.
# Total startup stays well under the 60-second readiness budget.

# Tuning for the published grading VM (2 vCPU / 4 GB RAM). The agent runs
# REMOTE-FIRST there (both graded local-first runs died on that box:
# TIMEOUT, then 5.3% — local 1B inference thrashes in 4 GB), so Ollama is
# only the last-resort fallback tier. Keep its footprint minimal instead of
# pinning weights the run should never need: one model loaded at a time,
# unloaded after idling, one decode at a time on the 2 cores.
export OLLAMA_KEEP_ALIVE=5m
export OLLAMA_MAX_LOADED_MODELS=1
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

# No model warm-up: the run is remote-first, and preloading 2.3 GB of local
# weights on a 4 GB box just steals RAM from the process that actually
# answers tasks. Ollama loads a model on demand iff a fallback ever needs it.

exec python harness_main.py
