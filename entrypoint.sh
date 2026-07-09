#!/bin/sh
# Container entrypoint — Track 1 submission.
# Starts the bundled Ollama server (the FREE local tier: its tokens score as
# zero), waits briefly for readiness, then runs the harness contract.
# If Ollama fails to come up, the run proceeds anyway: the router falls back
# to remote-only per task, which costs tokens but never fails the submission.
# Total startup stays well under the 60-second readiness budget.

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

exec python harness_main.py
