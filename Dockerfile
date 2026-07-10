# Hybrid Token-Efficient Routing Agent — Track 1, AMD Developer Hackathon: ACT II
#
# SELF-CONTAINED submission image: Python agent + Ollama + the two 1B-class
# free-tier models baked into the image layers, so the judging VM needs no
# sibling services. The judging VM runs linux/amd64 — build with:
#   docker buildx build --platform linux/amd64 -t <registry>/<image>:latest --push .
#
# Contract (see harness_main.py): reads /input/tasks.json, writes
# /output/results.json, exits 0/1. The harness injects FIREWORKS_API_KEY,
# FIREWORKS_BASE_URL, and ALLOWED_MODELS at runtime — nothing is hardcoded.
# Compressed image stays well under the 10GB cap (~4GB: two ~1GB models +
# the Ollama runtime) — and under the organizers' revised ~5GB "comfortably
# safe" guidance for avoiding intermittent PULL_ERRORs under scoring load.
# Grading VM is 2 vCPU / 4 GB RAM with NO model runtime pre-installed, so
# Ollama + weights being baked in below is a hard requirement, not a nicety.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*

# Ollama serves the FREE local tier (llama3.2:1b + qwen2.5:1.5b) — local
# tokens score as zero, so every task resolved here costs nothing.
RUN curl -fsSL https://ollama.com/install.sh | sh

# Bake the local models into the image at build time: the judging VM must
# not need network pulls, and first-token latency stays inside the budget.
ENV OLLAMA_HOST=127.0.0.1:11434
RUN ollama serve & \
    SERVER=$!; \
    i=0; until curl -sf http://127.0.0.1:11434/ >/dev/null 2>&1; do \
        i=$((i+1)); [ $i -gt 60 ] && echo "ollama never came up" && exit 1; \
        sleep 1; \
    done; \
    ollama pull llama3.2:1b && \
    ollama pull qwen2.5:1.5b && \
    kill $SERVER

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
