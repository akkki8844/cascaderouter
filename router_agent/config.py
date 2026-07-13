"""Config loading. Everything tunable lives in config.yaml, nothing hardcoded.

The task domain, standardized-environment specs, and Fireworks model list are
all unknown until launch day — so model ids, thresholds, and strategy are
config values the team can change without touching code.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class LocalModelConfig:
    backend: str = "mock"  # mock | openai_compatible
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"  # Ollama ignores the key but the client requires one
    model_a: str = "llama3.2:1b"
    model_b: str = "qwen2.5:1.5b"
    critic_model: str = "llama3.2:1b"
    max_tokens: int = 512
    temperature: float = 0.0
    # None = generous default (local CPU dev); harness mode sets 25.0 to stay
    # under the competition's 30s-per-request ceiling
    request_timeout: float | None = None


@dataclass
class RemoteModelConfig:
    backend: str = "mock"  # mock | fireworks
    base_url: str = "https://api.fireworks.ai/inference/v1"
    api_key_env: str = "FIREWORKS_API_KEY"
    # Tier order matters: cheapest first. Gemma first for the Gemma bonus prize.
    # Official Track 1 allowed list (organizer announcement, 2026-07-08),
    # pre-ordered as order_tiers() would emit it. The harness-injected
    # ALLOWED_MODELS always overrides this default.
    tiers: list = field(default_factory=lambda: [
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/kimi-k2p7-code",
    ])
    max_tokens: int = 512
    temperature: float = 0.0
    request_timeout: float | None = None

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        return key


@dataclass
class RouterConfig:
    # always_remote (v0) | always_local (v1) | heuristic (v2) | cascade (v3)
    strategy: str = "cascade"
    # Hard run-wide ceiling on billed remote (Fireworks) tokens; 0 disables.
    # Harness mode sets this (REMOTE_TOKEN_BUDGET env, default 480) so the
    # leaderboard number is bounded no matter what the hidden task set does.
    remote_token_budget: int = 0
    escalation_threshold: float = 0.5
    critique_enabled: bool = True
    agreement_fuzzy_threshold: float = 0.90
    # heuristic (v2) knobs
    heuristic_max_prompt_chars: int = 400
    heuristic_hard_markers: list = field(default_factory=lambda: [
        "prove", "derive", "step by step", "multi-step", "riddle",
    ])


@dataclass
class CacheConfig:
    enabled: bool = True
    similarity_threshold: float = 0.95
    path: str = "cache/task_cache.jsonl"
    # persist=False keeps the cache purely in-memory for the current run —
    # required in harness mode: the competition forbids shipping/precomputing
    # cached answers, and a container must not carry state between runs.
    persist: bool = True


@dataclass
class LoggingConfig:
    decisions_path: str = "logs/decisions.jsonl"


@dataclass
class Config:
    local: LocalModelConfig = field(default_factory=LocalModelConfig)
    remote: RemoteModelConfig = field(default_factory=RemoteModelConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dataclass(instance, data: dict):
    for key, value in (data or {}).items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg = Config()
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _merge_dataclass(cfg.local, raw.get("local"))
        _merge_dataclass(cfg.remote, raw.get("remote"))
        _merge_dataclass(cfg.router, raw.get("router"))
        _merge_dataclass(cfg.cache, raw.get("cache"))
        _merge_dataclass(cfg.logging, raw.get("logging"))
    return cfg


def _model_size_hint(model_id: str) -> float:
    """Best-effort parameter count parsed from a model id ('gemma2-9b-it' -> 9,
    'llama-v3p1-70b-instruct' -> 70). Unknown sizes sort last."""
    sizes = re.findall(r"(\d+(?:\.\d+)?)\s*b\b", model_id.lower())
    return max((float(s) for s in sizes), default=0.0)


def order_tiers(models: list[str]) -> list[str]:
    """Order the launch-day ALLOWED_MODELS list into escalation tiers.

    Gemma-family models first (largest first) — that keeps escalations
    Gemma-first for the 'Best Use of Gemma via Fireworks' bonus prize —
    then the remaining models, also largest-capability first.
    """
    gemma = sorted((m for m in models if "gemma" in m.lower()),
                   key=_model_size_hint, reverse=True)
    rest = sorted((m for m in models if "gemma" not in m.lower()),
                  key=_model_size_hint, reverse=True)
    return (gemma + rest) or models


def apply_env_overrides(cfg: Config) -> Config:
    """Apply the competition harness's runtime environment contract.

    The judging harness injects FIREWORKS_BASE_URL (ALL remote calls must go
    through it or they score zero tokens), FIREWORKS_API_KEY (read lazily via
    RemoteModelConfig.api_key), and ALLOWED_MODELS (the only permitted model
    ids, revealed on launch day — calling anything else invalidates the
    submission). Nothing here may be hardcoded.
    """
    base_url = os.environ.get("FIREWORKS_BASE_URL", "").strip()
    if base_url:
        cfg.remote.base_url = base_url
        cfg.remote.backend = "openai_compatible"
    allowed = os.environ.get("ALLOWED_MODELS", "").strip()
    if allowed:
        models = [m.strip() for m in allowed.split(",") if m.strip()]
        if models:
            cfg.remote.tiers = order_tiers(models)
    return cfg
