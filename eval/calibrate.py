"""Calibration: fit the escalation threshold instead of guessing it.

Implements the Calibration Methodology from 03_ARCHITECTURE: read logged
cascade decisions (which carry per-task features + correctness from eval
runs), fit a logistic regression predicting P(free-tier answer is correct),
then pick the escalation threshold as a constrained optimization — the most
permissive threshold (fewest escalations) whose predicted accuracy stays
above the target floor.

Run after at least one cascade eval run:
    python -m eval.calibrate --accuracy-floor 0.9

Prints the recommended `router.escalation_threshold` for config.yaml.
Requires scikit-learn (in requirements.txt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_samples(decisions_path: str) -> tuple[list[list[float]], list[int]]:
    X, y = [], []
    path = Path(decisions_path)
    if not path.exists():
        raise SystemExit(f"No decision log at {decisions_path} — run an eval first.")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        f = d.get("features", {})
        if "correct" not in f:
            continue  # decision not from a labeled eval run
        if d.get("route") == "cache":
            continue  # cache hits carry no fresh signal
        X.append([
            1.0 if f.get("agreement") else 0.0,
            float(f.get("similarity", 0.0)),
            1.0 if str(f.get("critique_confidence", "")).lower() == "high" else 0.0,
            min(float(f.get("prompt_chars", 0)) / 1000.0, 1.0),
        ])
        # label: was the FREE-tier answer good enough (i.e., correct without
        # needing remote)? For decisions that escalated, the free tier failed
        # by construction only if the router was right to escalate — we use
        # final correctness on free routes, and 0 for escalated ones as a
        # conservative proxy until we log the local draft's own correctness.
        free = d.get("route") in ("local_agreement", "local_critique", "local_only")
        y.append(1 if (free and f.get("correct")) else 0)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", default="logs/decisions.jsonl")
    parser.add_argument("--accuracy-floor", type=float, default=0.9,
                        help="minimum acceptable predicted accuracy on tasks "
                             "kept local (set above the real, undisclosed bar)")
    args = parser.parse_args()

    X, y = load_samples(args.decisions)
    if len(X) < 10 or len(set(y)) < 2:
        raise SystemExit(
            f"Only {len(X)} usable samples (need >=10 with both outcomes). "
            "Run more labeled cascade evals first.")

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    # Constrained optimization: lowest threshold (most tasks stay local /
    # fewest escalations) whose kept-local subset stays above the floor.
    best = None
    for threshold in sorted(set(round(p, 3) for p in probs)):
        kept = [(p, label) for p, label in zip(probs, y) if p >= threshold]
        if not kept:
            continue
        acc = sum(label for _, label in kept) / len(kept)
        if acc >= args.accuracy_floor:
            best = (threshold, acc, len(kept))
            break

    print(f"samples={len(X)}  positives={sum(y)}")
    if best is None:
        print("No threshold satisfies the accuracy floor — local models are "
              "too weak for this task domain. Consider a stronger local model "
              "before more routing cleverness (see 11_WINNING_STRATEGY).")
        return
    threshold, acc, kept = best
    print(f"recommended router.escalation_threshold: {threshold}")
    print(f"  predicted kept-local accuracy: {acc:.1%} on {kept}/{len(X)} tasks")
    print("Update config.yaml, then re-run the full eval to confirm.")


if __name__ == "__main__":
    main()
