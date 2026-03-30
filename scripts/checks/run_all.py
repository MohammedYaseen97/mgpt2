"""
Run all assignment checks in phase order.

Phases not yet implemented are skipped with a clear message rather than
causing a hard failure — this lets you track progress incrementally.
"""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str], *, label: str) -> bool:
    """Run cmd; print outcome; return True on success."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> None:
    failures: list[str] = []

    # ------------------------------------------------------------------
    # Phase A — tokenizer artifacts
    # ------------------------------------------------------------------
    if not run(
        [sys.executable, "-m", "scripts.checks.check_phase_a_tokenizer"],
        label="Phase A — tokenizer",
    ):
        failures.append("Phase A: tokenizer check failed")

    # ------------------------------------------------------------------
    # Phase B — raw JSONL data (SFT + DPO)
    # ------------------------------------------------------------------
    if not run(
        [sys.executable, "-m", "scripts.checks.check_jsonl"],
        label="Phase B — JSONL data integrity",
    ):
        failures.append("Phase B: JSONL check failed")

    # ------------------------------------------------------------------
    # Phase B — tokenized shards  (pretraining + SFT)
    # ------------------------------------------------------------------
    if not run(
        [sys.executable, "-m", "scripts.checks.check_shards"],
        label="Phase B — shard integrity",
    ):
        failures.append("Phase B: shard check failed")

    # ------------------------------------------------------------------
    # Phase C — pretrain run
    # ------------------------------------------------------------------
    if not run(
        [sys.executable, "-m", "scripts.checks.check_pretrain_run"],
        label="Phase C — pretrain run",
    ):
        failures.append("Phase C: pretrain run check failed")

    # ------------------------------------------------------------------
    # Phase D / E — TODO
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Phase D / E checks — TODO")
    print(f"{'=' * 60}")
    print("  SKIP  implement eval/sft_eval.py   (Phase D)")
    print("  SKIP  implement eval/dpo_eval.py   (Phase E)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    if failures:
        print(f"OVERALL: {len(failures)} phase(s) failed")
        for f in failures:
            print(f"  ✗  {f}")
        sys.exit(1)
    else:
        print("OVERALL: all implemented checks passed.")


if __name__ == "__main__":
    main()

