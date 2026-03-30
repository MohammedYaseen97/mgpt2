"""
Phase C check: verify pretrain runs are comparable and complete.

Checks (in order):
  1. Both run directories exist (auto-discovered or supplied via --baseline / --mgpt2)
  2. run_info.json present in each run
  3. Token budget equality: total_batch_size × max_steps must match
  4. Hyperparams match: seed, block_size, max_lr, min_lr_ratio, weight_decay
     warmup fraction (warmup_steps / max_steps) is compared with tolerance
  5. Artifacts exist: final_metrics.json, log.txt, at least one model_*.pt
  6. HellaSwag scores are present for both models
  7. (Warning only) lm_eval status — skipped is acceptable for smoke runs

Exit 0 = all checks passed (warnings do not fail).
Exit 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

HYPERPARAM_KEYS = ("seed", "block_size", "max_lr", "min_lr_ratio", "weight_decay")
WARMUP_FRACTION_TOLERANCE = 0.01   # allow ±1 pp difference in warmup fraction


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _latest_run(prefix: str) -> Path | None:
    """Return the lexicographically latest runs/ directory whose name starts with prefix."""
    candidates = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    )
    return candidates[-1] if candidates else None


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# individual checks — each returns (passed: bool, message: str)
# ──────────────────────────────────────────────────────────────────────────────

def check_run_info(run_dir: Path) -> tuple[bool, str]:
    p = run_dir / "run_info.json"
    if not p.exists():
        return False, f"Missing run_info.json in {run_dir.name}"
    return True, f"run_info.json present in {run_dir.name}"


def check_token_budget(baseline_info: dict, mgpt2_info: dict) -> tuple[bool, str]:
    def budget(info: dict) -> int:
        t = info["train"]
        return t["total_batch_size"] * t["max_steps"]

    b = budget(baseline_info)
    m = budget(mgpt2_info)
    if b != m:
        return (
            False,
            f"Token budgets differ: baseline={b:,}  mgpt2={m:,}  "
            f"(total_batch_size × max_steps)",
        )
    return True, f"Token budgets equal: {b:,} tokens each"


def check_hyperparams(baseline_info: dict, mgpt2_info: dict) -> tuple[bool, str]:
    bt = baseline_info["train"]
    mt = mgpt2_info["train"]
    mismatches: list[str] = []

    for key in HYPERPARAM_KEYS:
        bv, mv = bt.get(key), mt.get(key)
        if bv != mv:
            mismatches.append(f"{key}: baseline={bv}  mgpt2={mv}")

    # warmup as a fraction of max_steps
    b_frac = bt.get("warmup_steps", 0) / bt["max_steps"]
    m_frac = mt.get("warmup_steps", 0) / mt["max_steps"]
    if abs(b_frac - m_frac) > WARMUP_FRACTION_TOLERANCE:
        mismatches.append(
            f"warmup_fraction: baseline={b_frac:.4f}  mgpt2={m_frac:.4f}"
        )

    if mismatches:
        detail = ";  ".join(mismatches)
        return False, f"Hyperparam mismatch — {detail}"
    return True, "Hyperparams match across both runs"


def check_artifacts(run_dir: Path) -> tuple[bool, str]:
    missing: list[str] = []

    if not (run_dir / "final_metrics.json").exists():
        missing.append("final_metrics.json")
    if not (run_dir / "log.txt").exists():
        missing.append("log.txt")
    if not list(run_dir.glob("model_*.pt")):
        missing.append("model_*.pt checkpoint")

    if missing:
        return False, f"{run_dir.name}: missing artifacts — {', '.join(missing)}"
    return True, f"{run_dir.name}: all artifacts present"


def check_hellaswag(run_dir: Path, metrics: dict) -> tuple[bool, str]:
    # Accept either the inline training score or a dedicated hellaswag_eval.json
    has_inline = metrics.get("final_hellaswag_acc") is not None
    has_full = (run_dir / "hellaswag_eval.json").exists()

    if has_full:
        data = _load_json(run_dir / "hellaswag_eval.json")
        acc = data.get("acc_norm", data.get("accuracy"))
        return True, f"{run_dir.name}: full HellaSwag acc={acc}"
    if has_inline:
        acc = metrics["final_hellaswag_acc"]
        return True, f"{run_dir.name}: inline HellaSwag acc={acc} (smoke/partial)"
    return False, f"{run_dir.name}: no HellaSwag score found"


def check_lm_eval(run_dir: Path, metrics: dict) -> tuple[bool, str]:
    """Warning-only: lm_eval skipped is acceptable (e.g. smoke run)."""
    status = metrics.get("lm_eval_status", "unknown")
    if isinstance(status, dict) and "overall_ppl" in status:
        ppl = status["overall_ppl"]
        return True, f"{run_dir.name}: lm_eval perplexity={ppl:.2f}"
    if isinstance(status, str) and status.startswith("skipped"):
        return True, f"{run_dir.name}: lm_eval skipped (ok for smoke)"
    if (run_dir / "lm_eval.json").exists():
        return True, f"{run_dir.name}: lm_eval.json present"
    return True, f"{run_dir.name}: lm_eval status={status!r} (warning only)"


# ──────────────────────────────────────────────────────────────────────────────
# runner
# ──────────────────────────────────────────────────────────────────────────────

def _report(passed: bool, msg: str, failures: list[str]) -> None:
    tag = "OK  " if passed else "FAIL"
    print(f"  [{tag}] {msg}")
    if not passed:
        failures.append(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check Phase C pretrain runs.")
    ap.add_argument(
        "--baseline",
        type=Path,
        help="Path to baseline run directory (default: latest runs/baseline_* dir)",
    )
    ap.add_argument(
        "--mgpt2",
        type=Path,
        help="Path to mgpt2 run directory (default: latest runs/mgpt2_* dir)",
    )
    args = ap.parse_args()

    baseline_dir = args.baseline or _latest_run("baseline_")
    mgpt2_dir = args.mgpt2 or _latest_run("mgpt2_")

    failures: list[str] = []

    print(f"\n{'=' * 60}")
    print("  Phase C — pretrain run checks")
    print(f"{'=' * 60}")

    # ── 1. directories exist ──────────────────────────────────────────────────
    for label, d in [("baseline", baseline_dir), ("mgpt2", mgpt2_dir)]:
        if d is None or not d.exists():
            _report(False, f"No {label} run directory found in {RUNS_DIR}", failures)
    if failures:
        sys.exit(1)

    print(f"\n  baseline : {baseline_dir.name}")
    print(f"  mgpt2    : {mgpt2_dir.name}\n")

    # ── 2. run_info.json ──────────────────────────────────────────────────────
    for run_dir in (baseline_dir, mgpt2_dir):
        _report(*check_run_info(run_dir), failures)

    if failures:
        sys.exit(1)

    baseline_info = _load_json(baseline_dir / "run_info.json")
    mgpt2_info = _load_json(mgpt2_dir / "run_info.json")

    # ── 3. token budget ───────────────────────────────────────────────────────
    _report(*check_token_budget(baseline_info, mgpt2_info), failures)

    # ── 4. hyperparams ────────────────────────────────────────────────────────
    _report(*check_hyperparams(baseline_info, mgpt2_info), failures)

    # ── 5. artifacts ──────────────────────────────────────────────────────────
    for run_dir in (baseline_dir, mgpt2_dir):
        _report(*check_artifacts(run_dir), failures)

    # ── 6. HellaSwag ──────────────────────────────────────────────────────────
    for run_dir in (baseline_dir, mgpt2_dir):
        if (run_dir / "final_metrics.json").exists():
            metrics = _load_json(run_dir / "final_metrics.json")
        else:
            metrics = {}
        _report(*check_hellaswag(run_dir, metrics), failures)

    # ── 7. lm_eval (warning only) ─────────────────────────────────────────────
    for run_dir in (baseline_dir, mgpt2_dir):
        metrics = _load_json(run_dir / "final_metrics.json") if (run_dir / "final_metrics.json").exists() else {}
        ok, msg = check_lm_eval(run_dir, metrics)
        tag = "WARN" if not ok else "OK  "
        print(f"  [{tag}] {msg}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if failures:
        print(f"OVERALL: {len(failures)} check(s) failed")
        for f in failures:
            print(f"  ✗  {f}")
        sys.exit(1)
    else:
        print("OVERALL: Phase C checks passed.")


if __name__ == "__main__":
    main()
