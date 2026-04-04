"""Orchestrate DPO training from a YAML config file.

Single-GPU:
    python scripts/run_dpo.py --config configs/dpo_mgpt2.yaml

Creates:
    runs/dpo_{name}_{timestamp}/
        config.yaml         verbatim copy of input config
        run_info.json       git hash, SFT checkpoint sha256, hyperparams
        log.txt             step-level metrics written by train_dpo.py
        metrics.jsonl       one JSON object per logged step
        final_metrics.json  end-of-run summary
        model_*.pt          checkpoints written by train_dpo.py
        dpo_eval.json       win-rate + SFT regression check
"""

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON    = str(REPO_ROOT / "virtual" / "bin" / "python")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DPO from a YAML config.")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _find_sft_checkpoint() -> Path:
    """Auto-discover latest Phase D SFT checkpoint (production run preferred)."""
    prod = sorted(p for p in REPO_ROOT.glob("runs/sft_*/model_*.pt")
                  if "smoke" not in p.parent.name)
    if prod:
        return prod[-1]
    any_ = sorted(REPO_ROOT.glob("runs/sft_*/model_*.pt"))
    if any_:
        return any_[-1]
    raise FileNotFoundError(
        "No SFT checkpoint found under runs/sft_*/. Run Phase D first."
    )


def _resolve_checkpoint(spec: str) -> Path:
    if spec == "auto":
        p = _find_sft_checkpoint()
        print(f"auto-discovered SFT checkpoint: {p}")
        return p
    p = Path(spec)
    return p if p.is_absolute() else REPO_ROOT / p


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def _make_run_dir(config: dict, config_path: Path) -> tuple[Path, Path]:
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / f"dpo_{config['name']}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")

    sft_path = _resolve_checkpoint(config.get("sft_checkpoint", "auto"))

    tok      = config.get("tokenizer", {})
    tok_info = {"kind": tok.get("kind")}
    if tok.get("model_file"):
        mp = REPO_ROOT / tok["model_file"]
        if mp.exists():
            tok_info["path"]   = tok["model_file"]
            tok_info["sha256"] = _sha256(mp)

    run_info = {
        "name":            f"dpo_{config['name']}_{ts}",
        "config_file":     str(config_path.relative_to(REPO_ROOT)),
        "git_hash":        _git_hash(),
        "timestamp":       ts,
        "sft_checkpoint":  str(sft_path),
        "sft_sha256":      _sha256(sft_path) if sft_path.exists() else "missing",
        "tokenizer":       tok_info,
        "train":           config.get("train", {}),
        "dpo_eval_status": "pending",
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    return run_dir, sft_path


# ---------------------------------------------------------------------------
# Build train_dpo.py command
# ---------------------------------------------------------------------------

def _build_cmd(config: dict, run_dir: Path, sft_path: Path) -> list[str]:
    t = config.get("train", {})
    cmd_args = [
        "--sft-checkpoint",   str(sft_path),
        "--shards-dir",       t.get("shards_dir",      "data/shards_dpo"),
        "--seed",             str(t.get("seed",                     1337)),
        "--batch-size",       str(t.get("batch_size",                 32)),
        "--micro-batch-size", str(t.get("micro_batch_size",            4)),
        "--beta",             str(t.get("beta",                      0.1)),
        "--max-lr",           str(t.get("max_lr",                   1e-6)),
        "--min-lr-ratio",     str(t.get("min_lr_ratio",              0.1)),
        "--warmup-steps",     str(t.get("warmup_steps",               20)),
        "--epochs",           str(t.get("epochs",                      1)),
        "--weight-decay",     str(t.get("weight_decay",              0.1)),
        "--eval-interval",    str(t.get("eval_interval",              50)),
        "--log-dir",          str(run_dir),
    ]
    if t.get("max_steps", 0):
        cmd_args += ["--max-steps", str(t["max_steps"])]

    return [sys.executable, str(REPO_ROOT / "train_dpo.py")] + cmd_args


# ---------------------------------------------------------------------------
# Post-process log.txt → metrics.jsonl + final_metrics.json
# ---------------------------------------------------------------------------

def _post_process(run_dir: Path) -> None:
    log_path = run_dir / "log.txt"
    if not log_path.exists():
        return

    metrics: list[dict] = []
    by_kind: dict[str, list] = {}

    for line in log_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            step, kind, val = int(parts[0]), parts[1], float(parts[2])
        except ValueError:
            continue
        entry = {"step": step, "kind": kind, "value": val}
        metrics.append(entry)
        by_kind.setdefault(kind, []).append(val)

    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m) for m in metrics)
    )
    summary = {
        "total_steps":          metrics[-1]["step"] if metrics else 0,
        "final_train_loss":     by_kind.get("train_loss",   [None])[-1],
        "final_train_margin":   by_kind.get("train_margin", [None])[-1],
        "final_val_loss":       by_kind.get("val_loss",     [None])[-1],
        "final_val_win_rate":   by_kind.get("val_win_rate", [None])[-1],
        "dpo_eval_status":      "pending",
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Post-training dpo_eval
# ---------------------------------------------------------------------------

def _try_dpo_eval(run_dir: Path, config: dict, sft_path: Path) -> str:
    ckpts = sorted(run_dir.glob("model_*.pt"))
    if not ckpts:
        return "skipped — no checkpoint found"

    t          = config.get("train", {})
    tok        = config.get("tokenizer", {})
    tok_model  = tok.get("model_file", "tokenizer/artifacts/mgpt2.model")
    dpo_shards = t.get("shards_dir", "data/shards_dpo")

    try:
        subprocess.check_call([
            PYTHON, "-m", "eval.dpo_eval",
            "--dpo-checkpoint",  str(ckpts[-1]),
            "--sft-checkpoint",  str(sft_path),
            "--tokenizer-model", str(REPO_ROOT / tok_model),
            "--dpo-shards-dir",  dpo_shards,
            "--sft-shards-dir",  "data/shards_sft",
            "--device",          "cuda",
            "--out",             str(run_dir / "dpo_eval.json"),
        ], cwd=REPO_ROOT)
        return "complete"
    except subprocess.CalledProcessError:
        return "failed — see dpo_eval output"
    except Exception as e:
        return f"skipped — {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args        = _parse_args()
    args.config = args.config.resolve()
    config      = yaml.safe_load(args.config.read_text())

    run_dir, sft_path = _make_run_dir(config, args.config)
    print(f"run dir    : {run_dir}")
    print(f"sft ckpt   : {sft_path}")

    cmd = _build_cmd(config, run_dir, sft_path)
    print("+", " ".join(cmd))

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    _post_process(run_dir)

    dpo_status = _try_dpo_eval(run_dir, config, sft_path)
    for fname in ("run_info.json", "final_metrics.json"):
        p = run_dir / fname
        if p.exists():
            data = json.loads(p.read_text())
            data["dpo_eval_status"] = dpo_status
            p.write_text(json.dumps(data, indent=2))

    if result.returncode != 0:
        print(f"train_dpo.py exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"done — results in {run_dir}")


if __name__ == "__main__":
    main()
