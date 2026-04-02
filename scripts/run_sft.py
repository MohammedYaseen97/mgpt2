"""Orchestrate SFT training from a YAML config file.

Single-GPU:
    python scripts/run_sft.py --config configs/sft_mgpt2.yaml

Creates:
    runs/sft_{name}_{timestamp}/
        config.yaml         verbatim copy of input config
        run_info.json       git hash, pretrained checkpoint sha256, hyperparams
        log.txt             step-level metrics written by train_sft.py
        metrics.jsonl       one JSON object per logged step
        final_metrics.json  end-of-run summary
        model_*.pt          checkpoints written by train_sft.py
        sft_eval.json       val loss + prompt suite generations
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
    p = argparse.ArgumentParser(description="Run SFT from a YAML config.")
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


def _find_pretrained_checkpoint() -> Path:
    """Auto-discover latest mgpt2 pretrain checkpoint (production run preferred)."""
    # prefer production runs (no 'smoke' in name)
    prod_ckpts = sorted(REPO_ROOT.glob("runs/mgpt2_custom_tokenizer_*/model_*.pt"))
    prod_ckpts = [p for p in prod_ckpts if "smoke" not in p.parent.name]
    if prod_ckpts:
        return prod_ckpts[-1]
    # fall back to any mgpt2 run (including smoke)
    all_ckpts = sorted(REPO_ROOT.glob("runs/mgpt2_custom_tokenizer_*/model_*.pt"))
    if all_ckpts:
        return all_ckpts[-1]
    raise FileNotFoundError(
        "No mgpt2 pretrain checkpoint found under runs/. Run Phase C first."
    )


def _resolve_checkpoint(spec: str) -> Path:
    if spec == "auto":
        p = _find_pretrained_checkpoint()
        print(f"auto-discovered pretrained checkpoint: {p}")
        return p
    p = Path(spec)
    return p if p.is_absolute() else REPO_ROOT / p


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def _make_run_dir(config: dict, config_path: Path) -> Path:
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / f"sft_{config['name']}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")

    pretrained_spec = config.get("model", {}).get("pretrained_checkpoint", "auto")
    pretrained_path = _resolve_checkpoint(pretrained_spec)

    tok      = config.get("tokenizer", {})
    tok_info = {"kind": tok.get("kind")}
    if tok.get("model_file"):
        mp = REPO_ROOT / tok["model_file"]
        if mp.exists():
            tok_info["path"]   = tok["model_file"]
            tok_info["sha256"] = _sha256(mp)

    run_info = {
        "name":                  f"sft_{config['name']}_{ts}",
        "config_file":           str(config_path.relative_to(REPO_ROOT)),
        "git_hash":              _git_hash(),
        "timestamp":             ts,
        "pretrained_checkpoint": str(pretrained_path),
        "pretrained_sha256":     _sha256(pretrained_path) if pretrained_path.exists() else "missing",
        "tokenizer":             tok_info,
        "train":                 config.get("train", {}),
        "data":                  config.get("data",  {}),
        "sft_eval_status":       "pending",
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    return run_dir


# ---------------------------------------------------------------------------
# Build train_sft.py command
# ---------------------------------------------------------------------------

def _build_cmd(config: dict, run_dir: Path) -> tuple[list[str], Path]:
    t    = config.get("train", {})
    data = config.get("data",  {})
    mod  = config.get("model", {})

    pretrained_path = _resolve_checkpoint(mod.get("pretrained_checkpoint", "auto"))

    cmd_args = [
        "--pretrained-checkpoint", str(pretrained_path),
        "--shards-dir",            data.get("shards_dir", "data/shards_sft"),
        "--seed",                  str(t.get("seed",            1337)),
        "--batch-size",            str(t.get("batch_size",        64)),
        "--micro-batch-size",      str(t.get("micro_batch_size",   8)),
        "--max-lr",                str(t.get("max_lr",           3e-4)),
        "--min-lr-ratio",          str(t.get("min_lr_ratio",      0.1)),
        "--warmup-steps",          str(t.get("warmup_steps",       50)),
        "--epochs",                str(t.get("epochs",              3)),
        "--weight-decay",          str(t.get("weight_decay",      0.1)),
        "--eval-interval",         str(t.get("eval_interval",      50)),
        "--log-dir",               str(run_dir),
    ]
    if t.get("max_steps", 0):
        cmd_args += ["--max-steps", str(t["max_steps"])]

    train_py = [str(REPO_ROOT / "train_sft.py")] + cmd_args
    return [sys.executable] + train_py, pretrained_path


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
        "total_steps":       metrics[-1]["step"] if metrics else 0,
        "final_train_loss":  by_kind["train"][-1] if "train" in by_kind else None,
        "final_val_loss":    by_kind["val"][-1]   if "val"   in by_kind else None,
        "sft_eval_status":   "pending",
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Post-training sft_eval
# ---------------------------------------------------------------------------

def _try_sft_eval(run_dir: Path, config: dict) -> str:
    ckpts = sorted(run_dir.glob("model_*.pt"))
    if not ckpts:
        return "skipped — no checkpoint found"

    tok      = config.get("tokenizer", {})
    tok_model = tok.get("model_file", "tokenizer/artifacts/mgpt2.model")
    shards_dir = config.get("data", {}).get("shards_dir", "data/shards_sft")

    try:
        subprocess.check_call([
            PYTHON, "-m", "eval.sft_eval",
            "--checkpoint",      str(ckpts[-1]),
            "--tokenizer-model", str(REPO_ROOT / tok_model),
            "--shards-dir",      shards_dir,
            "--device",          "cuda",
            "--out",             str(run_dir / "sft_eval.json"),
        ], cwd=REPO_ROOT)
        return "complete"
    except subprocess.CalledProcessError:
        return "failed — see sft_eval output"
    except Exception as e:
        return f"skipped — {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args        = _parse_args()
    args.config = args.config.resolve()
    config      = yaml.safe_load(args.config.read_text())

    run_dir = _make_run_dir(config, args.config)
    print(f"run dir : {run_dir}")

    cmd, _ = _build_cmd(config, run_dir)
    print("+", " ".join(cmd))

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    _post_process(run_dir)

    sft_status = _try_sft_eval(run_dir, config)
    for fname in ("run_info.json", "final_metrics.json"):
        p = run_dir / fname
        if p.exists():
            data = json.loads(p.read_text())
            data["sft_eval_status"] = sft_status
            p.write_text(json.dumps(data, indent=2))

    if result.returncode != 0:
        print(f"train_sft.py exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"done — results in {run_dir}")


if __name__ == "__main__":
    main()
