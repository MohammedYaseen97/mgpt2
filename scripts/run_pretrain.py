"""Orchestrate a pretraining run from a YAML config file.

Single-GPU:
    python scripts/run_pretrain.py --config configs/pretrain_baseline.yaml

Multi-GPU (torchrun / DDP):
    python scripts/run_pretrain.py --config configs/pretrain_mgpt2.yaml --nproc 4

Creates:
    runs/{name}_{timestamp}/
        config.yaml         verbatim copy of input config
        run_info.json       git hash, tokenizer sha256, hyperparams
        log.txt             step-level metrics written by train.py
        metrics.jsonl       one JSON object per logged step
        final_metrics.json  end-of-run summary
        model_*.pt          checkpoints written by train.py
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

# maps config tokenizer kind strings → train.py --tokenizer-kind choices
_KIND_MAP = {
    "gpt2_tiktoken":  "gpt2",
    "mgpt2_regex_bpe": "mgpt2",
    "gpt2":           "gpt2",
    "mgpt2":          "mgpt2",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pretraining from a YAML config.")
    p.add_argument("--config", type=Path, required=True, help="Path to pretrain_*.yaml")
    p.add_argument("--nproc",  type=int,  default=1,    help="GPUs to use (>1 → torchrun)")
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


def _make_run_dir(config: dict, config_path: Path) -> Path:
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / f"{config['name']}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, run_dir / "config.yaml")

    tok = config.get("tokenizer", {})
    tok_info: dict = {"kind": tok.get("kind")}
    if tok.get("model_file"):
        model_path = REPO_ROOT / tok["model_file"]
        if model_path.exists():
            tok_info["path"]   = tok["model_file"]
            tok_info["sha256"] = _sha256(model_path)

    run_info = {
        "name":        f"{config['name']}_{ts}",
        "config_file": str(config_path.relative_to(REPO_ROOT)),
        "git_hash":    _git_hash(),
        "timestamp":   ts,
        "tokenizer":   tok_info,
        "train":       config.get("train", {}),
        "data":        config.get("data",  {}),
        # HellaSwag is run for both tokenizers.
        # render_example() receives the model's own enc so token IDs always
        # map to the correct embeddings.  This gives a valid apples-to-apples
        # comparison: does multilingual training hurt English performance?
        "hellaswag_note": "HellaSwag uses each model's own tokenizer via render_example(enc=enc)",
        "lm_eval_status": "pending",
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    return run_dir


def _derive_max_steps(shards_dir: str, total_batch_size: int) -> int | None:
    """Read total_tokens from the shard manifest and compute one-epoch step count."""
    manifest_path = REPO_ROOT / shards_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    total_tokens = manifest.get("total_tokens")
    if not total_tokens:
        return None
    return total_tokens // total_batch_size


def _build_cmd(config: dict, run_dir: Path, nproc: int) -> list[str]:
    t    = config.get("train",     {})
    tok  = config.get("tokenizer", {})
    data = config.get("data",      {})

    tok_kind        = _KIND_MAP.get(tok.get("kind", "gpt2"), "gpt2")
    shards_dir      = data.get("shards_dir", "data/shards_gpt2")
    total_batch_size = t.get("total_batch_size", 524288)

    # max_steps: explicit config wins; otherwise derive from shard manifest (one epoch)
    max_steps = t.get("max_steps")
    if max_steps is None:
        max_steps = _derive_max_steps(shards_dir, total_batch_size)
        if max_steps is not None:
            print(f"max_steps derived from manifest: {max_steps} "
                  f"(= {total_batch_size} tokens/step × {max_steps} steps)")
        else:
            max_steps = 19073  # Karpathy default; replace once shards exist
            print(f"WARNING: shard manifest not found at {shards_dir}/manifest.json — "
                  f"falling back to max_steps={max_steps}")

    train_args = [
        "--shards-dir",        shards_dir,
        "--seed",              str(t.get("seed",             1337)),
        "--total-batch-size",  str(total_batch_size),
        "--micro-batch-size",  str(t.get("micro_batch_size", 16)),
        "--max-lr",            str(t.get("max_lr",           3e-3)),
        "--min-lr-ratio",      str(t.get("min_lr_ratio",     0.1)),
        "--warmup-steps",      str(t.get("warmup_steps",     715)),
        "--max-steps",         str(max_steps),
        "--weight-decay",      str(t.get("weight_decay",     0.1)),
        "--eval-interval",          str(t.get("eval_interval",          250)),
        "--hellaswag-max-examples", str(t.get("hellaswag_max_examples", 0)),
        "--log-dir",           str(run_dir),
        "--tokenizer-kind",    tok_kind,
    ]
    if tok.get("model_file"):
        train_args += ["--tokenizer-model", tok["model_file"]]

    train_py = [str(REPO_ROOT / "train.py")] + train_args

    if nproc > 1:
        return ["torchrun", "--standalone", f"--nproc_per_node={nproc}"] + train_py
    return [sys.executable] + train_py


def _post_process(run_dir: Path) -> None:
    """Parse log.txt → metrics.jsonl + final_metrics.json."""
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
        "total_steps":         metrics[-1]["step"] if metrics else 0,
        "final_train_loss":    by_kind["train"][-1] if "train" in by_kind else None,
        "final_val_loss":      by_kind["val"][-1]   if "val"   in by_kind else None,
        "final_hellaswag_acc": by_kind["hella"][-1] if "hella" in by_kind else None,
        "lm_eval_status":      "pending",
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2))


def _try_lm_eval(run_dir: Path, config: dict) -> str:
    """Run eval/lm_eval.py against the bucketed eval manifest. Skips gracefully if absent."""
    eval_manifest = config.get("eval", {}).get("eval_manifest")
    if not eval_manifest or not (REPO_ROOT / eval_manifest).exists():
        return "skipped — eval_manifest not found"

    ckpts = sorted(run_dir.glob("model_*.pt"))
    if not ckpts:
        return "skipped — no checkpoint found"

    tok      = config.get("tokenizer", {})
    tok_args = ["--tokenizer-kind", _KIND_MAP.get(tok.get("kind", "gpt2"), "gpt2")]
    if tok.get("model_file"):
        tok_args += ["--tokenizer-model", tok["model_file"]]

    try:
        subprocess.check_call([
            PYTHON, "-m", "eval.lm_eval",
            "--checkpoint",    str(ckpts[-1]),
            "--eval-manifest", str(REPO_ROOT / eval_manifest),
            "--device",        "cuda",
            "--out",           str(run_dir / "lm_eval.json"),
            *tok_args,
        ], cwd=REPO_ROOT)
        return "complete"
    except subprocess.CalledProcessError:
        return "failed — see lm_eval output"
    except Exception as e:
        return f"skipped — {e}"


def main() -> None:
    args        = _parse_args()
    args.config = args.config.resolve()
    config      = yaml.safe_load(args.config.read_text())

    run_dir = _make_run_dir(config, args.config)
    print(f"run dir : {run_dir}")

    cmd = _build_cmd(config, run_dir, args.nproc)
    print("+", " ".join(cmd))

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    _post_process(run_dir)

    lm_status = _try_lm_eval(run_dir, config)
    for fname in ("run_info.json", "final_metrics.json"):
        p = run_dir / fname
        if p.exists():
            data = json.loads(p.read_text())
            data["lm_eval_status"] = lm_status
            p.write_text(json.dumps(data, indent=2))

    if result.returncode != 0:
        print(f"train.py exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"done — results in {run_dir}")


if __name__ == "__main__":
    main()
