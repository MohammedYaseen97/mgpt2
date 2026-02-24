"""
TODO: Orchestrate pretraining runs from config files.

Rules:
- Must not change `train.py` logic unless you explicitly choose to.
- Must write a run directory with:
  - copied config
  - git hash
  - tokenizer identity
  - metrics outputs
  - checkpoints

Suggested approach:
- parse YAML in `configs/pretrain_*.yaml`
- set env vars / arguments for `train.py`
- call `python train.py` as a subprocess
"""

raise NotImplementedError("TODO: implement run_pretrain.py")

