"""
Run all assignment checks.

This intentionally fails until you complete TODO phases.
"""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    # Phase A: tokenizer must exist
    run([sys.executable, "-m", "scripts.checks.check_phase_a_tokenizer"])

    # Phase B/C/D/E checks are TODO by design.
    raise SystemExit(
        "Phase A passed. Phase B+ checks are TODO: implement them as part of the assignment."
    )


if __name__ == "__main__":
    main()

