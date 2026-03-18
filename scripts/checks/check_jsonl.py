"""Phase B check — raw JSONL data integrity.

Verifies the output of build_sft_data.py and build_dpo_data.py.
Complements check_shards.py (which covers tokenized arrays); this
script checks the human-readable JSONL files and their manifests.

Checks
------
SFT  (data/sft/)
   - manifest present with required keys
   - line count matches manifest n_train + n_val
   - every line: valid JSON, required fields present (prompt, response, lang)
   - no empty prompt or response strings
   - lang values drawn from the expected set
   - language counts in train+val match manifest lang_counts

DPO  (data/dpo/)
   - manifest present with required keys
   - line count matches manifest n_train + n_val
   - every line: valid JSON, required fields (prompt, chosen, rejected, lang)
   - no empty prompt or chosen strings
   - rejected == "" for every row  (deferred strategy — expected pre-Phase E)
   - manifest rejected_populated == False  (guard for tokenize_dpo_shards.py)
   - lang values drawn from the expected set
   - language counts match manifest lang_counts

Exit code
   0  all checks passed
   1  one or more checks failed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXPECTED_LANGS = {"eng_Latn", "hin_Deva", "kan_Knda", "hin_Latn", "kan_Latn"}

# ---------------------------------------------------------------------------
# Harness (same pattern as check_shards.py)
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _chk(name: str, fn) -> bool:
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        msg = str(e) or "assertion failed"
        _results.append((name, False, msg))
        print(f"  FAIL  {name}  →  {msg}")
        return False
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        _results.append((name, False, msg))
        print(f"  FAIL  {name}  →  {msg}")
        return False

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    """Read all lines from a JSONL file. Asserts each line is valid JSON."""
    rows = []
    bad  = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad.append(f"line {i}: {e}")
    assert not bad, f"{len(bad)} invalid JSON lines (first: {bad[0]})"
    return rows


def _count_langs(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        lang = r.get("lang", "__missing__")
        counts[lang] = counts.get(lang, 0) + 1
    return counts

# ---------------------------------------------------------------------------
# SFT checks
# ---------------------------------------------------------------------------

def check_sft(sft_dir: Path) -> None:
    print(f"\n  [sft]")

    m_path = sft_dir / "manifest.json"

    def manifest_exists():
        assert m_path.exists(), f"manifest not found: {m_path}"
    if not _chk("sft/manifest_exists", manifest_exists):
        print("  SKIP  sft/* — cannot proceed without manifest")
        return

    m = json.loads(m_path.read_text(encoding="utf-8"))

    required_manifest_keys = {"n_train", "n_val", "lang_counts", "seed"}

    def manifest_keys():
        missing = required_manifest_keys - set(m.keys())
        assert not missing, f"manifest missing keys: {sorted(missing)}"
    _chk("sft/manifest_keys", manifest_keys)

    train_path = sft_dir / "train.jsonl"
    val_path   = sft_dir / "val.jsonl"

    def files_exist():
        assert train_path.exists(), f"missing: {train_path}"
        assert val_path.exists(),   f"missing: {val_path}"
    if not _chk("sft/files_exist", files_exist):
        print("  SKIP  sft/rows — files missing")
        return

    # load
    try:
        train_rows = _read_jsonl(train_path)
        val_rows   = _read_jsonl(val_path)
    except AssertionError as e:
        _chk("sft/json_valid", lambda: (_ for _ in ()).throw(e))
        return
    _chk("sft/json_valid", lambda: None)   # reached here → valid

    # line counts
    n_train_expect = m.get("n_train", -1)
    n_val_expect   = m.get("n_val",   -1)

    def line_count_train():
        assert len(train_rows) == n_train_expect, (
            f"train has {len(train_rows)} lines, manifest says {n_train_expect}"
        )
    def line_count_val():
        assert len(val_rows) == n_val_expect, (
            f"val has {len(val_rows)} lines, manifest says {n_val_expect}"
        )
    _chk("sft/line_count_train", line_count_train)
    _chk("sft/line_count_val",   line_count_val)

    all_rows = train_rows + val_rows

    # required fields
    def required_fields():
        bad = [i for i, r in enumerate(all_rows)
               if not all(k in r for k in ("prompt", "response", "lang"))]
        assert not bad, f"{len(bad)} rows missing fields (first idx: {bad[0]})"
    _chk("sft/required_fields", required_fields)

    # non-empty prompt + response
    def nonempty_prompt():
        bad = [i for i, r in enumerate(all_rows) if not r.get("prompt", "").strip()]
        assert not bad, f"{len(bad)} rows with empty prompt (first idx: {bad[0]})"
    def nonempty_response():
        bad = [i for i, r in enumerate(all_rows) if not r.get("response", "").strip()]
        assert not bad, f"{len(bad)} rows with empty response (first idx: {bad[0]})"
    _chk("sft/nonempty_prompt",   nonempty_prompt)
    _chk("sft/nonempty_response", nonempty_response)

    # lang values
    def valid_langs():
        bad = {r.get("lang") for r in all_rows} - EXPECTED_LANGS
        assert not bad, f"unexpected lang values: {bad}"
    _chk("sft/valid_langs", valid_langs)

    # lang counts match manifest
    def lang_counts_match():
        actual   = _count_langs(all_rows)
        expected = m.get("lang_counts", {})
        for lang, exp_count in expected.items():
            act_count = actual.get(lang, 0)
            assert act_count == exp_count, (
                f"lang {lang!r}: manifest says {exp_count}, found {act_count}"
            )
    _chk("sft/lang_counts_match", lang_counts_match)

# ---------------------------------------------------------------------------
# DPO checks
# ---------------------------------------------------------------------------

def check_dpo(dpo_dir: Path) -> None:
    print(f"\n  [dpo]")

    m_path = dpo_dir / "manifest.json"

    def manifest_exists():
        assert m_path.exists(), f"manifest not found: {m_path}"
    if not _chk("dpo/manifest_exists", manifest_exists):
        print("  SKIP  dpo/* — cannot proceed without manifest")
        return

    m = json.loads(m_path.read_text(encoding="utf-8"))

    required_manifest_keys = {"n_train", "n_val", "lang_counts", "seed", "rejected_populated"}

    def manifest_keys():
        missing = required_manifest_keys - set(m.keys())
        assert not missing, f"manifest missing keys: {sorted(missing)}"
    _chk("dpo/manifest_keys", manifest_keys)

    # rejected_populated must be False at this stage
    def rejected_flag():
        assert m.get("rejected_populated") is False, (
            f"rejected_populated={m.get('rejected_populated')!r}; "
            "expected False — run generate_dpo_rejected.py after Phase C"
        )
    _chk("dpo/rejected_populated_flag", rejected_flag)

    train_path = dpo_dir / "train.jsonl"
    val_path   = dpo_dir / "val.jsonl"

    def files_exist():
        assert train_path.exists(), f"missing: {train_path}"
        assert val_path.exists(),   f"missing: {val_path}"
    if not _chk("dpo/files_exist", files_exist):
        print("  SKIP  dpo/rows — files missing")
        return

    try:
        train_rows = _read_jsonl(train_path)
        val_rows   = _read_jsonl(val_path)
    except AssertionError as e:
        _chk("dpo/json_valid", lambda: (_ for _ in ()).throw(e))
        return
    _chk("dpo/json_valid", lambda: None)

    n_train_expect = m.get("n_train", -1)
    n_val_expect   = m.get("n_val",   -1)

    def line_count_train():
        assert len(train_rows) == n_train_expect, (
            f"train has {len(train_rows)} lines, manifest says {n_train_expect}"
        )
    def line_count_val():
        assert len(val_rows) == n_val_expect, (
            f"val has {len(val_rows)} lines, manifest says {n_val_expect}"
        )
    _chk("dpo/line_count_train", line_count_train)
    _chk("dpo/line_count_val",   line_count_val)

    all_rows = train_rows + val_rows

    # required fields
    def required_fields():
        bad = [i for i, r in enumerate(all_rows)
               if not all(k in r for k in ("prompt", "chosen", "rejected", "lang"))]
        assert not bad, f"{len(bad)} rows missing fields (first idx: {bad[0]})"
    _chk("dpo/required_fields", required_fields)

    # non-empty prompt + chosen
    def nonempty_prompt():
        bad = [i for i, r in enumerate(all_rows) if not r.get("prompt", "").strip()]
        assert not bad, f"{len(bad)} rows with empty prompt (first idx: {bad[0]})"
    def nonempty_chosen():
        bad = [i for i, r in enumerate(all_rows) if not r.get("chosen", "").strip()]
        assert not bad, f"{len(bad)} rows with empty chosen (first idx: {bad[0]})"
    _chk("dpo/nonempty_prompt", nonempty_prompt)
    _chk("dpo/nonempty_chosen", nonempty_chosen)

    # rejected must be "" for every row (deferred strategy)
    def rejected_empty():
        bad = [i for i, r in enumerate(all_rows) if r.get("rejected", "") != ""]
        assert not bad, (
            f"{len(bad)} rows have non-empty rejected — "
            "unexpected before generate_dpo_rejected.py runs"
        )
    _chk("dpo/rejected_is_empty", rejected_empty)

    # lang values
    def valid_langs():
        bad = {r.get("lang") for r in all_rows} - EXPECTED_LANGS
        assert not bad, f"unexpected lang values: {bad}"
    _chk("dpo/valid_langs", valid_langs)

    # lang counts match manifest
    def lang_counts_match():
        actual   = _count_langs(all_rows)
        expected = m.get("lang_counts", {})
        for lang, exp_count in expected.items():
            act_count = actual.get(lang, 0)
            assert act_count == exp_count, (
                f"lang {lang!r}: manifest says {exp_count}, found {act_count}"
            )
    _chk("dpo/lang_counts_match", lang_counts_match)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data = REPO_ROOT / "data"

    print("=" * 60)
    print("Phase B — JSONL data integrity checks")
    print("=" * 60)

    print("\n[1] SFT JSONL")
    check_sft(data / "sft")

    print("\n[2] DPO JSONL")
    check_dpo(data / "dpo")

    # summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print(f"Results: {passed} passed, {failed} failed  ({len(_results)} total)")

    if failed:
        print("\nFailed checks:")
        for name, ok, msg in _results:
            if not ok:
                print(f"  FAIL  {name}  →  {msg}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
