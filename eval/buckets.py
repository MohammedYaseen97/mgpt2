"""
Simple script-based bucketing used for reporting.

These buckets are intentionally minimal; adjust only if you can justify it
and you update all reports accordingly.
"""

from __future__ import annotations


def bucket(text: str) -> str:
    # Devanagari: U+0900..U+097F, Kannada: U+0C80..U+0CFF
    has_deva = any("\u0900" <= ch <= "\u097f" for ch in text)
    has_knda = any("\u0c80" <= ch <= "\u0cff" for ch in text)
    has_latin = any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)
    flags = (has_latin, has_deva, has_knda)
    if flags == (True, False, False):
        return "latin"
    if flags == (False, True, False):
        return "devanagari"
    if flags == (False, False, True):
        return "kannada"
    if flags == (False, False, False):
        return "other"
    return "mixed"

