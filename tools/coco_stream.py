# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Streaming reader for COCO json files that are too large for ``json.load``.

The robobin2026-v1 train annotations are ~7 GB and expand to tens of GB of Python objects, so
loading one is not viable. :func:`iter_array` walks a single top-level array through a sliding
text window and decodes one element at a time, keeping peak memory flat regardless of file size.
"""

import json
from pathlib import Path

SEPARATORS = " \t\r\n,"


def find_top_level_key(buf: str, key: str) -> int:
    """Return the index of top-level ``"key"`` in buf, or -1. A top-level key follows ``{`` or ``,``."""
    needle = f'"{key}"'
    i = buf.find(needle)
    while i != -1:
        j = i - 1
        while j >= 0 and buf[j] in " \t\r\n":
            j -= 1
        if j >= 0 and buf[j] in "{,":
            return i
        i = buf.find(needle, i + 1)
    return -1


def iter_array(path: Path, key: str, chunk_size: int = 1 << 23):
    """Yield the items of the top-level JSON array ``key`` without loading the whole file."""
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        # Phase 1: slide a window forward until the top-level key shows up, then enter its array.
        buf = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                raise KeyError(f"top-level key {key!r} not found in {path}")
            buf = (buf[-64:] if len(buf) > 64 else buf) + chunk
            i = find_top_level_key(buf, key)
            if i != -1:
                break
        buf = buf[buf.index("[", i) + 1 :]

        # Phase 2: decode one element at a time, refilling only when an element is still incomplete.
        # The window is compacted in bulk rather than per element, so decoding an item carries no
        # O(chunk_size) string copy.
        pos = 0
        while True:
            while pos < len(buf) and buf[pos] in SEPARATORS:
                pos += 1
            if pos >= len(buf):
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buf = buf[pos:] + chunk
                pos = 0
                continue
            if buf[pos] == "]":
                return
            while True:
                try:
                    item, end = decoder.raw_decode(buf, pos)
                    break
                except ValueError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buf += chunk
            yield item
            pos = end
            if pos > chunk_size:
                buf = buf[pos:]
                pos = 0
