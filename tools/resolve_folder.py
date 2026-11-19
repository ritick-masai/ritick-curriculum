#!/usr/bin/env python3
"""Find the lecture folder for a session inside a batch.

Usage:
    python3 tools/resolve_folder.py <batch_id> "<session name>"

Prints the folder path (relative to the repo root), e.g.
    IITP-AIMLPX-ADV-2608/module-02-core-machine-learning-engineering/lecture-02-fitting-a-first-linear-regression

Folder names follow README.md: lowercase, hyphens only, numbered lectures.
The folder must already exist. This script never creates one.
Exit code 0 = found. Exit code 2 = not found or ambiguous (an ASK: line says why).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def slug(text: str) -> str:
    text = text.lower().replace("&", " ")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    batch_id, session = sys.argv[1], sys.argv[2]
    batch = ROOT / batch_id
    if not batch.is_dir():
        batches = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and re.match(r"^[A-Z0-9-]+$", p.name))
        print(f"ASK: batch folder '{batch_id}' not found. Batches: {', '.join(batches)}")
        return 2
    want = slug(session)
    hits = sorted(
        d for d in batch.glob("module-*/lecture-*")
        if d.is_dir() and re.sub(r"^lecture-\d+-", "", d.name) == want
    )
    if len(hits) != 1:
        found = "no lecture folder" if not hits else f"{len(hits)} lecture folders"
        print(f"ASK: {found} match '{want}' in {batch_id}. Check the session name against the curriculum CSV.")
        return 2
    print(hits[0].relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
