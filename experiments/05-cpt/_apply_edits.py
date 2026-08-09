"""Generic one-shot: apply an implement-agent's edits (+review fixes) to the repo.

Usage:
    python experiments/05-cpt/_apply_edits.py <journal.jsonl> [--fixes journal.jsonl]
Reads the FIRST result line with 'edits' (implement agent) and applies them.
If --fixes is given, also reads the result with 'fixes' (review agent) and
applies each fix block (json list of edits).

Strictly verifies each old_text occurs exactly once before replacing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def apply_edits(edits, tag: str) -> int:
    n = 0
    for e in edits:
        path = ROOT / e["file"]
        if not path.exists():
            raise SystemExit(f"FAIL {tag}: missing file {e['file']}")
        text = path.read_text(encoding="utf-8")
        old, new = e["old_text"], e["new_text"]
        cnt = text.count(old)
        if cnt != 1:
            raise SystemExit(
                f"FAIL {tag} [{e['file']}] @ {e.get('anchor','?')}: old_text occurs "
                f"{cnt} times (expected 1). old_len={len(old)}"
            )
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        n += 1
        print(f"  applied [{e['file']}] {e.get('anchor','?')}")
    return n


def main() -> int:
    journal = Path(sys.argv[1])
    fixes_journal = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--fixes" else None
    lines = journal.read_text(encoding="utf-8").splitlines()
    implement = None
    for line in lines:
        rec = json.loads(line)
        if rec.get("type") == "result" and "edits" in rec.get("result", {}):
            implement = rec["result"]
            break
    if implement is None:
        raise SystemExit("no implement result found")
    n_edits = apply_edits(implement["edits"], "implement")
    for f in implement.get("new_files", []):
        path = ROOT / f["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f["content"], encoding="utf-8")
        print(f"  wrote {f['path']} ({len(f['content'])} chars)")

    n_fixes = 0
    if fixes_journal:
        for line in fixes_journal.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("type") == "result" and "fixes" in rec.get("result", {}):
                fixes = rec["result"]["fixes"]
                if fixes.strip():
                    blocks = json.loads(fixes)  # list of edit dicts
                    n_fixes = apply_edits(blocks, "fixes")
                break
    print(f"\nDONE: {n_edits} edits + {len(implement.get('new_files', []))} files + {n_fixes} fixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
