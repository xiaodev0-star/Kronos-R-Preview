"""One-shot: apply Branch B edits from the workflow journal to the repo files.

Strictly verifies each old_text occurs exactly once in the target file before
replacing; fails loudly on any mismatch. Writes new_files verbatim.

Usage: python experiments/05-cpt/_apply_branchB_edits.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = Path(r"C:\Users\XiaoD\.claude\projects\D--Kronos-R-Preview\3bb10ea3-2995-49ad-b6c0-74f18f0f19b9\subagents\workflows\wf_f0b04982-510\journal.jsonl")


def main() -> int:
    lines = JOURNAL.read_text(encoding="utf-8").splitlines()
    impl = json.loads(lines[1])["result"]  # line 1 = implement agent result
    edits = impl["edits"]
    new_files = impl.get("new_files", [])

    n_applied = 0
    for e in edits:
        path = ROOT / e["file"]
        text = path.read_text(encoding="utf-8")
        old, new = e["old_text"], e["new_text"]
        cnt = text.count(old)
        if cnt != 1:
            raise SystemExit(
                f"FAIL {e['file']} @ {e['anchor']}: old_text occurs {cnt} times "
                f"(expected 1). old_len={len(old)} new_len={len(new)}"
            )
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        n_applied += 1
        print(f"  applied [{e['file']}] {e['anchor']}")

    for f in new_files:
        path = ROOT / f["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f["content"], encoding="utf-8")
        print(f"  wrote new file {f['path']} ({len(f['content'])} chars)")

    print(f"\nDONE: {n_applied}/{len(edits)} edits applied, {len(new_files)} new files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
