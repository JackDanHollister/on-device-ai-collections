#!/usr/bin/env python3
"""Check the integrity of the included original source files."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "provenance/source-files.json").read_text())
    for item in manifest["source_files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Source path escapes repository")
        raw = path.read_bytes()
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Source hash mismatch: " + item["path"])
    print(json.dumps({"ok": True, "copied_source_files_verified": len(manifest["source_files"]),
                      "raw_museum_records_included": False,
                      "scope": "Integrity of included source files."}, indent=2))


if __name__ == "__main__":
    main()
