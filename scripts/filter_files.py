#!/usr/bin/env python3
"""
Filter files in a folder, keeping only LLO compiler dump files that match:
  - *-02-original.txt
  - *-75-post-delay-converter.txt
  - *-78-final_bundles.txt

All other files under the folder are removed.
"""

import argparse
import re
from pathlib import Path


# Filename patterns to KEEP (all others are removed)
KEEP_PATTERNS = (
    re.compile(r"^\d+-[\w.-]+-02-original\.txt$"),
    re.compile(r"^\d+-[\w.-]+-75-post-delay-converter\.txt$"),
    re.compile(r"^\d+-[\w.-]+-78-final_bundles\.txt$"),
)


def should_keep(path: Path) -> bool:
    """Return True if the file matches one of the keep patterns."""
    name = path.name
    return any(p.match(name) for p in KEEP_PATTERNS)


def filter_folder(folder: Path, dry_run: bool = False) -> tuple[int, int]:
    """Remove all files that don't match the keep patterns. Returns (kept, removed)."""
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    kept = 0
    removed = 0

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if should_keep(path):
            kept += 1
            if dry_run:
                print(f"  KEEP: {path.relative_to(folder)}")
        else:
            removed += 1
            if dry_run:
                print(f"  REMOVE: {path.relative_to(folder)}")
            else:
                path.unlink()
                print(f"  Removed: {path.relative_to(folder)}")

    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove unused LLO files, keeping only 02-original, 75-post-delay-converter, and 78-final_bundles."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder path to filter (e.g. tests/matmul_t/tpu_compiler_dump or tests/matmul_t/tpu_compiler_dump/llo)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run (no files will be deleted):\n")

    kept, removed = filter_folder(args.folder, dry_run=args.dry_run)

    print(f"\nKept: {kept}, Removed: {removed}")


if __name__ == "__main__":
    main()
