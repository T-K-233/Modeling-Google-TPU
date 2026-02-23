#!/usr/bin/env python3
"""
Filter files in a folder, keeping only LLO compiler dump files for the given kernel:
  - *-{kernel}-NN-original.txt
  - *-{kernel}-NN-post-delay-converter.txt
  - *-{kernel}-NN-final_bundles.txt

All other files under the folder are removed.
"""

import argparse
import re
from pathlib import Path


def build_keep_patterns(kernel: str) -> tuple[re.Pattern[str], ...]:
    """Build keep patterns. If kernel is given, restrict to that kernel and allow any NN before original."""
    escaped = re.escape(kernel)
    return (
        re.compile(rf"^\d+-{escaped}-\d+-original\.txt$"),
        re.compile(rf"^\d+-{escaped}-\d+-post-delay-converter\.txt$"),
        re.compile(rf"^\d+-{escaped}-\d+-final_bundles\.txt$"),
    )


def should_keep(path: Path, patterns: tuple[re.Pattern[str], ...]) -> bool:
    """Return True if the file matches one of the keep patterns."""
    name = path.name
    return any(p.match(name) for p in patterns)


def filter_folder(
    folder: Path, patterns: tuple[re.Pattern[str], ...], dry_run: bool = False
) -> tuple[int, int]:
    """Remove all files that don't match the keep patterns. Returns (kept, removed)."""
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    kept = 0
    removed = 0

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if should_keep(path, patterns):
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
        description="Remove unused LLO files, keeping only files for the given kernel (original, post-delay-converter, final_bundles)."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder path to filter (e.g. tests/matmul_t/tpu_compiler_dump or tests/matmul_t/tpu_compiler_dump/llo)",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        metavar="NAME",
        required=True,
        help="Kernel name to keep (e.g. reduce.7). Allows any NN before original, post-delay-converter, final_bundles.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting",
    )
    args = parser.parse_args()

    patterns = build_keep_patterns(args.kernel)

    if args.dry_run:
        print("Dry run (no files will be deleted):\n")

    kept, removed = filter_folder(args.folder, patterns, dry_run=args.dry_run)

    print(f"\nKept: {kept}, Removed: {removed}")


if __name__ == "__main__":
    main()
