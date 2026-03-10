#!/usr/bin/env python3
"""Filter unused TPU dump files in an LLO folder.

Modes:
1) Manual (`--kernel`): keep files matching kernel-name patterns.
2) Auto (default): resolve TLP inlined_call sites to exact kernel IDs and
   keep only those used LLO/HLO artifacts.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tpu.parser import BundleParser

_TLP_FILE_RE = re.compile(r"^(?P<id>\d+)-79-final_bundles\.txt$")
_KERNEL_FINAL_RE = re.compile(r"^(?P<id>\d+)-(?P<kernel>.+)-(?P<stage>\d+)-final_bundles\.txt$")
_ENTRY_SIG_RE = re.compile(r"ENTRY\s+%\S+\s*\((?P<params>.*)\)\s*->\s*(?P<out>\S+)")
_SHAPE_RE = re.compile(r"([A-Za-z0-9_]+)\[([0-9,\s]*)\]")

_DTYPE_BYTES: dict[str, int] = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "s32": 4,
    "u32": 4,
    "s16": 2,
    "u16": 2,
    "s8": 1,
    "u8": 1,
}
_BUNDLE_SLOT_FIELDS = (
    "mxu0", "mxu1", "mxu2", "mxu3",
    "xlu0", "xlu1", "xlu2",
    "valu0", "valu1", "valu2", "valu3",
    "eup",
    "load0", "load1", "load2",
    "store0",
    "salu0", "salu1",
)


def _iter_valid_slots(bundle):
    return bundle.iter_valid_slots()


@dataclass(frozen=True)
class KernelCandidate:
    kernel_id: int
    kernel_name: str
    stem: str  # "{id}-{kernel_name}"
    operand_count: int
    output_operand_nbytes: tuple[int, ...]


def _canonical_shape(raw_shape: str) -> str:
    m = _SHAPE_RE.search(raw_shape)
    if m is None:
        return ""
    dtype = m.group(1).lower()
    dims = ",".join(part.strip() for part in m.group(2).split(",") if part.strip())
    return f"{dtype}[{dims}]"


def _shape_nbytes(shape: str | None) -> int:
    if not shape:
        return 0
    m = _SHAPE_RE.search(shape)
    if m is None:
        return 0
    dtype = m.group(1).lower()
    elem_size = _DTYPE_BYTES.get(dtype, 0)
    if elem_size == 0:
        return 0
    dims_text = m.group(2).strip()
    if not dims_text:
        return elem_size
    count = 1
    for part in dims_text.split(","):
        part = part.strip()
        if part:
            count *= int(part)
    return count * elem_size


def _parse_hlo_output_nbytes(hlo_path: Path) -> int:
    if not hlo_path.exists():
        return 0
    entry_line = ""
    with hlo_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("ENTRY "):
                entry_line = line.strip()
                break
    if not entry_line:
        return 0
    m = _ENTRY_SIG_RE.search(entry_line)
    if m is None:
        return 0
    return _shape_nbytes(_canonical_shape(m.group("out")))


def _count_call_args(args: list[object]) -> int:
    return len(args)


def _kernel_pattern_to_regex(pattern: str) -> str:
    parts: list[str] = []
    for ch in pattern:
        if ch == "*":
            parts.append(".*")
        elif ch == "?":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def _build_kernel_patterns(kernel_patterns: list[str]) -> list[re.Pattern[str]]:
    keep: list[re.Pattern[str]] = []
    for pattern in kernel_patterns:
        kernel_regex = _kernel_pattern_to_regex(pattern)
        keep.extend(
            [
                re.compile(rf"^\d+-{kernel_regex}-\d+-original\.txt$"),
                re.compile(rf"^\d+-{kernel_regex}-\d+-post-delay-converter\.txt$"),
                re.compile(rf"^\d+-{kernel_regex}-\d+-final_bundles\.txt$"),
            ]
        )
    return keep


def _discover_tlp_ids(folder: Path) -> list[int]:
    tlp_ids: list[int] = []
    for path in sorted(folder.glob("*-79-final_bundles.txt")):
        m = _TLP_FILE_RE.match(path.name)
        if m:
            tlp_ids.append(int(m.group("id")))
    return tlp_ids


def _build_tlp_patterns(tlp_ids: list[int]) -> list[re.Pattern[str]]:
    keep: list[re.Pattern[str]] = []
    for tlp_id in tlp_ids:
        tid = re.escape(str(tlp_id))
        keep.extend(
            [
                re.compile(rf"^{tid}-79-final_bundles\.txt$"),
                re.compile(rf"^{tid}-\d+-post-delay-converter\.txt$"),
                re.compile(rf"^{tid}-hlo\.txt$"),
            ]
        )
    return keep


def should_keep(path: Path, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.match(path.name) for p in patterns)


def filter_folder(
    folder: Path,
    *,
    keep_names: set[str] | None = None,
    patterns: list[re.Pattern[str]] | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    keep_names = keep_names or set()
    patterns = patterns or []

    kept = 0
    removed = 0

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if (path.name in keep_names) or should_keep(path, patterns):
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


def _collect_kernel_metadata(llo_dir: Path, parser: BundleParser) -> dict[str, list[KernelCandidate]]:
    by_name: dict[str, list[KernelCandidate]] = {}
    seen_stems: set[str] = set()
    for path in sorted(llo_dir.glob("*-final_bundles.txt")):
        m = _KERNEL_FINAL_RE.match(path.name)
        if m is None:
            continue
        kernel_id = int(m.group("id"))
        kernel_name = m.group("kernel")
        stage = int(m.group("stage"))
        if stage == 79:
            continue
        stem = f"{kernel_id}-{kernel_name}"
        if stem in seen_stems:
            continue
        seen_stems.add(stem)

        symbol_table, bundles = parser.parse_program(llo_dir / stem)
        operand_kinds: dict[int, str] = {}
        operand_sizes: dict[int, int] = {}
        for bundle in bundles.values():
            for instr in _iter_valid_slots(bundle):
                opcode = str(instr.opcode)
                if not opcode.startswith("inlined_call_operand."):
                    continue
                idx = instr.operand_index
                if idx < 0:
                    continue
                operand_kinds[idx] = str(instr.operand_kind)
                sym = symbol_table.get(f"#operand{idx}")
                if sym is not None:
                    operand_sizes[idx] = int(sym.size)

        operand_count = (max(operand_kinds.keys()) + 1) if operand_kinds else 0
        output_indices = [i for i, kind in sorted(operand_kinds.items()) if kind == "output"]
        if not output_indices and operand_count > 0:
            output_indices = [operand_count - 1]
        output_sizes = tuple(operand_sizes.get(i, 0) for i in output_indices)

        by_name.setdefault(kernel_name, []).append(
            KernelCandidate(
                kernel_id=kernel_id,
                kernel_name=kernel_name,
                stem=stem,
                operand_count=operand_count,
                output_operand_nbytes=output_sizes,
            )
        )

    return by_name


def _resolve_kernel_candidate(
    *,
    tlp_id: int,
    callee: str,
    arg_count: int,
    expected_output_nbytes: int | None,
    candidates_by_name: dict[str, list[KernelCandidate]],
) -> KernelCandidate:
    candidates = list(candidates_by_name.get(callee, []))
    if not candidates:
        raise KeyError(f"No kernel candidate found for callee '{callee}'")

    filtered = [c for c in candidates if c.operand_count in (0, arg_count)]
    if filtered:
        candidates = filtered

    if expected_output_nbytes is not None:
        filtered = [
            c for c in candidates
            if c.output_operand_nbytes and c.output_operand_nbytes[0] == expected_output_nbytes
        ]
        if filtered:
            candidates = filtered

    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda c: abs(c.kernel_id - tlp_id))


def _discover_used_files_from_tlp(llo_dir: Path) -> set[str]:
    parser = BundleParser()
    tlp_ids = _discover_tlp_ids(llo_dir)
    if not tlp_ids:
        raise ValueError(f"No TLP programs (*-79-final_bundles.txt) found in {llo_dir}")

    kernel_candidates_by_name = _collect_kernel_metadata(llo_dir, parser)
    used_file_names: set[str] = set()

    for tlp_id in sorted(tlp_ids):
        tlp_stem = llo_dir / str(tlp_id)
        _symbol_table, bundles = parser.parse_program(tlp_stem)
        call_instructions = []
        for bundle in bundles.values():
            for instr in _iter_valid_slots(bundle):
                if str(instr.opcode) == "inlined_call":
                    call_instructions.append(instr)

        expected_output_nbytes = _parse_hlo_output_nbytes(llo_dir / f"{tlp_id}-hlo.txt")
        total_calls = len(call_instructions)
        for i, instr in enumerate(call_instructions):
            callee = str(instr.callee)
            arg_count = _count_call_args(instr.call_args)
            expected = expected_output_nbytes if i == total_calls - 1 else None
            resolved = _resolve_kernel_candidate(
                tlp_id=tlp_id,
                callee=callee,
                arg_count=arg_count,
                expected_output_nbytes=expected,
                candidates_by_name=kernel_candidates_by_name,
            )

            kernel_id_text = str(resolved.kernel_id)
            used_file_names.add(f"{kernel_id_text}-hlo.txt")
            for p in llo_dir.glob(f"{resolved.stem}-*.txt"):
                if re.match(
                    rf"^{re.escape(resolved.stem)}-\d+-(original|post-delay-converter|final_bundles)\.txt$",
                    p.name,
                ):
                    used_file_names.add(p.name)

        tlp_id_text = str(tlp_id)
        used_file_names.add(f"{tlp_id_text}-79-final_bundles.txt")
        used_file_names.add(f"{tlp_id_text}-hlo.txt")
        for p in llo_dir.glob(f"{tlp_id_text}-*-post-delay-converter.txt"):
            used_file_names.add(p.name)

    return used_file_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove unused LLO/HLO artifacts from an LLO folder. "
            "If --kernel is omitted, kernels are auto-resolved from TLP inlined_call instructions."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder path (e.g. tests/vector_add_bf16/tpu_compiler_dump/llo)",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        metavar="NAME",
        action="append",
        help=(
            "Kernel name pattern to keep (supports * and ?). Repeatable. "
            "If omitted, kernels are resolved automatically."
        ),
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    llo_dir = folder / "llo" if (folder / "llo").is_dir() else folder
    kernels = list(args.kernel or [])
    tlp_ids = _discover_tlp_ids(llo_dir)

    patterns: list[re.Pattern[str]] = []
    keep_names: set[str] = set()
    if kernels:
        patterns.extend(_build_kernel_patterns(kernels))
        patterns.extend(_build_tlp_patterns(tlp_ids))
    else:
        keep_names = _discover_used_files_from_tlp(llo_dir)
        kernel_files = sorted(
            name for name in keep_names
            if re.match(r"^\d+-.+-\d+-final_bundles\.txt$", name)
            and not _TLP_FILE_RE.match(name)
        )
        print(f"Auto-resolved used kernels: {len(kernel_files)} final_bundles files")

    if args.dry_run:
        print("Dry run (no files will be deleted):\n")

    kept, removed = filter_folder(
        llo_dir,
        keep_names=keep_names,
        patterns=patterns,
        dry_run=args.dry_run,
    )
    print(f"\nKept: {kept}, Removed: {removed}")


if __name__ == "__main__":
    main()
