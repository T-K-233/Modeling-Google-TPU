"""
Example script: run the bundle parser on a program and print the parsed symbol table
and instruction bundles.
"""
import argparse
import re
from pathlib import Path

from tpu.parser import BundleParser
from tpu.instruction import InstructionBundle

# Matches kernel bundle files: {id}-{kernel_name}-{stage}-final_bundles.txt
_KERNEL_FINAL_RE = re.compile(r"^(?P<id>\d+)-(?P<kernel>.+)-(?P<stage>\d+)-final_bundles\.txt$")
# Matches TLP bundle files: {id}-79-final_bundles.txt
_TLP_FINAL_RE = re.compile(r"^(?P<id>\d+)-79-final_bundles\.txt$")


def resolve_llo_dir(path: Path) -> Path:
    """Resolve to the directory containing *-final_bundles.txt. Implicitly appends tpu_compiler_dump/llo if needed."""
    path = Path(path)
    if list(path.glob("*-final_bundles.txt")):
        return path
    fallback = path / "tpu_compiler_dump" / "llo"
    if fallback.is_dir() and list(fallback.glob("*-final_bundles.txt")):
        return fallback
    return path


def _derive_stem_from_bundle_file(path: Path) -> Path:
    """Given a *-final_bundles.txt path, return the partial_path (stem) for the parser."""
    name = path.name
    if name.endswith("-final_bundles.txt"):
        stem_name = name[: -len("-final_bundles.txt")]
        # kernel: 1772821898188315897-convolution_add_fusion-78 -> 1772821898188315897-convolution_add_fusion
        # tlp:    1772821898168098007-79 -> 1772821898168098007
        if _KERNEL_FINAL_RE.match(name):
            stem_name = stem_name.rsplit("-", 1)[0]  # drop stage
        elif _TLP_FINAL_RE.match(name):
            stem_name = stem_name.rsplit("-", 1)[0]  # drop "79"
        return path.parent / stem_name
    return path


def find_kernel_stem(llo_dir: Path, kernel_name: str) -> Path:
    """Find the first *-{kernel_name}-*-final_bundles.txt and return its stem path."""
    candidates = sorted(llo_dir.glob(f"*-{kernel_name}-*-final_bundles.txt"))
    if not candidates:
        raise FileNotFoundError(f"No kernel matching '*-{kernel_name}-*-final_bundles.txt' in {llo_dir}")
    return _derive_stem_from_bundle_file(candidates[0])


def collect_all_stems(llo_dir: Path) -> list[tuple[str, Path]]:
    """Collect (label, partial_path) for every *-final_bundles.txt in llo_dir. Deduplicated by stem."""
    seen: set[Path] = set()
    result: list[tuple[str, Path]] = []
    for path in sorted(llo_dir.glob("*-final_bundles.txt")):
        stem_path = _derive_stem_from_bundle_file(path)
        if stem_path in seen:
            continue
        seen.add(stem_path)
        label = stem_path.name
        result.append((label, stem_path))
    return result


def list_available(llo_dir: Path) -> None:
    """Print kernels and TLP stems found in llo_dir."""
    kernels: set[str] = set()
    tlp_ids: list[int] = []
    for path in sorted(llo_dir.glob("*-final_bundles.txt")):
        m = _KERNEL_FINAL_RE.match(path.name)
        if m:
            kernels.add(m.group("kernel"))
            continue
        m = _TLP_FINAL_RE.match(path.name)
        if m:
            tlp_ids.append(int(m.group("id")))
    print("Kernels (use --kernel NAME):")
    for k in sorted(kernels):
        print(f"  {k}")
    print("TLP stems (use --tlp-id ID; partial_path = <llo_dir>/<id>):")
    for i in sorted(tlp_ids):
        print(f"  {i}")


def main() -> None:
    default_llo = Path(__file__).resolve().parent.parent / "tests/linear_f32/tpu_compiler_dump/llo"

    parser = argparse.ArgumentParser(
        description="Parse a TPU LLO program and print symbol table and instruction bundles.",
    )
    parser.add_argument(
        "-d",
        "--llo-dir",
        type=Path,
        default=default_llo,
        help=f"LLO directory containing *-final_bundles.txt (default: %(default)s)",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        metavar="NAME",
        help="Kernel name to parse (e.g. convolution_add_fusion, copy.1). Finds *-NAME-*-final_bundles.txt",
    )
    parser.add_argument(
        "--tlp-id",
        type=int,
        metavar="ID",
        help="TLP numeric id to parse (e.g. 79). Uses <llo_dir>/<id> as stem.",
    )
    parser.add_argument(
        "-s",
        "--stem",
        type=str,
        metavar="STEM",
        help="Explicit stem (e.g. 1772821898188315897-convolution_add_fusion). Overrides --kernel/--tlp-id.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available kernel names and TLP ids in --llo-dir, then exit.",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="load_entire_program",
        help="Parse entire program: every kernel and TLP in --llo-dir.",
    )
    args = parser.parse_args()

    llo_dir = Path(args.llo_dir)
    if not llo_dir.is_dir():
        parser.error(f"Not a directory: {llo_dir}")
    llo_dir = resolve_llo_dir(llo_dir)

    if args.list:
        list_available(llo_dir)
        return

    if args.load_entire_program:
        stems = collect_all_stems(llo_dir)
        if not stems:
            parser.error(f"No *-final_bundles.txt found in {llo_dir}")
        bundle_parser = BundleParser()
        for label, partial_path in stems:
            print("\n" + "=" * 60)
            print(f"Program: {label}")
            print("=" * 60)
            try:
                symbol_table, bundles = bundle_parser.parse_program(partial_path)
            except FileNotFoundError as e:
                print(f"  Skip: {e}")
                continue
            print("\n--- Symbol table (allocations) ---")
            for alloc_id, alloc in symbol_table.items():
                print(f"  {alloc_id}: space={alloc.space} size=0x{alloc.size:x} base_address=0x{alloc.base_address:x}")
            print("\n--- Parsed bundles ---")
            for addr in sorted(bundles.keys()):
                bundle = bundles[addr]
                print(f"\n0x{addr:x} ({len(bundle.instructions)} instructions)")
                print(bundle)
        return

    if args.stem:
        partial_path = llo_dir / args.stem
    elif args.kernel:
        partial_path = find_kernel_stem(llo_dir, args.kernel)
    elif args.tlp_id is not None:
        partial_path = llo_dir / str(args.tlp_id)
    else:
        # Default to convolution_add_fusion only when using default llo_dir
        if llo_dir.resolve() != default_llo.resolve():
            parser.error(
                "No program selected. Specify --kernel NAME, --tlp-id ID, --stem STEM, or --all to parse entire program."
            )
        partial_path = find_kernel_stem(llo_dir, "convolution_add_fusion")

    bundle_parser = BundleParser()
    symbol_table, bundles = bundle_parser.parse_program(partial_path)

    print("=== Symbol table (allocations) ===")
    for alloc_id, alloc in symbol_table.items():
        print(f"  {alloc_id}: space={alloc.space} size=0x{alloc.size:x} base_address=0x{alloc.base_address:x}")

    print("\n=== Parsed bundles ===")
    for addr in sorted(bundles.keys()):
        bundle: InstructionBundle = bundles[addr]
        print(f"\n0x{addr:x} ({len(bundle.instructions)} instructions)")
        print(bundle)


if __name__ == "__main__":
    main()
