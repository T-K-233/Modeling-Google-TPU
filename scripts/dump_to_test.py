#!/usr/bin/env python3
"""Convert TPU kernel dumps (zip files) to filtered test cases in tests/.

Each zip is extracted, LLO artifacts are filtered (auto-resolve from TLP inlined_call),
and a minimal test case is created under tests/<name>/ with:
  - tpu_compiler_dump/llo/  (filtered LLO files only)
  - source.py               (stub documenting the kernel)

Usage:
  python scripts/dump_to_test.py [--dry-run] [dumps/...]
  python scripts/dump_to_test.py dumps/matmul_bf16.zip dumps/softmax_f32.zip
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.filter_files import (
    _discover_used_files_from_tlp,
    _discover_tlp_ids,
    _TLP_FILE_RE,
)

# Test name -> source.py content (kernel name inferred from zip stem)
SOURCE_TEMPLATES: dict[str, str] = {
    "vector_add_bf16": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def vector_add_bf16(x: jnp.ndarray, y: jnp.ndarray):
    z = x + y
    return z

a = jnp.arange(8 * 128 * 2, dtype=jnp.bfloat16).reshape(8, 256)
b = jnp.arange(8 * 128 * 2, dtype=jnp.bfloat16).reshape(8, 256)

c = jax.jit(vector_add_bf16)(a, b)
print(c)

download_files("vector_add_bf16")
''',
    "vector_add_f32": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def vector_add_f32(x: jnp.ndarray, y: jnp.ndarray):
    z = x + y
    return z

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
b = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

c = jax.jit(vector_add_f32)(a, b)
print(c)

download_files("vector_add_f32")
''',
    "linear_f32": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def linear_f32(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray):
    y = jnp.matmul(x, w, preferred_element_type=jnp.float32)
    y = y + b
    return y

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
b = jnp.arange(8 * 128, dtype=jnp.float32).reshape(128, 8)
d = jnp.arange(8, dtype=jnp.float32)

c = jax.jit(linear_f32)(a, b, d)
print(c)

download_files("linear_f32")
''',
    "matmul_bf16": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def matmul_bf16(x: jnp.ndarray, w: jnp.ndarray):
    return jnp.matmul(x, w, preferred_element_type=jnp.bfloat16)

a = jnp.arange(8 * 128 * 2, dtype=jnp.bfloat16).reshape(8, 256)
b = jnp.arange(256 * 8 * 2, dtype=jnp.bfloat16).reshape(256, 8)

c = jax.jit(matmul_bf16)(a, b)
print(c)

download_files("matmul_bf16")
''',
    "matmul_f32": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def matmul_f32(x: jnp.ndarray, w: jnp.ndarray):
    return jnp.matmul(x, w, preferred_element_type=jnp.float32)

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
b = jnp.arange(128 * 8, dtype=jnp.float32).reshape(128, 8)

c = jax.jit(matmul_f32)(a, b)
print(c)

download_files("matmul_f32")
''',
    "linear_tanh_f32": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def linear_tanh_f32(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray):
    y = jnp.matmul(x, w, preferred_element_type=jnp.float32)
    y = y + b
    return jnp.tanh(y)

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
b = jnp.arange(128 * 8, dtype=jnp.float32).reshape(128, 8)
d = jnp.arange(8, dtype=jnp.float32)

c = jax.jit(linear_tanh_f32)(a, b, d)
print(c)

download_files("linear_tanh_f32")
''',
    "reduce_row_f32": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def reduce_row_f32(x: jnp.ndarray):
    return jnp.sum(x, axis=1)

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

c = jax.jit(reduce_row_f32)(a)
print(c)

download_files("reduce_row_f32")
''',
    "reduce_column_f32": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def reduce_column_f32(x: jnp.ndarray):
    return jnp.sum(x, axis=0)

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

c = jax.jit(reduce_column_f32)(a)
print(c)

download_files("reduce_column_f32")
''',
    "softmax_f32": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def softmax_f32(x: jnp.ndarray):
    return jax.nn.softmax(x)

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

c = jax.jit(softmax_f32)(a)
print(c)

download_files("softmax_f32")
''',
    "fused_nonlinear_f32": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def fused_nonlinear_f32(x: jnp.ndarray):
    # Fused nonlinear (e.g. gelu/relu) - exact op from dump
    return jax.nn.gelu(x)

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

c = jax.jit(fused_nonlinear_f32)(a)
print(c)

download_files("fused_nonlinear_f32")
''',
    "vector_add_f8": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def vector_add_f8(x: jnp.ndarray, y: jnp.ndarray):
    z = x + y
    return z

# f8e4m3 from JAX
a = jnp.arange(8 * 128, dtype=jnp.float8_e4m3fn).reshape(8, 128)
b = jnp.arange(8 * 128, dtype=jnp.float8_e4m3fn).reshape(8, 128)

c = jax.jit(vector_add_f8)(a, b)
print(c)

download_files("vector_add_f8")
''',
    "vector_broadcast_add_f32": '''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def vector_broadcast_add_f32(x: jnp.ndarray, y: jnp.ndarray):
    return x + y  # broadcast add

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
b = jnp.arange(8, dtype=jnp.float32)

c = jax.jit(vector_broadcast_add_f32)(a, b)
print(c)

download_files("vector_broadcast_add_f32")
''',
}


def _default_source(test_name: str) -> str:
    """Generic source.py when no template exists."""
    return f'''jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def {test_name}(x: jnp.ndarray):
    # Kernel from dump: {test_name}
    return x

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

c = jax.jit({test_name})(a)
print(c)

download_files("{test_name}")
'''


def _test_name_from_zip(zip_path: Path) -> str:
    """Derive test name from zip stem, e.g. vector_add_bf16_tiled -> vector_add_bf16_tiled."""
    return zip_path.stem


def _resolve_llo_dir(extract_root: Path) -> Path:
    """Find the LLO directory (content/tpu_compiler_dump/llo or tpu_compiler_dump/llo)."""
    candidates = [
        extract_root / "content" / "tpu_compiler_dump" / "llo",
        extract_root / "tpu_compiler_dump" / "llo",
        extract_root / "llo",
    ]
    for p in candidates:
        if p.is_dir() and list(p.glob("*-final_bundles.txt")):
            return p
    raise FileNotFoundError(
        f"No LLO directory with *-final_bundles.txt found under {extract_root}"
    )


def convert_dump(
    zip_path: Path,
    tests_dir: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = True,
) -> bool:
    """Convert a single dump zip to a filtered test case. Returns True on success."""
    zip_path = zip_path.resolve()
    if not zip_path.exists() or not zip_path.suffix == ".zip":
        print(f"  Skip (not a zip): {zip_path}")
        return False

    test_name = _test_name_from_zip(zip_path)
    out_dir = tests_dir / test_name
    llo_out = out_dir / "tpu_compiler_dump" / "llo"

    if not dry_run and out_dir.exists() and not overwrite:
        print(f"  Skip (exists, use --overwrite): {out_dir}")
        return False

    with tempfile.TemporaryDirectory(prefix="dump_to_test_") as tmp:
        extract_root = Path(tmp)
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)

        try:
            llo_dir = _resolve_llo_dir(extract_root)
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            return False

        tlp_ids = _discover_tlp_ids(llo_dir)
        if not tlp_ids:
            print(f"  Skip (no TLP *-79-final_bundles.txt): {zip_path}")
            return False

        try:
            keep_names = _discover_used_files_from_tlp(llo_dir)
        except (KeyError, ValueError) as e:
            print(f"  Error resolving kernels: {e}")
            return False

        kept_files = sorted(keep_names)
        kernel_count = sum(
            1
            for n in kept_files
            if re.match(r"^\d+-.+-\d+-final_bundles\.txt$", n)
            and not _TLP_FILE_RE.match(n)
        )
        tlp_count = len(tlp_ids)
        print(f"  {test_name}: {len(kept_files)} LLO files ({kernel_count} kernels, {tlp_count} TLPs)")

        if dry_run:
            for f in kept_files[:5]:
                print(f"    KEEP: {f}")
            if len(kept_files) > 5:
                print(f"    ... and {len(kept_files) - 5} more")
            return True

        llo_out.mkdir(parents=True, exist_ok=True)
        for fname in kept_files:
            src = llo_dir / fname
            if src.exists():
                shutil.copy2(src, llo_out / fname)

        # Resolve template: exact match, or base name (e.g. vector_add_bf16_tiled -> vector_add_bf16)
        source_content = SOURCE_TEMPLATES.get(test_name)
        if source_content is None:
            base = re.sub(r"_tiled$", "", test_name)
            source_content = SOURCE_TEMPLATES.get(base, _default_source(test_name))
        (out_dir / "source.py").write_text(source_content, encoding="utf-8")
        print(f"  Wrote {out_dir}/")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TPU kernel dump zips to filtered test cases in tests/.",
    )
    parser.add_argument(
        "dumps",
        type=Path,
        nargs="*",
        default=[REPO_ROOT / "dumps"],
        help="Zip file(s) or directory containing zips (default: dumps/)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        dest="no_overwrite",
        help="Do not overwrite existing test directories",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "tests",
        help="Output tests directory (default: tests/)",
    )
    args = parser.parse_args()

    tests_dir = args.output.resolve()
    dumps: list[Path] = []
    for p in args.dumps:
        p = Path(p).resolve()
        if p.is_dir():
            dumps.extend(sorted(p.glob("*.zip")))
        elif p.is_file():
            dumps.append(p)

    if not dumps:
        print("No zip files found.")
        sys.exit(1)

    print(f"Converting {len(dumps)} dump(s) to {tests_dir}/")
    if args.dry_run:
        print("(dry run)\n")

    ok = 0
    for zip_path in dumps:
        if convert_dump(
            zip_path,
            tests_dir,
            dry_run=args.dry_run,
            overwrite=not args.no_overwrite,
        ):
            ok += 1

    print(f"\nDone: {ok}/{len(dumps)} converted.")


if __name__ == "__main__":
    main()
