#!/usr/bin/env python3
"""Visualize current TPU BF16 data layouts used by this model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tpu.tiling import convert_to_bf16_tile_layout, pack_bf16_register, pack_bf16_mxu_rhs_coalesced

# Fixed output paths (project root).
_OUT_DIR = Path(__file__).resolve().parent.parent
TOY_PNG = _OUT_DIR / "toy_layouts.png"
ACTUAL_PNG = _OUT_DIR / "actual_layouts.png"


def _build_bf16_tile_index_map() -> torch.Tensor:
    logical = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
    return convert_to_bf16_tile_layout(logical, 8, 128).to(torch.int32)


def _build_toy_bf16_tile_maps() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Compute-level numbering: row pairs interleaved (row0: 0,2,4,...,14; row1: 1,3,...,15; row2: 16,18,...).
    logical = torch.empty((8, 8), dtype=torch.float32)
    for r in range(8):
        start = (r // 2) * 16 + (r % 2)
        logical[r] = torch.arange(start, start + 16, 2)
    packed = convert_to_bf16_tile_layout(logical, 8, 8).to(torch.int32)
    tiled = packed.reshape(4, 8, 2)
    return logical.to(torch.int32), tiled, packed


def _build_bf16_register_index_map() -> torch.Tensor:
    low = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
    high = low + 10_000
    packed = pack_bf16_register(low.to(torch.bfloat16), high.to(torch.bfloat16), 8, 128)
    return packed.to(torch.int32)


def _build_mxu_rhs_index_map() -> torch.Tensor:
    # Use +1 so "0" in output means empty slot; subtract back to recover source indices.
    source = torch.arange(1, 128 * 8 + 1, dtype=torch.float32).reshape(128, 8)
    packed = pack_bf16_mxu_rhs_coalesced(source.to(torch.bfloat16)).to(torch.int32)
    index_map = packed - 1
    index_map[packed == 0] = -1
    return index_map


def _import_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install it with `pip install matplotlib`."
        ) from exc
    return plt


def _annotate_grid(ax, data: np.ndarray, fontsize: int = 7) -> None:
    rows, cols = data.shape
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, str(int(data[r, c])), ha="center", va="center", fontsize=fontsize)


def _plot_toy(output_path: Path) -> None:
    plt = _import_plt()
    toy_logical, toy_tiled, toy_packed = _build_toy_bf16_tile_maps()

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), constrained_layout=True)

    im0 = axes[0].imshow(toy_logical.numpy(), aspect="equal", interpolation="nearest", cmap="Blues")
    axes[0].set_title("Toy logical view (compute-level): BF16 matrix (8x8)")
    axes[0].set_ylabel("logical row")
    axes[0].set_xlabel("logical col")
    _annotate_grid(axes[0], toy_logical.numpy())
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(toy_tiled[:, :, 0].numpy(), aspect="equal", interpolation="nearest", cmap="Blues")
    axes[1].set_title("Toy tiled view (pair component 0): tiled[4,8,2][:,:,0] = even rows")
    axes[1].set_ylabel("tile row")
    axes[1].set_xlabel("tile col")
    _annotate_grid(axes[1], toy_tiled[:, :, 0].numpy())
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(toy_tiled[:, :, 1].numpy(), aspect="equal", interpolation="nearest", cmap="Blues")
    axes[2].set_title("Toy tiled view (pair component 1): tiled[4,8,2][:,:,1] = odd rows")
    axes[2].set_ylabel("tile row")
    axes[2].set_xlabel("tile col")
    _annotate_grid(axes[2], toy_tiled[:, :, 1].numpy())
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    im3 = axes[3].imshow(toy_packed.numpy(), aspect="auto", interpolation="nearest", cmap="Blues")
    axes[3].set_title("Toy physical storage view (memory/register image): flattened packed (4x16)")
    axes[3].set_ylabel("packed row")
    axes[3].set_xlabel("packed col")
    _annotate_grid(axes[3], toy_packed.numpy())
    fig.colorbar(im3, ax=axes[3], shrink=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")


def _plot_actual(output_path: Path) -> None:
    plt = _import_plt()

    tile = _build_bf16_tile_index_map().numpy()
    reg = _build_bf16_register_index_map().numpy()
    rhs = _build_mxu_rhs_index_map().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), constrained_layout=True)

    im0 = axes[0].imshow(tile, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0].set_title("Memory/Register physical image: BF16 tile layout (logical 8x128 -> packed 4x256)")
    axes[0].set_ylabel("packed row")
    axes[0].set_xlabel("packed col")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(reg, aspect="auto", interpolation="nearest", cmap="magma")
    axes[1].set_title("Register physical image: vpack BF16 result (8x256), low rows [0:4], high rows [4:8]")
    axes[1].set_ylabel("register row")
    axes[1].set_xlabel("register col")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    rhs_ma = np.ma.masked_where(rhs < 0, rhs)
    cmap = plt.cm.cividis.copy()
    cmap.set_bad(color="#eeeeee")
    im2 = axes[2].imshow(rhs_ma, aspect="auto", interpolation="nearest", cmap=cmap)
    axes[2].set_title("Memory physical image: MXU RHS coalesced VMEM (logical 128x8 -> packed 8x256), gray=unused")
    axes[2].set_ylabel("packed row")
    axes[2].set_xlabel("packed col")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")


def main() -> None:
    _plot_toy(TOY_PNG)
    _plot_actual(ACTUAL_PNG)


if __name__ == "__main__":
    main()
