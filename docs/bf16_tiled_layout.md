# BF16 Tiled Layout Notes (TPU VREG)

This note records the BF16 register-layout behavior modeled in code.

## Source-backed behavior

- OpenXLA documents TPU BF16 tiling as repeated tiling `(8,128)(2,1)`.
- For a logical matrix with shape `[8,128]{1,0}`, BF16 packing pairs values in the second-minor dimension (sublanes), producing an equivalent tiled view `[4,128,2]{2,1,0}`.
- OpenXLA's matrix example shows pairwise row packing:
  - `packed[r, c, 0] = logical[2*r, c]`
  - `packed[r, c, 1] = logical[2*r+1, c]`

References:
- https://openxla.org/xla/tiled_layout#types_of_tiling
- https://openxla.org/xla/tiled_layout#examples_of_tiling_formats

## Mapping implemented in this repository

For one BF16 logical matrix `logical` of shape `(8,128)`:

1. Convert to tiled row pairs `(4,128,2)`.
2. Flatten the minor pair to VMEM/VREG row shape `(4,256)`.

For packed BF16 vector registers used by `vpack.c.bf16` / `vunpack.c.*.bf16`:

- Register image is modeled as `(8,256)` BF16 values.
- Rows `[0:4]` store the low BF16 tile (for `vunpack.c.l.bf16`).
- Rows `[4:8]` store the high BF16 tile (for `vunpack.c.h.bf16`).

This keeps register byte size at 4096 bytes and makes low/high unpack behavior explicit.
