# BF16 MXU RHS Coalesced VMEM Layout

This note documents the hardware-style change for BF16 matmul RHS handling:

- no parser-time opcode rewriting
- no runtime register provenance/tag state
- fixed instruction semantics
- correctness from physical VMEM preload layout

## Why This Change

`matmul_bf16` uses the path:

`vld (coalesced + offset) -> vunpack.c.[l|h].bf16 -> vmatpush.msra.mxu*`

With logical BF16 payload (`128x8`, 2048 bytes), this path consumes data as a
full register image. To match the instruction stream directly, the RHS operand
must be preloaded in a coalesced physical layout (`8x256` BF16, 4096 bytes).

## Layout Definition

For logical RHS `B` with shape `(128, 8)`, write into flat BF16 buffer
`P` (length `2048`, i.e. `8x256`) as:

`q = r // 2`  
`major = q % 8`  
`minor = q // 8`  
`base = major * 256 + minor * 16 + (r % 2)`  
`P[base + 2*c] = B[r, c]` for `r in [0,127], c in [0,7]`

All unspecified `P` entries are zero.

This mapping is implemented by:

- `tpu/tiling.py` -> `pack_bf16_mxu_rhs_coalesced(...)`

## Parser/Loader Behavior

`inlined_call_operand.vmem` BF16 operands with shape `[K, 8]` are modeled with
physical VMEM size at least `4096` bytes, so preloaded coalesced RHS tiles fit
without overlapping operands.

