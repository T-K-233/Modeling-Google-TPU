# TPU Reference Notes (March 2026)

This document records external TPU/XLA references used to refine simulator semantics and tests.

## Primary Sources

1. Cloud TPU v5e architecture
- https://cloud.google.com/tpu/docs/v5e
- Used for: chip-level components (MXU, vector unit, scalar unit), HBM capacity.

2. OpenXLA tiled layout
- https://openxla.org/xla/tiled_layout
- Used for: TPU tile shape `(8,128)` and BF16 repeated tiling `(8,128)(2,1)`.

3. OpenXLA shapes and memory spaces
- https://openxla.org/xla/shapes
- Used for: memory space IDs (`S(0)=HBM`, `S(1)=VMEM`, `S(2)+ device-specific`).

4. JAX scaling book TPU internals appendix
- https://jax-ml.github.io/scaling-book/tpus/
- Used for: lane/sublane model `(8,128)`, VREG count context, VMEM role and size notes.

5. TPU v2/v3 core architecture writeup (IEEE Micro preprint mirror)
- https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf
- Used for: vector lane/sub-lane execution framing and VLIW architecture background.

6. Hot Chips 2020 training TPU slides
- https://hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf
- Used for: bundle-slot summary (scalar/vector/matrix/misc/immediates) in scalar control model notes.

## What Was Aligned In Code

- Added missing vector ISA ops used by TPU LLO dumps (`vadd.s32`, `vand/vor/vxor`, shifts, compares, `vsel`, `vpow2`, `vrcp`, `vtanh`, `vweird`, mask ops, etc.).
- Added VM mask register modeling (`vm*`) for vector predicate/dataflow instructions.
- Fixed parser handling of temporary SSA ids like `%71` so producer/consumer chains (e.g. `vpow2 -> vpop.eup`) resolve correctly.
- Updated operand placement so VMEM inlined operands do not overlap temporary VMEM allocations.
- Extended tests for additional kernels that now have stable golden checks.

## Current Gaps

The following areas still need more reverse-engineering kernels to confidently claim full TPU-faithful behavior:

- Full `softmax_f32` output fusion (`.../1772533135288708747-fusion`) for edge-case transcendental/permute behavior.
- `fused_nonlinear_f32` transcendental sequence (`vweird`/bit-manip pipeline).
- `lane_reduce_bf16_nonzero_init` first stage (`convert_reduce_fusion`) for 16-row BF16 reduction packing/XLU pop semantics.

