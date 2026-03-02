# Kernel Generation Notes (BF16 ISA Debug)

After running each `jax_source.py` on TPU and collecting LLO dumps, these are the
instruction patterns to check first:

1. `tests/vector_add_bf16_16x128/jax_source.py`
- Goal: force BF16 load unpack on a taller tile.
- Check for both `vunpack.c.l.bf16` and `vunpack.c.h.bf16`.
- Good for validating low/high half selection from one BF16 register image.

2. `tests/lane_reduce_bf16_nonzero_init/jax_source.py`
- Goal: validate scalar BF16 init path.
- Check for `vstv` + `vunpack.i.l.bf16`.
- Uses nonzero init (`3.5`) so scalar conversion errors are easy to spot.

3. `tests/matmul_bf16/jax_source.py`
- Goal: baseline BF16 input matmul with FP32 output.
- Check `vunpack.c.[l|h].bf16`, `vmatpush*`, and `vmatmul*` ordering.
- Directly comparable to the current failing `tests/matmul_bf16` case.

4. `tests/matmul_bf16_rhs_transpose/jax_source.py`
- Goal: stress transposed RHS handling.
- Useful for identifying whether matrix feed order or transpose semantics are
  mismatched in current MXU modeling.
