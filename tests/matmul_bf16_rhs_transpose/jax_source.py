import os

""" Set up JAX Pallas environment """
DUMP_ROOT = "/content/tpu_compiler_dump/"
HLO_DUMP_PATH = os.path.join(DUMP_ROOT, "hlo")
LLO_DUMP_PATH = os.path.join(DUMP_ROOT, "llo")
MOSAIC_DUMP_PATH = os.path.join(DUMP_ROOT, "mosaic")

os.makedirs(HLO_DUMP_PATH, exist_ok=True)
os.makedirs(LLO_DUMP_PATH, exist_ok=True)
os.makedirs(MOSAIC_DUMP_PATH, exist_ok=True)

os.environ["XLA_FLAGS"] = (
    f"--xla_dump_hlo_as_text "
    f"--xla_dump_to={HLO_DUMP_PATH} "
    f"--xla_dump_hlo_pass_re=.* "
)

os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={LLO_DUMP_PATH} "
    f"--xla_jf_dump_hlo_text=true "
    f"--xla_jf_dump_llo_text=true "
    f"--xla_jf_dump_llo_html=false "
    f"--xla_jf_dump_llo_static_gaps=true "
    f"--xla_jf_emit_annotations=true "
    f"--xla_jf_debug_level=2 "
    f"--xla_mosaic_dump_to={MOSAIC_DUMP_PATH} "
    f"--xla_mosaic_enable_dump_debug_info=true "
    f"--xla_mosaic_enable_llo_source_annotations=true"
)


""" Everything else follows """
# Import JAX after setting env vars
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def matmul_rhs_transpose_kernel(x_ref, y_ref, z_ref):
    # y_ref is [8,128], transposed to [128,8] inside kernel.
    z_ref[...] = (x_ref[...].astype(jnp.float32) @ jnp.swapaxes(y_ref[...].astype(jnp.float32), 0, 1)).astype(jnp.float32)


@jax.named_call
def matmul_bf16_rhs_transpose(x: jax.Array, y: jax.Array):
    m, k = x.shape
    n, k2 = y.shape
    assert k == k2
    return pl.pallas_call(
        matmul_rhs_transpose_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
    )(x, y)


a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128).astype(jnp.bfloat16)
b = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128).astype(jnp.bfloat16)

c = jax.jit(matmul_bf16_rhs_transpose)(a, b)
print(c.shape, c.dtype)
print(c)


!zip -r /content/tpu_compiler_dump.zip /content/tpu_compiler_dump/
from google.colab import files
files.download("/content/tpu_compiler_dump.zip")
