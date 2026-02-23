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


def vadd_kernel(x_ref, y_ref, o_ref):
    x = x_ref[...]
    y = y_ref[...]
    o_ref[...] = x + y


@jax.named_call
def vadd(x: jax.Array, y: jax.Array) -> jax.Array:
    bm, bn = 8, 128
    m, n = x.shape
    assert x.shape == y.shape

    grid = (m // bm, n // bn)  # (2, 1) for (8, 128)

    return pl.pallas_call(
        vadd_kernel,
        grid=grid,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    )(x, y)


a = jnp.ones((16, 256), dtype=jnp.float32) * 2
b = jnp.ones((16, 256), dtype=jnp.float32)

c = jax.jit(vadd)(a, b)
print(c.shape, c.dtype)
print(c)


!zip -r /content/tpu_compiler_dump.zip /content/tpu_compiler_dump/
from google.colab import files
files.download("/content/tpu_compiler_dump.zip")
