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


@jax.named_call
def lane_reduce_bf16_nonzero_init(x: jax.Array, init: jax.Array) -> jax.Array:
    # Explicit BF16 scalar init helps exercise scalar->vector BF16 conversion path.
    acc = jnp.sum(x.astype(jnp.float32), axis=1)
    return (acc + init.astype(jnp.float32)).astype(jnp.bfloat16)


a = jnp.arange(16 * 128, dtype=jnp.float32).reshape(16, 128).astype(jnp.bfloat16)
init = jnp.array(3.5, dtype=jnp.bfloat16)

c = jax.jit(lane_reduce_bf16_nonzero_init)(a, init)
print(c.shape, c.dtype)
print(c)


!zip -r /content/tpu_compiler_dump.zip /content/tpu_compiler_dump/
from google.colab import files
files.download("/content/tpu_compiler_dump.zip")
