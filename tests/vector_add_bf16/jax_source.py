jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def vector_add_bf16(x: jnp.ndarray, y: jnp.ndarray):
    z = x + y
    return z

a = jnp.arange(8 * 128 * 2, dtype=jnp.bfloat16).reshape(8, 256)
b = jnp.arange(8 * 128 * 2, dtype=jnp.bfloat16).reshape(8, 256)

c = jax.jit(vector_add_bf16)(a, b)
print(c)

download_files("vector_add_bf16")