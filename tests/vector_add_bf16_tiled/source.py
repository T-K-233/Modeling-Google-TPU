jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def vector_add_bf16_tiled(x: jnp.ndarray, y: jnp.ndarray):
    z = x + y
    return z

a = jnp.arange(32 * 512, dtype=jnp.float32).reshape(32, 512)
b = jnp.arange(32 * 512, dtype=jnp.float32).reshape(32, 512)

c = jax.jit(vector_add_bf16_tiled)(a, b)
print(c)

download_files("vector_add_bf16_tiled")
