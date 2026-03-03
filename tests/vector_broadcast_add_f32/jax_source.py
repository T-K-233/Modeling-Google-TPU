jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def vector_broadcast_add_f32(x: jnp.ndarray, y: jnp.ndarray):
    z = x + y
    return z

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
b = jnp.arange(128, dtype=jnp.float32)

c = jax.jit(vector_broadcast_add_f32)(a, b)
print(c)

download_files("vector_broadcast_add_f32")