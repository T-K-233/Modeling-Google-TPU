jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def reduce_column_f32(x: jnp.ndarray):
    z = jnp.sum(x, axis=0)
    return z

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

c = jax.jit(reduce_column_f32)(a)
print(c)

download_files("reduce_column_f32")
