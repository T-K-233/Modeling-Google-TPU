jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def vector_add_f8(x: jnp.ndarray, y: jnp.ndarray):
    z = x + y
    return z

a = jnp.arange(8 * 128 * 4, dtype=jnp.float8_e4m3fn).reshape(8, 512)
b = jnp.arange(8 * 128 * 4, dtype=jnp.float8_e4m3fn).reshape(8, 512)

c = jax.jit(vector_add_f8)(a, b)
print(c[0:10, 0:10])

download_files("vector_add_f8")
