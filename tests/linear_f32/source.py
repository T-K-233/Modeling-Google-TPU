jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def linear_f32(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray):
    y = jnp.matmul(x, w, preferred_element_type=jnp.float32)
    y = y + b
    return y

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
b = jnp.arange(8 * 128, dtype=jnp.float32).reshape(128, 8)
d = jnp.arange(8, dtype=jnp.float32)

c = jax.jit(linear_f32)(a, b, d)
print(c)

download_files("linear_f32")