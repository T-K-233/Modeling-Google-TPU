jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def softmax_f32(x: jnp.ndarray):
    x = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=-1, keepdims=True)

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

c = jax.jit(softmax_f32)(a)
print(c)

download_files("softmax_f32")