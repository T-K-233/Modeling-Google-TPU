jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def fused_nonlinear_f32(x: jnp.ndarray, y: jnp.ndarray):
    z = jnp.sin(x) * jnp.exp(y) + 0.5 * x
    return z

a = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
b = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)

c = jax.jit(fused_nonlinear_f32)(a, b)
print(c)

download_files("fused_nonlinear_f32")
