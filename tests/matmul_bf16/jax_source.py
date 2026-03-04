jax, jnp, partial, pl, pltpu = initialize_runtime()

@jax.named_call
def matmul_bf16(x: jnp.ndarray, y: jnp.ndarray):
    return jnp.matmul(x, y, preferred_element_type=jnp.float32)

a = jnp.arange(8 * 8, dtype=jnp.float32).reshape(8, 8)
b = jnp.arange(8 * 8, dtype=jnp.float32).reshape(8, 8)

c = jax.jit(matmul_bf16)(a, b)
print(c)

download_files("matmul_bf16")
