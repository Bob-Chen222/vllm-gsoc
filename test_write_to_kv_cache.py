from vllm.v1.attention.backends.pallas import write_to_kv_cache
import jax
import jax.numpy as jnp


def test_write_to_kv_cache():
    num_tokens = 2
    num_heads = 4
    head_size = 8
    num_blocks = 4
    block_size = 16
    num_kv_heads = 2

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    kv_cache = jnp.zeros(num_blocks, num_kv_heads, block_size, head_size)
    key, subkey = jax.random.split(key)
    query = jax.random.normal(subkey, (num_tokens, num_heads, head_size))
    key, subkey = jax.random.split(key)
    value = jax.random.normal(subkey, (num_tokens, num_heads, head_size))

    slot_mapping = jnp.array([[0, 1], [2, 3]])
    write_to_kv_cache(
        query,
        value,
        kv_cache,
        slot_mapping,
    )
    print("kv_cache value:", kv_cache)

if __name__ == "__main__":
    test_write_to_kv_cache()
    print("Test passed successfully.")
