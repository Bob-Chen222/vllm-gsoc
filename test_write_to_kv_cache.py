# from vllm.v1.attention.backends.pallas import write_to_kv_cache
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
    kv_cache = jax.device_put(jnp.zeros((num_blocks, num_kv_heads, block_size, head_size)), jax.devices("cpu")[0])
    key, subkey = jax.random.split(key)
    query = jax.device_put(jax.random.normal(subkey, (num_tokens, num_heads, head_size)), jax.devices("cpu")[0])
    print("query is:", query)
    key, subkey = jax.random.split(key)
    value = jax.device_put(jax.random.normal(subkey, (num_tokens, num_heads, head_size)), jax.devices("cpu")[0])

    slot_mapping = jax.device_put(jnp.array([0,1,2,3]), jax.devices("cpu")[0])
    kv_cache = write_to_kv_cache(
        query,
        value,
        kv_cache,
        slot_mapping,
    )
    print("kv_cache is:", kv_cache)

def write_to_kv_cache(
    key: jax.Array,
    value: jax.Array,
    kv_cache: jax.Array,
    slot_mapping: jax.Array,
) -> None:
    """ Write the key and values to the KV cache.

    Args:
        key: shape = [num_tokens, num_kv_heads * head_size]
        value: shape = [num_tokens, num_kv_heads *  head_size]
        kv_cache = [num_blocks, block_size, num_kv_heads * 2, head_size]

    """
    _, _, num_combined_kv_heads, head_size = kv_cache.shape
    num_kv_heads = num_combined_kv_heads // 2

    key = jnp.reshape(key, (-1, num_kv_heads, head_size))
    value = jnp.reshape(value, (-1, num_kv_heads, head_size))

    kv = jnp.concatenate([key, value], axis=-1).reshape(-1, num_combined_kv_heads,
                                                  head_size)

    # torch.ops.xla.dynamo_set_buffer_donor_(kv_cache, True)

    # NOTE(Bob): need to check if this is correct
    kv_cache = kv_cache.reshape((-1,) + kv_cache.shape[2:])
    kv_cache = kv_cache.at[slot_mapping].set(kv)
    return kv_cache

if __name__ == "__main__":
    test_write_to_kv_cache()
