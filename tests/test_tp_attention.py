"""Minimal test: tokamax attention forward+backward with TP-sharded heads."""
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.distributed.initialize()

mesh = jax.make_mesh((jax.device_count(),), ("tp",))
jax.set_mesh(mesh)

print(f"devices={jax.device_count()}, process={jax.process_index()}/{jax.process_count()}")

B, T, H, K = 1, 128, jax.device_count() * 2, 64  # H divisible by tp
h = H  # same kv heads for simplicity

# Create TP-sharded Q, K, V
q_sharding = NamedSharding(mesh, P(None, None, "tp", None))
rng = jax.random.key(0)
q = jax.device_put(jax.random.normal(rng, (B, T, H, K), dtype=jnp.bfloat16), q_sharding)
k = jax.device_put(jax.random.normal(rng, (B, T, h, K), dtype=jnp.bfloat16), q_sharding)
v = jax.device_put(jax.random.normal(rng, (B, T, h, K), dtype=jnp.bfloat16), q_sharding)

print(f"q.shape={q.shape}, sharding={q.sharding}")
print(f"k.shape={k.shape}, sharding={k.sharding}")

from tokamax import dot_product_attention

# Test 1: Forward only (should work with q_sharding)
print("\n--- Test 1: Forward with q_sharding ---")
try:
    out = jax.jit(lambda q, k, v: dot_product_attention(
        q, k, v, is_causal=True, implementation="mosaic", q_sharding=q_sharding,
    ))(q, k, v)
    print(f"Forward OK: out.shape={out.shape}")
except Exception as e:
    print(f"Forward FAILED: {e}")

# Test 2: Forward+Backward with q_sharding (is_causal)
print("\n--- Test 2: Forward+Backward with q_sharding (is_causal) ---")
try:
    def loss_fn(q, k, v):
        out = dot_product_attention(
            q, k, v, is_causal=True, implementation="mosaic", q_sharding=q_sharding,
        )
        return out.sum()

    grad_fn = jax.jit(jax.grad(loss_fn))
    dq = grad_fn(q, k, v)
    print(f"Grad OK: dq.shape={dq.shape}")
except Exception as e:
    print(f"Grad FAILED: {type(e).__name__}: {e}")

# Test 2b: Forward+Backward with q_sharding + Mask object
print("\n--- Test 2b: Forward+Backward with q_sharding + Mask ---")
try:
    from tokamax._src.ops.attention.base import Mask
    k_start = jax.device_put(jnp.zeros(T, dtype=jnp.int32), NamedSharding(mesh, P(None)))
    k_end = jax.device_put(jnp.full(T, T, dtype=jnp.int32), NamedSharding(mesh, P(None)))
    mask = Mask(k_start=k_start, k_end=k_end)

    def loss_fn_mask(q, k, v):
        out = dot_product_attention(
            q, k, v, mask=mask, is_causal=False, implementation="mosaic", q_sharding=q_sharding,
        )
        return out.sum()

    grad_fn_mask = jax.jit(jax.grad(loss_fn_mask))
    dq_mask = grad_fn_mask(q, k, v)
    print(f"Grad OK: dq_mask.shape={dq_mask.shape}")
except Exception as e:
    import traceback; traceback.print_exc(file=__import__('sys').stdout)
    print(f"Grad FAILED: {type(e).__name__}: {e}")

# Test 3: Forward+Backward WITHOUT q_sharding (current broken path)
print("\n--- Test 3: Forward+Backward without q_sharding ---")
try:
    def loss_fn_no_sharding(q, k, v):
        out = dot_product_attention(
            q, k, v, is_causal=True, implementation="mosaic",
        )
        return out.sum()

    grad_fn2 = jax.jit(jax.grad(loss_fn_no_sharding))
    dq2 = grad_fn2(q, k, v)
    print(f"Grad OK: dq2.shape={dq2.shape}")
except Exception as e:
    print(f"Grad FAILED: {type(e).__name__}: {e}")

print("\nDone.")
